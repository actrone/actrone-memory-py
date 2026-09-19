from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import AuthenticationError, AuthorizationError, RedisError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from actrone_memory.exceptions import StoreConnectionError
from actrone_memory.logging import bind_logger
from actrone_memory.metrics import track_store_op
from actrone_memory.models import SessionMetadata, Turn

log = bind_logger(__name__)

# Transient failures worth another attempt. `BusyLoadingError` and
# `MaxConnectionsError` subclass ConnectionError, so both are covered.
# Permanent faults (unknown command, bad argument type) are excluded: retrying
# them just delays the error and hides the root cause.
_TRANSIENT_REDIS_ERRORS: tuple[type[BaseException], ...] = (
    RedisConnectionError,
    RedisTimeoutError,
)

# redis-py models rejected credentials and insufficient permissions as
# ConnectionError subclasses, but neither resolves itself between attempts, so
# they are carved back out of the transient set above.
_PERMANENT_REDIS_ERRORS: tuple[type[BaseException], ...] = (
    AuthenticationError,
    AuthorizationError,
)


def _is_transient_redis_failure(exc: BaseException) -> bool:
    """Retry predicate that looks through the store's own error wrapper.

    Every method below converts a driver error into ``StoreConnectionError``
    before returning, so a predicate matching only raw ``redis`` types would
    never match and the retry policy would silently never fire. The original
    driver exception is preserved on ``StoreConnectionError.cause``.
    """
    cause = exc.cause if isinstance(exc, StoreConnectionError) else exc
    if isinstance(cause, _PERMANENT_REDIS_ERRORS):
        return False
    return isinstance(cause, _TRANSIENT_REDIS_ERRORS)


# Retry transient Redis errors: 3 attempts, exponential backoff 0.5s to 5s with jitter.
_REDIS_RETRY = retry(
    retry=retry_if_exception(_is_transient_redis_failure),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=5),
    reraise=True,
)


class RedisStore:
    """Redis L1 hot memory store.

    Target read latency: < 1 ms P99.
    All operations retry up to 3 times on transient RedisError with exponential
    backoff + jitter before raising StoreConnectionError to the caller.
    """

    def __init__(self, client: Redis, session_ttl_hours: int, max_turns: int) -> None:
        self._r = client
        self._ttl = timedelta(hours=session_ttl_hours)
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _turns_key(agent_id: str, session_id: str) -> str:
        return f"agent:{agent_id}:session:{session_id}:turns"

    @staticmethod
    def _meta_key(agent_id: str, session_id: str) -> str:
        return f"agent:{agent_id}:session:{session_id}:meta"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_REDIS_RETRY
    @track_store_op("l1", "append_turn")
    async def append_turn(self, agent_id: str, session_id: str, turn: Turn) -> None:
        """Append a turn to the session list and refresh the TTL.

        Uses a pipeline to execute rpush + ltrim + expire atomically in one round-trip.
        The list is capped at max_turns; oldest entries are evicted by ltrim.
        """
        key = self._turns_key(agent_id, session_id)
        meta_key = self._meta_key(agent_id, session_id)
        ttl_seconds = int(self._ttl.total_seconds())

        try:
            pipe = self._r.pipeline()
            pipe.rpush(key, turn.model_dump_json())
            # O(1), ltrim keeps only the newest max_turns entries
            pipe.ltrim(key, -self._max_turns, -1)
            pipe.expire(key, ttl_seconds)

            now_iso = datetime.now(UTC).isoformat()
            meta: dict[str, str] = {
                "agent_id": agent_id,
                "session_id": session_id,
                "last_active": now_iso,
            }
            # redis-py async methods are typed `Awaitable[T] | T` (shared sync/async
            # surface), so awaiting them trips mypy's [misc] union-await check.
            existing_created = await self._r.hget(meta_key, "created_at")  # type: ignore[misc]
            if not existing_created:
                meta["created_at"] = now_iso
            pipe.hset(meta_key, mapping=meta)
            pipe.hincrby(meta_key, "turn_count", 1)
            pipe.expire(meta_key, ttl_seconds)
            await pipe.execute()
        except RedisError as exc:
            log.error(
                "redis.append_turn.failed", agent_id=agent_id, session_id=session_id, error=str(exc)
            )
            raise StoreConnectionError("Redis", exc) from exc

    @_REDIS_RETRY
    @track_store_op("l1", "get_recent_turns")
    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        """Fetch the n most recent turns for a session. O(n) list range read."""
        key = self._turns_key(agent_id, session_id)
        count = n if n is not None else self._max_turns
        try:
            raw: list[bytes] = await self._r.lrange(key, -count, -1)  # type: ignore[misc]
            return [Turn.model_validate(json.loads(item)) for item in raw]
        except RedisError as exc:
            log.error(
                "redis.get_turns.failed", agent_id=agent_id, session_id=session_id, error=str(exc)
            )
            raise StoreConnectionError("Redis", exc) from exc

    @_REDIS_RETRY
    @track_store_op("l1", "get_session_metadata")
    async def get_session_metadata(self, agent_id: str, session_id: str) -> SessionMetadata | None:
        """Return metadata for a session, or None if the session does not exist or has expired."""
        meta_key = self._meta_key(agent_id, session_id)
        try:
            data = await self._r.hgetall(meta_key)  # type: ignore[misc]
            if not data:
                return None
            decoded = {k.decode(): v.decode() for k, v in data.items()}
            return SessionMetadata(
                agent_id=decoded.get("agent_id", agent_id),
                session_id=decoded.get("session_id", session_id),
                turn_count=int(decoded.get("turn_count", 0)),
                created_at=datetime.fromisoformat(decoded["created_at"])
                if "created_at" in decoded
                else None,
                last_active=datetime.fromisoformat(decoded["last_active"])
                if "last_active" in decoded
                else None,
            )
        except RedisError as exc:
            raise StoreConnectionError("Redis", exc) from exc

    @_REDIS_RETRY
    @track_store_op("l1", "clear_session")
    async def clear_session(self, agent_id: str, session_id: str) -> None:
        """Delete all short-term memory for a session. Does not affect L2 Qdrant memories."""
        try:
            await self._r.delete(
                self._turns_key(agent_id, session_id),
                self._meta_key(agent_id, session_id),
            )
        except RedisError as exc:
            raise StoreConnectionError("Redis", exc) from exc

    @_REDIS_RETRY
    @track_store_op("l1", "turn_count")
    async def turn_count(self, agent_id: str, session_id: str) -> int:
        """Return the number of turns currently in the session list."""
        try:
            return await self._r.llen(self._turns_key(agent_id, session_id))  # type: ignore[no-any-return, misc]
        except RedisError as exc:
            raise StoreConnectionError("Redis", exc) from exc

    @_REDIS_RETRY
    @track_store_op("l1", "try_acquire_summary_lock")
    async def try_acquire_summary_lock(
        self, agent_id: str, session_id: str, ttl_seconds: int
    ) -> bool:
        """Atomically claim the right to summarise a session (SET key NX EX ttl).

        Returns True for the single caller that wins the race; subsequent callers
        get False until the key expires after ``ttl_seconds``. The TTL is never
        released early, so it doubles as a re-summarisation cooldown and works
        across every process in the fleet, not just within one event loop.
        """
        key = f"agent:{agent_id}:session:{session_id}:summary_lock"
        try:
            acquired = await self._r.set(key, "1", nx=True, ex=ttl_seconds)
            return bool(acquired)
        except RedisError as exc:
            raise StoreConnectionError("Redis", exc) from exc

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def from_url(
        cls,
        url: str,
        session_ttl_hours: int,
        max_turns: int,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
    ) -> RedisStore:
        """Connect to Redis with an explicit connection pool.

        Args:
            url: Redis connection URL (e.g. redis://localhost:6379).
            session_ttl_hours: How long sessions persist before expiring.
            max_turns: Maximum turns stored per session.
            max_connections: Connection pool size. Default 10 is suitable for
                most single-service deployments. Raise under high concurrency.
            socket_timeout: Seconds before a socket operation times out.
            socket_connect_timeout: Seconds before a connection attempt times out.
        """
        try:
            pool = ConnectionPool.from_url(
                url,
                decode_responses=False,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
            )
            client: Redis = Redis(connection_pool=pool)
            await client.ping()
            return cls(client, session_ttl_hours, max_turns)
        except RedisError as exc:
            raise StoreConnectionError("Redis", exc) from exc

    async def close(self) -> None:
        await self._r.aclose()
