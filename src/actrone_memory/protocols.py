from __future__ import annotations

from typing import Protocol, runtime_checkable

from actrone_memory.models import ContentType, MemoryEntry, SessionMetadata, Turn


@runtime_checkable
class L1Store(Protocol):
    """Hot session tier — recent conversation turns.

    Both the Redis-backed :class:`~actrone_memory.l1.redis_store.RedisStore` and
    the dependency-free :class:`~actrone_memory.in_memory.InMemoryStore`
    structurally satisfy this protocol, so :class:`MemoryManager` depends on the
    seam, not a concrete backend.
    """

    async def append_turn(self, agent_id: str, session_id: str, turn: Turn) -> None: ...

    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]: ...

    async def turn_count(self, agent_id: str, session_id: str) -> int: ...

    async def clear_session(self, agent_id: str, session_id: str) -> None: ...

    async def get_session_metadata(
        self, agent_id: str, session_id: str
    ) -> SessionMetadata | None: ...

    async def try_acquire_summary_lock(
        self, agent_id: str, session_id: str, ttl_seconds: int
    ) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class L2Store(Protocol):
    """Cold semantic tier — long-term episodic memories.

    Implemented by :class:`~actrone_memory.l2.qdrant_store.QdrantStore` and the
    dependency-free :class:`~actrone_memory.in_memory.InMemoryStore`.
    """

    async def upsert(self, entry: MemoryEntry) -> None: ...

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        threshold: float,
        limit: int = 20,
        content_types: list[ContentType] | None = None,
    ) -> list[MemoryEntry]: ...

    async def delete(self, memory_id: str) -> None: ...

    async def delete_agent_memories(self, agent_id: str) -> None: ...

    async def close(self) -> None: ...
