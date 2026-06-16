from __future__ import annotations

import pytest
import pytest_asyncio
from testcontainers.redis import RedisContainer

from actrone_memory.l1.redis_store import RedisStore
from actrone_memory.models import Turn


@pytest.fixture(scope="module")
def redis_url():
    with RedisContainer("redis:7.2-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"


@pytest_asyncio.fixture
async def store(redis_url: str) -> RedisStore:
    s = await RedisStore.from_url(redis_url, session_ttl_hours=1, max_turns=5)
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_append_and_retrieve_turn(store: RedisStore):
    turn = Turn(session_id="s1", user_message="hi", assistant_message="hello", token_count=10)
    await store.append_turn("agent-1", "s1", turn)

    turns = await store.get_recent_turns("agent-1", "s1")
    assert len(turns) == 1
    assert turns[0].user_message == "hi"
    assert turns[0].assistant_message == "hello"


@pytest.mark.asyncio
async def test_max_turns_cap(store: RedisStore):
    for i in range(7):
        turn = Turn(session_id="s2", user_message=f"q{i}", assistant_message=f"a{i}", token_count=5)
        await store.append_turn("agent-1", "s2", turn)

    turns = await store.get_recent_turns("agent-1", "s2")
    # Capped at 5 (max_turns from fixture)
    assert len(turns) == 5
    # Most recent 5 retained
    assert turns[-1].user_message == "q6"


@pytest.mark.asyncio
async def test_clear_session(store: RedisStore):
    turn = Turn(session_id="s3", user_message="x", assistant_message="y", token_count=5)
    await store.append_turn("agent-1", "s3", turn)
    await store.clear_session("agent-1", "s3")

    turns = await store.get_recent_turns("agent-1", "s3")
    assert turns == []


@pytest.mark.asyncio
async def test_session_metadata(store: RedisStore):
    turn = Turn(session_id="s4", user_message="m", assistant_message="n", token_count=5)
    await store.append_turn("agent-1", "s4", turn)

    meta = await store.get_session_metadata("agent-1", "s4")
    assert meta is not None
    assert meta.agent_id == "agent-1"
    assert meta.session_id == "s4"
    assert meta.turn_count == 1


@pytest.mark.asyncio
async def test_missing_session_metadata_returns_none(store: RedisStore):
    meta = await store.get_session_metadata("agent-1", "nonexistent-session")
    assert meta is None
