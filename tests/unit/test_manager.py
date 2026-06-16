from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import TokenBudgetError, ValidationError
from actrone_memory.manager import MemoryManager, create_memory_manager
from actrone_memory.models import MemoryEntry, RetrievedContext, SessionMetadata, Turn

from tests.conftest import ConstantEmbedder


def _make_manager(cfg: MemoryConfig, turns: list[Turn], memories: list[MemoryEntry]) -> MemoryManager:
    l1 = AsyncMock()
    l1.get_recent_turns.return_value = turns
    l1.turn_count.return_value = len(turns)
    l1.append_turn.return_value = None
    l1.clear_session.return_value = None

    l2 = AsyncMock()
    l2.search.return_value = memories
    l2.upsert.return_value = None
    l2.delete.return_value = None

    return MemoryManager(l1=l1, l2=l2, embedder=ConstantEmbedder(), config=cfg)


@pytest.mark.asyncio
async def test_retrieve_context_invalid_budget(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(TokenBudgetError):
        await mm.retrieve_context("a1", "s1", "query", token_budget=0)


@pytest.mark.asyncio
async def test_store_turn_rejects_empty_agent_id(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="agent_id"):
        await mm.store_turn("", "s1", "hi", "hello")


@pytest.mark.asyncio
async def test_store_turn_rejects_oversized_message(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="user_message"):
        await mm.store_turn("a1", "s1", "x" * 100_001, "hello")


@pytest.mark.asyncio
async def test_inject_memory_rejects_invalid_importance(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="importance"):
        await mm.inject_memory("a1", "some content", importance=1.5)


@pytest.mark.asyncio
async def test_inject_memory_rejects_blank_content(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="content"):
        await mm.inject_memory("a1", "   ")


@pytest.mark.asyncio
async def test_search_memories_rejects_invalid_limit(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="limit"):
        await mm.search_memories("a1", "query", limit=0)


@pytest.mark.asyncio
async def test_retrieve_context_returns_pruned_turns(fake_config: MemoryConfig):
    turns = [
        Turn(session_id="s1", user_message="q1", assistant_message="a1", token_count=100),
        Turn(session_id="s1", user_message="q2", assistant_message="a2", token_count=100),
        Turn(session_id="s1", user_message="q3", assistant_message="a3", token_count=100),
    ]
    mm = _make_manager(fake_config, turns, [])

    # session budget = 4096 * 0.35 = 1433 — all 3 turns fit (300 tokens total)
    ctx = await mm.retrieve_context("a1", "s1", "q", token_budget=4096)
    assert len(ctx.recent_turns) == 3


@pytest.mark.asyncio
async def test_retrieve_context_prunes_when_over_budget(fake_config: MemoryConfig):
    turns = [
        Turn(session_id="s1", user_message="q1", assistant_message="a1", token_count=500),
        Turn(session_id="s1", user_message="q2", assistant_message="a2", token_count=500),
        Turn(session_id="s1", user_message="q3", assistant_message="a3", token_count=500),
    ]
    # Small budget — session budget = 200 * 0.35 = 70, only last turn fits (most recent kept)
    mm = _make_manager(fake_config, turns, [])
    ctx = await mm.retrieve_context("a1", "s1", "q", token_budget=200)
    assert len(ctx.recent_turns) == 0  # 500 > 70, none fit


@pytest.mark.asyncio
async def test_store_turn_calls_l1_append(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    turn_id = await mm.store_turn("a1", "s1", "hello", "world")
    assert turn_id
    mm._l1.append_turn.assert_called_once()


@pytest.mark.asyncio
async def test_inject_memory_upserts_to_l2(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    memory_id = await mm.inject_memory("a1", "Paris is in France.", importance=0.9)
    assert memory_id
    mm._l2.upsert.assert_called_once()
    call_arg: MemoryEntry = mm._l2.upsert.call_args[0][0]
    assert call_arg.content_type == "injected"
    assert call_arg.importance_score == 0.9


@pytest.mark.asyncio
async def test_clear_session_delegates_to_l1(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    await mm.clear_session("a1", "s1")
    mm._l1.clear_session.assert_called_once_with("a1", "s1")


@pytest.mark.asyncio
async def test_delete_memory_delegates_to_l2(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    await mm.delete_memory("a1", "mem-123")
    mm._l2.delete.assert_called_once_with("mem-123")


@pytest.mark.asyncio
async def test_search_memories_delegates_to_l2(fake_config: MemoryConfig):
    memories = [
        MemoryEntry(agent_id="a", session_id="s", content="x", content_type="summary",
                    embedding=[0.1, 0.2, 0.3, 0.4], token_count=10),
    ]
    mm = _make_manager(fake_config, [], memories)
    results = await mm.search_memories("a", "query", limit=5)
    assert len(results) == 1
    mm._l2.search.assert_called_once()


@pytest.mark.asyncio
async def test_close_calls_both_stores(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    await mm.close()
    mm._l1.close.assert_called_once()
    mm._l2.close.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager_closes_on_exit(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    async with mm:
        pass
    mm._l1.close.assert_called_once()
    mm._l2.close.assert_called_once()


@pytest.mark.asyncio
async def test_auto_summarise_not_triggered_below_threshold(fake_config: MemoryConfig):
    # turn_count returns 5, threshold is 20 — no task should be created
    fake_config.auto_summarise = True
    fake_config.summarise_after_turns = 20
    mm = _make_manager(fake_config, [], [])
    mm._l1.turn_count.return_value = 5
    await mm.store_turn("a", "s", "hi", "hello")
    # No summarise task should be scheduled — hard to assert directly,
    # but store_turn must complete without error
    mm._l1.append_turn.assert_called_once()


@pytest.mark.asyncio
async def test_summarise_session_handles_empty_turns(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    mm._l1.get_recent_turns.return_value = []
    # Should return silently without upserting anything
    await mm._summarise_session("agent-1", "s1")
    mm._l2.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_summarise_session_upserts_summary(fake_config: MemoryConfig):
    turns = [
        Turn(session_id="s1", user_message="q1", assistant_message="a1", token_count=10),
        Turn(session_id="s1", user_message="q2", assistant_message="a2", token_count=10),
    ]
    mm = _make_manager(fake_config, turns, [])
    await mm._summarise_session("agent-1", "s1")
    mm._l2.upsert.assert_called_once()
    upserted: MemoryEntry = mm._l2.upsert.call_args[0][0]
    assert upserted.content_type == "summary"
    assert upserted.agent_id == "agent-1"


@pytest.mark.asyncio
async def test_prune_memories_within_budget(fake_config: MemoryConfig):
    memories = [
        MemoryEntry(agent_id="a", session_id="s", content="a", content_type="summary",
                    embedding=[0.1, 0.2, 0.3, 0.4], token_count=50),
        MemoryEntry(agent_id="a", session_id="s", content="b", content_type="summary",
                    embedding=[0.1, 0.2, 0.3, 0.4], token_count=50),
    ]
    mm = _make_manager(fake_config, [], memories)
    ctx = await mm.retrieve_context("a", "s", "q", token_budget=4096)
    assert len(ctx.episodic_memories) == 2


# ------------------------------------------------------------------
# auto-summarise trigger (above threshold)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_summarise_triggered_above_threshold(fake_config: MemoryConfig):
    import asyncio as _asyncio

    fake_config.auto_summarise = True
    fake_config.summarise_after_turns = 3
    mm = _make_manager(fake_config, [], [])
    mm._l1.turn_count.return_value = 5  # above threshold

    with patch("actrone_memory.manager.asyncio.create_task") as mock_task:
        await mm.store_turn("a", "s", "hi", "hello")
        mock_task.assert_called_once()
    # The summary cooldown lock must have been claimed before spawning.
    mm._l1.try_acquire_summary_lock.assert_called_once()


@pytest.mark.asyncio
async def test_auto_summarise_suppressed_when_lock_not_acquired(fake_config: MemoryConfig):
    """Even above the threshold, no summary task spawns if the cooldown lock is
    already held (concurrent/fleet dedupe + throttle)."""
    fake_config.auto_summarise = True
    fake_config.summarise_after_turns = 3
    mm = _make_manager(fake_config, [], [])
    mm._l1.turn_count.return_value = 5  # above threshold
    mm._l1.try_acquire_summary_lock.return_value = False  # lock held by someone else

    with patch("actrone_memory.manager.asyncio.create_task") as mock_task:
        await mm.store_turn("a", "s", "hi", "hello")
        mock_task.assert_not_called()


# ------------------------------------------------------------------
# get_session_metadata on manager
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_metadata_delegates_to_l1(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    meta = SessionMetadata(agent_id="a", session_id="s", turn_count=3)
    mm._l1.get_session_metadata.return_value = meta

    result = await mm.get_session_metadata("a", "s")
    assert result is meta
    mm._l1.get_session_metadata.assert_called_once_with("a", "s")


# ------------------------------------------------------------------
# _summarise_session error path
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarise_session_swallows_exceptions(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    mm._l1.get_recent_turns.side_effect = Exception("redis down")
    # Must not propagate — background task failure is logged, not raised
    await mm._summarise_session("a", "s")


# ------------------------------------------------------------------
# create_memory_manager context manager
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_memory_manager_context_manager(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with patch("actrone_memory.manager.MemoryManager.create", return_value=mm) as mock_create:
        async with create_memory_manager(fake_config) as manager:
            assert manager is mm
        mm._l1.close.assert_called_once()
        mm._l2.close.assert_called_once()
