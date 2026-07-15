"""Contract tests for the LangGraph checkpointer adapter.

Skipped automatically when langgraph is not installed.
Run with: pip install actrone-memory[langgraph] && pytest tests/contract/test_langgraph.py
"""
from __future__ import annotations

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.integrations.langgraph import ActroneCheckpointer


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.store_turn = AsyncMock(return_value="turn-1")
    mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[MagicMock(user_message="hi", assistant_message="hello")]
        )
    )
    return mm


@pytest.fixture
def checkpointer(mock_mm: AsyncMock) -> ActroneCheckpointer:
    return ActroneCheckpointer(agent_id="agent-1", memory_manager=mock_mm)


@pytest.mark.asyncio
async def test_build_context_threads_thread_id_and_renders(mock_mm: AsyncMock) -> None:
    # Tier-1 framework-free path: thread_id is the session scope for recency.
    mock_mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[MagicMock(user_message="hi", assistant_message="hello")],
            episodic_memories=[],
        )
    )
    cp = ActroneCheckpointer(agent_id="agent-1", memory_manager=mock_mm)
    context = await cp.build_context("recent conversation", thread_id="t1")
    assert "hi" in context and "hello" in context
    assert mock_mm.retrieve_context.call_args[0][1] == "t1"


@pytest.mark.asyncio
async def test_aput_persists_last_two_messages_as_a_turn(
    checkpointer: ActroneCheckpointer, mock_mm: AsyncMock
) -> None:
    config = {"configurable": {"thread_id": "t1"}}
    checkpoint = {"id": "ckpt-1", "channel_values": {"messages": ["u-msg", "a-msg"]}}
    out = await checkpointer.aput(config, checkpoint, {}, {})
    mock_mm.store_turn.assert_awaited_once_with("agent-1", "t1", "u-msg", "a-msg")
    assert out["configurable"]["checkpoint_id"] == "ckpt-1"


@pytest.mark.asyncio
async def test_aget_returns_channel_values_from_recent_turns(
    checkpointer: ActroneCheckpointer,
) -> None:
    out = await checkpointer.aget({"configurable": {"thread_id": "t1"}})
    assert out is not None
    messages = out["channel_values"]["messages"]
    assert {"role": "user", "content": "hi"} in messages
    assert {"role": "assistant", "content": "hello"} in messages


@pytest.mark.asyncio
async def test_aget_returns_none_when_no_turns(mock_mm: AsyncMock) -> None:
    mock_mm.retrieve_context = AsyncMock(return_value=MagicMock(recent_turns=[]))
    checkpointer = ActroneCheckpointer(agent_id="agent-1", memory_manager=mock_mm)
    assert await checkpointer.aget({"configurable": {"thread_id": "t1"}}) is None


def test_import_error_without_langgraph() -> None:
    modules = {"langgraph": None, "langgraph.checkpoint.base": None}
    with patch.dict("sys.modules", modules), pytest.raises(
        ImportError, match=r"pip install actrone-memory\[langgraph\]"
    ):
        ActroneCheckpointer(agent_id="agent-1")
