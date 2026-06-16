"""Contract tests for the LlamaIndex adapter.

Skipped automatically when llama-index-core is not installed.
Run with: pip install actrone-memory[llamaindex] && pytest tests/contract/test_llamaindex.py
"""
from __future__ import annotations

import pytest

llama_index = pytest.importorskip("llama_index", reason="llama-index-core not installed")

from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.integrations.llamaindex import ActroneLlamaMemory
from actrone_memory.models import MemoryEntry, Turn


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.store_turn = AsyncMock(return_value="turn-id-1")
    mm.clear_session = AsyncMock()
    mm.retrieve_context = AsyncMock(return_value=MagicMock(
        recent_turns=[
            MagicMock(user_message="What is 2+2?", assistant_message="4"),
        ],
        episodic_memories=[
            MemoryEntry(
                agent_id="agent-1",
                session_id="default",
                content="Arithmetic is fun.",
                content_type="summary",
                importance_score=0.5,
                token_count=5,
            )
        ],
    ))
    return mm


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneLlamaMemory:
    return ActroneLlamaMemory(
        agent_id="agent-1",
        session_id="default",
        memory_manager=mock_mm,
    )


@pytest.mark.asyncio
async def test_aget_returns_chat_messages(memory: ActroneLlamaMemory) -> None:
    from llama_index.core.base.llms.types import MessageRole

    messages = await memory.aget(input="what is 2+2?")

    # Should include system message with episodic memory + 2 turn messages.
    assert len(messages) >= 3
    roles = [m.role for m in messages]
    assert MessageRole.SYSTEM in roles
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles


@pytest.mark.asyncio
async def test_aput_user_message_buffered(memory: ActroneLlamaMemory, mock_mm: AsyncMock) -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    await memory.aput(ChatMessage(role=MessageRole.USER, content="Hello"))
    # Not yet stored — waiting for assistant response.
    mock_mm.store_turn.assert_not_awaited()
    assert memory._pending_user_msg == "Hello"


@pytest.mark.asyncio
async def test_aput_full_turn_stores(memory: ActroneLlamaMemory, mock_mm: AsyncMock) -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    await memory.aput(ChatMessage(role=MessageRole.USER, content="Hello"))
    await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"))

    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "Hello", "Hi there!")
    assert memory._pending_user_msg == ""


@pytest.mark.asyncio
async def test_areset_clears_session(memory: ActroneLlamaMemory, mock_mm: AsyncMock) -> None:
    await memory.areset()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")
    assert memory._pending_user_msg == ""


@pytest.mark.asyncio
async def test_aget_all_delegates_to_aget(memory: ActroneLlamaMemory) -> None:
    messages = await memory.aget_all()
    assert isinstance(messages, list)


def test_import_error_without_llamaindex() -> None:
    with patch.dict("sys.modules", {"llama_index": None, "llama_index.core": None, "llama_index.core.memory": None}):
        with pytest.raises(ImportError, match="pip install actrone-memory\\[llamaindex\\]"):
            ActroneLlamaMemory(agent_id="agent-1")
