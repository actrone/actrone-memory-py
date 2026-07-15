"""Contract tests for the LangChain memory adapter (Tier-2 native ``BaseMemory``).

Skipped automatically when langchain-core is not installed.
Run with: pip install actrone-memory[langchain] && pytest tests/contract/test_langchain.py
"""
from __future__ import annotations

import pytest

langchain_core = pytest.importorskip("langchain_core", reason="langchain-core not installed")

from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.integrations.langchain import ActroneMemory
from actrone_memory.models import MemoryEntry


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[MagicMock(user_message="What is 2+2?", assistant_message="4")],
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
        )
    )
    mm.store_turn = AsyncMock(return_value="turn-1")
    mm.clear_session = AsyncMock()
    return mm


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneMemory:
    return ActroneMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)


@pytest.mark.asyncio
async def test_build_context_returns_governed_string(memory: ActroneMemory) -> None:
    # Tier-1 framework-free path: a single governed system-context block.
    context = await memory.build_context("what is 2+2?")
    assert "Arithmetic is fun." in context  # episodic memory
    assert "What is 2+2?" in context  # recent turn


@pytest.mark.asyncio
async def test_build_context_empty_when_nothing_relevant(mock_mm: AsyncMock) -> None:
    mock_mm.retrieve_context = AsyncMock(
        return_value=MagicMock(recent_turns=[], episodic_memories=[])
    )
    memory = ActroneMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)
    assert await memory.build_context("anything") == ""


def test_memory_variables_is_history(memory: ActroneMemory) -> None:
    assert memory.memory_variables == ["history"]


@pytest.mark.asyncio
async def test_load_memory_variables_returns_history_with_turns_and_memories(
    memory: ActroneMemory,
) -> None:
    out = await memory.load_memory_variables({"input": "what is 2+2?"})
    assert "history" in out
    assert "Human: What is 2+2?" in out["history"]
    assert "AI: 4" in out["history"]
    assert "Arithmetic is fun." in out["history"]


@pytest.mark.asyncio
async def test_save_context_persists_the_turn(memory: ActroneMemory, mock_mm: AsyncMock) -> None:
    await memory.save_context({"input": "hi"}, {"response": "hello"})
    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "hi", "hello")


@pytest.mark.asyncio
async def test_clear_delegates_to_manager(memory: ActroneMemory, mock_mm: AsyncMock) -> None:
    await memory.clear()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")


def test_import_error_without_langchain() -> None:
    modules = {"langchain_core": None, "langchain_core.memory": None}
    with patch.dict("sys.modules", modules), pytest.raises(
        ImportError, match=r"pip install actrone-memory\[langchain\]"
    ):
        ActroneMemory(agent_id="agent-1", session_id="default")
