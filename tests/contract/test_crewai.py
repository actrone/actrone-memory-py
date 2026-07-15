"""Contract tests for the CrewAI memory adapter.

Skipped automatically when crewai is not installed.
Run with: pip install actrone-memory[crewai] && pytest tests/contract/test_crewai.py
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

crewai = pytest.importorskip("crewai", reason="crewai not installed")

from unittest.mock import AsyncMock, patch

from actrone_memory.integrations.crewai import ActroneCrewMemory
from actrone_memory.models import MemoryEntry


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.store_turn = AsyncMock(return_value="turn-1")
    mm.clear_session = AsyncMock()
    mm.search_memories = AsyncMock(
        return_value=[
            MemoryEntry(
                agent_id="agent-1",
                session_id="default",
                content="Paris is the capital of France.",
                content_type="summary",
                importance_score=0.9,
                token_count=10,
                timestamp=datetime.now(UTC),
            )
        ]
    )
    return mm


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneCrewMemory:
    return ActroneCrewMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)


@pytest.mark.asyncio
async def test_build_context_returns_governed_string(mock_mm: AsyncMock) -> None:
    from unittest.mock import MagicMock

    # Tier-1 framework-free path: a single governed system-context block.
    mock_mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[],
            episodic_memories=[
                MemoryEntry(
                    agent_id="agent-1",
                    session_id="default",
                    content="Paris is the capital of France.",
                    content_type="summary",
                    importance_score=0.9,
                    token_count=10,
                    timestamp=datetime.now(UTC),
                )
            ],
        )
    )
    memory = ActroneCrewMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)
    context = await memory.build_context("capital of France")
    assert "Paris is the capital of France." in context


@pytest.mark.asyncio
async def test_save_persists_value_with_task_input(
    memory: ActroneCrewMemory, mock_mm: AsyncMock
) -> None:
    await memory.save("The answer is 42.", metadata={"task_input": "the question"})
    mock_mm.store_turn.assert_awaited_once()
    kwargs = mock_mm.store_turn.call_args.kwargs
    assert kwargs["user_message"] == "the question"
    assert kwargs["assistant_message"] == "The answer is 42."


@pytest.mark.asyncio
async def test_search_returns_scored_results(memory: ActroneCrewMemory) -> None:
    results = await memory.search("capital of France", limit=5)
    assert len(results) == 1
    assert results[0]["content"] == "Paris is the capital of France."
    assert results[0]["score"] == 0.9
    assert results[0]["metadata"]["content_type"] == "summary"


@pytest.mark.asyncio
async def test_reset_delegates_to_manager(memory: ActroneCrewMemory, mock_mm: AsyncMock) -> None:
    await memory.reset()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")


def test_import_error_without_crewai() -> None:
    with patch.dict("sys.modules", {"crewai": None}), pytest.raises(
        ImportError, match=r"pip install actrone-memory\[crewai\]"
    ):
        ActroneCrewMemory(agent_id="agent-1", session_id="default")
