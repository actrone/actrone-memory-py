"""Contract tests for the DSPy Retrieve adapter.

Skipped automatically when dspy-ai is not installed.
Run with: pip install actrone-memory[dspy] && pytest tests/contract/test_dspy.py
"""
from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy", reason="dspy-ai not installed")

from unittest.mock import AsyncMock, patch

from actrone_memory.integrations.dspy import ActroneRM
from actrone_memory.models import MemoryEntry


def _make_memory(content: str) -> MemoryEntry:
    return MemoryEntry(
        agent_id="agent-1",
        session_id="default",
        content=content,
        content_type="summary",
        importance_score=0.8,
        token_count=len(content.split()),
    )


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.search_memories = AsyncMock(return_value=[
        _make_memory("Paris is the capital of France."),
        _make_memory("France is in Western Europe."),
    ])
    mm.store_turn = AsyncMock(return_value="turn-id-1")
    return mm


@pytest.fixture
def rm(mock_mm: AsyncMock) -> ActroneRM:
    return ActroneRM(agent_id="agent-1", k=5, memory_manager=mock_mm)


def test_forward_single_query_returns_prediction(rm: ActroneRM) -> None:
    prediction = rm.forward("capital of France")

    assert hasattr(prediction, "passages")
    assert isinstance(prediction.passages, list)
    assert len(prediction.passages) == 2
    assert "Paris" in prediction.passages[0]


def test_forward_list_of_queries_interleaves(rm: ActroneRM, mock_mm: AsyncMock) -> None:
    # Both queries return the same two passages from our mock.
    prediction = rm.forward(["capital of France", "location of France"])

    assert hasattr(prediction, "passages")
    # Interleaving: [q1[0], q2[0], q1[1], q2[1]] = 4 passages.
    assert len(prediction.passages) == 4


def test_call_syntax(rm: ActroneRM) -> None:
    """Verify rm("query") shorthand works."""
    prediction = rm("what is France?")
    assert hasattr(prediction, "passages")


def test_forward_respects_k_override(rm: ActroneRM, mock_mm: AsyncMock) -> None:
    rm.forward("test", k=3)
    # search_memories should have been called with limit=3.
    call_args = mock_mm.search_memories.call_args
    assert call_args[1]["limit"] == 3 or call_args[0][-1] == 3


@pytest.mark.asyncio
async def test_store_turn_delegates_to_manager(rm: ActroneRM, mock_mm: AsyncMock) -> None:
    await rm.store_turn("agent-1", "sess-1", "Hello", "Hi!")
    mock_mm.store_turn.assert_awaited_once_with("agent-1", "sess-1", "Hello", "Hi!")


@pytest.mark.asyncio
async def test_search_memories_returns_strings(rm: ActroneRM) -> None:
    results = await rm.search_memories("France", limit=2)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
    assert len(results) == 2


def test_default_k_is_five() -> None:
    rm = ActroneRM.__new__(ActroneRM)
    rm.k = 5
    assert rm.k == 5


def test_import_error_without_dspy() -> None:
    with patch.dict("sys.modules", {"dspy": None}):
        with pytest.raises(ImportError, match="pip install actrone-memory\\[dspy\\]"):
            ActroneRM(agent_id="agent-1")
