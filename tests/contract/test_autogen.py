"""Contract tests for the AutoGen 0.4 adapter.

Skipped automatically when autogen-core is not installed.
Run with: pip install actrone-memory[autogen] && pytest tests/contract/test_autogen.py
"""
from __future__ import annotations

import pytest

autogen_core = pytest.importorskip("autogen_core", reason="autogen-core not installed")

from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.integrations.autogen import ActroneAutoGenMemory
from actrone_memory.models import MemoryEntry


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.search_memories = AsyncMock(return_value=[
        MemoryEntry(
            agent_id="agent-1",
            session_id="default",
            content="Paris is the capital of France.",
            content_type="summary",
            importance_score=0.9,
            token_count=10,
        )
    ])
    mm.retrieve_context = AsyncMock(return_value=MagicMock(
        recent_turns=[], episodic_memories=[]
    ))
    mm.inject_memory = AsyncMock(return_value="mem-id-1")
    mm.clear_session = AsyncMock()
    return mm


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneAutoGenMemory:
    return ActroneAutoGenMemory(
        agent_id="agent-1",
        session_id="default",
        memory_manager=mock_mm,
    )


@pytest.mark.asyncio
async def test_build_context_returns_governed_string(mock_mm: AsyncMock) -> None:
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
                )
            ],
        )
    )
    memory = ActroneAutoGenMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)
    context = await memory.build_context("capital of France")
    assert "Paris is the capital of France." in context


@pytest.mark.asyncio
async def test_add_text_content(memory: ActroneAutoGenMemory, mock_mm: AsyncMock) -> None:
    from autogen_core.memory import MemoryContent, MemoryMimeType

    content = MemoryContent(
        content="The Eiffel Tower is in Paris.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"importance": 0.8},
    )
    await memory.add(content)
    mock_mm.inject_memory.assert_awaited_once()
    call_kwargs = mock_mm.inject_memory.call_args
    assert call_kwargs[0][1] == "The Eiffel Tower is in Paris."


@pytest.mark.asyncio
async def test_query_returns_memory_query_result(memory: ActroneAutoGenMemory) -> None:
    from autogen_core.memory import MemoryMimeType, MemoryQuery, MemoryQueryResult

    query = MemoryQuery(text="capital of France", n_results=5)
    result = await memory.query(query)

    assert isinstance(result, MemoryQueryResult)
    assert len(result.results) == 1
    assert result.results[0].content == "Paris is the capital of France."
    assert result.results[0].mime_type == MemoryMimeType.TEXT


@pytest.mark.asyncio
async def test_clear_delegates_to_manager(memory: ActroneAutoGenMemory, mock_mm: AsyncMock) -> None:
    await memory.clear()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")


@pytest.mark.asyncio
async def test_close_is_noop(memory: ActroneAutoGenMemory) -> None:
    # close() should not raise and should not call any manager method.
    await memory.close()


@pytest.mark.asyncio
async def test_add_non_text_content_is_noop(
    memory: ActroneAutoGenMemory, mock_mm: AsyncMock
) -> None:
    from autogen_core.memory import MemoryContent, MemoryMimeType

    content = MemoryContent(
        content=b"\x89PNG",  # binary image — not supported
        mime_type=MemoryMimeType.IMAGE,
        metadata={},
    )
    await memory.add(content)
    mock_mm.inject_memory.assert_not_awaited()


def test_import_error_without_autogen() -> None:
    """Verify ImportError is raised with a helpful message when autogen is absent."""
    modules = {"autogen_core": None, "autogen_core.memory": None}
    with patch.dict("sys.modules", modules), pytest.raises(
        ImportError, match=r"pip install actrone-memory\[autogen\]"
    ):
        ActroneAutoGenMemory(agent_id="agent-1")
