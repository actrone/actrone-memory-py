"""Unit tests for the framework-free Tier-1 context helpers.

These render governed memory into a system-prompt string and import no agent framework, so they
run in the base venv against a real local (service-free) ``MemoryManager``. They cover the shared
logic behind every adapter's ``build_context`` (both the conversational ``governed_context`` path
and the retrieval-shaped ``format_memories`` path).
"""

from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import (
    format_memories,
    format_memory_context,
    governed_context,
)
from actrone_memory.manager import MemoryManager
from actrone_memory.models import MemoryEntry, RetrievedContext


def _entry(content: str, score: float = 0.8) -> MemoryEntry:
    return MemoryEntry(
        agent_id="agent-1",
        session_id="default",
        content=content,
        content_type="summary",
        importance_score=score,
        token_count=len(content.split()),
    )


def test_format_memories_empty_is_empty_string() -> None:
    assert format_memories([]) == ""


def test_format_memories_renders_bulleted_block() -> None:
    block = format_memories([_entry("Paris is the capital of France."), _entry("Water boils.")])
    assert block.startswith("Relevant long-term memory:\n")
    assert "- Paris is the capital of France." in block
    assert "- Water boils." in block


def _ctx(**overrides: object) -> RetrievedContext:
    base: dict[str, object] = {
        "recent_turns": [],
        "episodic_memories": [],
        "total_tokens_used": 0,
        "token_budget": 4096,
        "retrieval_duration_ms": 0.0,
    }
    base.update(overrides)
    return RetrievedContext(**base)  # type: ignore[arg-type]


def test_format_memory_context_empty_when_nothing_relevant() -> None:
    assert format_memory_context(_ctx()) == ""


def test_format_memory_context_orders_memory_then_conversation() -> None:
    block = format_memory_context(_ctx(episodic_memories=[_entry("On-call is paged for Sev1.")]))
    assert "Relevant long-term memory:" in block
    assert "- On-call is paged for Sev1." in block


@pytest.mark.asyncio
async def test_governed_context_empty_then_populated() -> None:
    # relevance_threshold low enough that the injected memory surfaces for a loosely-related query.
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    try:
        assert await governed_context(mm, "agent-1", "s1", "anything at all") == ""

        await mm.inject_memory("agent-1", "Production deploys are frozen on Fridays.", 0.9)
        context = await governed_context(mm, "agent-1", "s1", "can we deploy on friday")
        assert "Production deploys are frozen on Fridays." in context
    finally:
        await mm.close()
