from __future__ import annotations

import pytest
from pydantic import ValidationError

from actrone_memory.models import MemoryEntry, RetrievedContext, Turn


def test_memory_entry_defaults_are_set():
    entry = MemoryEntry(
        agent_id="a1",
        session_id="s1",
        content="hello",
        content_type="turn",
    )
    assert entry.id  # UUID auto-generated
    assert entry.importance_score == 0.5
    assert entry.topic_tags == []
    assert entry.source_turn_ids == []


def test_memory_entry_importance_clamps():
    with pytest.raises(ValidationError):
        MemoryEntry(
            agent_id="a",
            session_id="s",
            content="x",
            content_type="turn",
            importance_score=1.5,
        )


def test_memory_entry_invalid_content_type():
    with pytest.raises(ValidationError):
        MemoryEntry(agent_id="a", session_id="s", content="x", content_type="invalid")  # type: ignore[arg-type]


def test_turn_defaults():
    turn = Turn(session_id="s", user_message="hi", assistant_message="hello")
    assert turn.id
    assert turn.tool_results == []
    assert turn.token_count == 0


def test_retrieved_context_budget_utilisation():
    ctx = RetrievedContext(
        recent_turns=[],
        episodic_memories=[],
        total_tokens_used=512,
        token_budget=4096,
        retrieval_duration_ms=10.0,
    )
    assert abs(ctx.budget_utilisation - 0.125) < 1e-6


def test_retrieved_context_zero_budget():
    ctx = RetrievedContext(
        recent_turns=[],
        episodic_memories=[],
        total_tokens_used=0,
        token_budget=0,
        retrieval_duration_ms=0.0,
    )
    assert ctx.budget_utilisation == 0.0
