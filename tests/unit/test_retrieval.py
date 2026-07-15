"""Unit tests for hybrid retrieval (Axis A3): BM25, Reciprocal Rank Fusion, and hybrid_rank."""

from __future__ import annotations

from datetime import UTC, datetime

from actrone_memory.models import MemoryEntry
from actrone_memory.retrieval import (
    bm25_scores,
    hybrid_rank,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_lowercases_alphanumeric() -> None:
    assert tokenize("The Quick, brown FOX-42!") == ["the", "quick", "brown", "fox", "42"]


def test_bm25_ranks_term_match_above_non_match() -> None:
    docs = {
        "a": "the capital of France is Paris",
        "b": "bananas are a yellow fruit",
        "c": "France shares a border with Spain",
    }
    scores = bm25_scores("France", docs)
    assert scores["a"] > 0
    assert scores["c"] > 0
    assert scores["b"] == 0  # no shared term


def test_bm25_empty_query_or_docs() -> None:
    assert bm25_scores("", {"a": "hello"}) == {"a": 0.0}
    assert bm25_scores("hello", {}) == {}


def test_reciprocal_rank_fusion_combines_channels() -> None:
    dense = ["x", "y", "z"]
    lexical = ["z", "x", "y"]
    fused = reciprocal_rank_fusion([dense, lexical])
    # z is 3rd in dense but 1st in lexical; x is 1st in dense, 2nd in lexical.
    # x: 1/61 + 1/62 ; z: 1/63 + 1/61 — x edges out z, both above y.
    ranked = sorted(fused, key=lambda i: fused[i], reverse=True)
    assert ranked[0] == "x"
    assert ranked[-1] == "y"


def test_reciprocal_rank_fusion_weights_validated() -> None:
    import pytest

    with pytest.raises(ValueError, match="weights length"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def _entry(mid: str, content: str, embedding: list[float]) -> MemoryEntry:
    return MemoryEntry(
        id=mid,
        agent_id="agent-1",
        session_id="s1",
        content=content,
        content_type="summary",
        importance_score=0.5,
        token_count=len(content.split()),
        embedding=embedding,
        timestamp=datetime.now(UTC),
    )


def test_hybrid_rank_lexical_rescues_keyword_match() -> None:
    # Dense favours A (higher cosine to the query vector); lexical favours B (exact keyword).
    query_embedding = [1.0, 0.0, 0.0]
    a = _entry("A", "bananas are a yellow fruit", [0.99, 0.14, 0.0])  # cosine ~0.99, no keyword
    b = _entry("B", "the capital of France is Paris", [0.72, 0.69, 0.0])  # cosine ~0.72, keyword

    ranked = hybrid_rank(
        [a, b],
        query_embedding,
        query_text="France",
        threshold=0.5,
        relevance_weight=0.7,
        recency_weight=0.3,
        limit=10,
    )
    assert {e.id for e in ranked} == {"A", "B"}
    # Fusion lifts B (it wins the lexical channel and is only just behind on dense).
    assert ranked[0].id == "B"


def test_hybrid_rank_respects_threshold_admission() -> None:
    query_embedding = [1.0, 0.0, 0.0]
    near = _entry("near", "France Paris", [0.99, 0.14, 0.0])  # cosine ~0.99 (admitted)
    far = _entry("far", "France Paris", [0.0, 1.0, 0.0])  # cosine 0.0 (below threshold)
    ranked = hybrid_rank(
        [near, far],
        query_embedding,
        query_text="France",
        threshold=0.5,
        relevance_weight=0.7,
        recency_weight=0.3,
        limit=10,
    )
    assert [e.id for e in ranked] == ["near"]  # far excluded despite the keyword match


def test_hybrid_rank_without_query_text_is_classic_blend() -> None:
    query_embedding = [1.0, 0.0, 0.0]
    a = _entry("A", "alpha", [0.99, 0.14, 0.0])  # higher cosine
    b = _entry("B", "beta", [0.72, 0.69, 0.0])  # lower cosine
    ranked = hybrid_rank(
        [a, b],
        query_embedding,
        query_text=None,  # no lexical channel → dense+recency blend
        threshold=0.5,
        relevance_weight=0.7,
        recency_weight=0.3,
        limit=10,
    )
    assert [e.id for e in ranked] == ["A", "B"]  # pure dense order preserved
