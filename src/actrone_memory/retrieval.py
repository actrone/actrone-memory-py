"""Hybrid retrieval — dense + lexical + recency fused with Reciprocal Rank Fusion (Axis A3).

The OSS default recall used a single ranking channel (embedding cosine, blended with recency). With
a weak/lexical default embedder that under-recalls; with a strong dense embedder it still misses
exact-keyword matches. **Reciprocal Rank Fusion (RRF)** combines several independent rankings into
one without needing their scores to be on the same scale — the standard, parameter-light way to do
hybrid (dense + lexical) retrieval.

This module is pure and dependency-free (BM25 is implemented here, no external index), so it works
identically in the in-process store and behind Qdrant. Fusion re-ranks **within the dense-threshold
admitted set**, so the "only sufficiently-relevant memories are returned" contract is unchanged —
RRF only reorders which relevant memory surfaces first, adding an exact-keyword (lexical) signal the
pure embedding channel lacks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from actrone_memory.in_memory import cosine_similarity
from actrone_memory.models import MemoryEntry

_WORD_RE = re.compile(r"[a-z0-9]+")
_RECENCY_WINDOW_SECONDS = 30 * 86400
# RRF constant: the canonical default (Cormack et al., 2009). Larger k → flatter contribution decay.
DEFAULT_RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens — the shared tokenisation for lexical scoring."""
    return _WORD_RE.findall(text.lower())


def bm25_scores(
    query: str, documents: dict[str, str], *, k1: float = 1.5, b: float = 0.75
) -> dict[str, float]:
    """Okapi BM25 relevance of ``query`` against each document, keyed by document id.

    Dependency-free and computed over the (small, bounded) candidate set only — no persistent
    index. Returns a score per document; a document sharing no query term scores ``0.0``.

    Time complexity: O(Q · D) over Q query terms and D candidate documents — acceptable because the
    candidate set is bounded (``max_episodic_memories`` / the over-fetch limit).
    """
    q_terms = set(tokenize(query))
    if not q_terms or not documents:
        return dict.fromkeys(documents, 0.0)

    doc_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
    doc_len = {doc_id: len(toks) for doc_id, toks in doc_tokens.items()}
    n_docs = len(documents)
    avgdl = (sum(doc_len.values()) / n_docs) if n_docs else 0.0

    # Document frequency per query term.
    df: Counter[str] = Counter()
    for toks in doc_tokens.values():
        present = set(toks)
        for term in q_terms:
            if term in present:
                df[term] += 1

    scores: dict[str, float] = {}
    for doc_id, toks in doc_tokens.items():
        tf = Counter(toks)
        length = doc_len[doc_id]
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            n_qi = df[term]
            # BM25 idf with the +1 form (always non-negative).
            idf = math.log(1 + (n_docs - n_qi + 0.5) / (n_qi + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * (length / avgdl if avgdl else 0.0))
            score += idf * (freq * (k1 + 1)) / denom if denom else 0.0
        scores[doc_id] = score
    return scores


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """Fuse several ranked id-lists into one score map via weighted Reciprocal Rank Fusion.

    ``score(d) = Σ_channel weight_channel · 1 / (k + rank_channel(d))`` where ``rank`` is 1-based.
    An id absent from a channel contributes nothing for that channel. Weights default to 1.0 each.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return fused


def fuse_channels(
    *,
    ids: Sequence[str],
    dense_ranking: Sequence[str],
    documents: dict[str, str],
    recency: dict[str, float],
    query_text: str | None,
    relevance_weight: float,
    recency_weight: float,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[str] | None:
    """RRF-fuse a dense ranking with a lexical (BM25) and recency ranking over the same id set.

    Returns the fused id ordering, or ``None`` when there is no lexical signal (blank query or no
    shared terms) so the caller can fall back to its classic dense+recency blend. Shared by the
    in-process store (dense = recomputed cosine) and the Qdrant store (dense = server cosine score),
    so the two backends rank identically given the same candidates.
    """
    if not (query_text and query_text.strip()):
        return None
    bm = bm25_scores(query_text, documents)
    lexical_ranking = [
        doc_id
        for doc_id, score in sorted(bm.items(), key=lambda x: x[1], reverse=True)
        if score > 0
    ]
    if not lexical_ranking:
        return None
    recency_ranking = [
        doc_id for doc_id, _ in sorted(recency.items(), key=lambda x: x[1], reverse=True)
    ]
    fused = reciprocal_rank_fusion(
        [list(dense_ranking), lexical_ranking, recency_ranking],
        weights=[relevance_weight, relevance_weight, recency_weight],
        k=rrf_k,
    )
    return sorted(ids, key=lambda i: fused.get(i, 0.0), reverse=True)


def _classic_blend(
    scored: list[tuple[float, MemoryEntry]],
    recency: dict[str, float],
    *,
    relevance_weight: float,
    recency_weight: float,
    limit: int,
) -> list[MemoryEntry]:
    """The pre-hybrid weighted-sum ranking — used when there is no lexical signal to fuse."""
    ranked = sorted(
        scored,
        key=lambda pair: relevance_weight * pair[0] + recency_weight * recency[pair[1].id],
        reverse=True,
    )
    return [entry for _, entry in ranked[:limit]]


def hybrid_rank(
    entries: Sequence[MemoryEntry],
    query_embedding: list[float],
    query_text: str | None,
    *,
    threshold: float,
    relevance_weight: float,
    recency_weight: float,
    limit: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[MemoryEntry]:
    """Rank the dense-threshold-admitted memories by fusing dense + lexical + recency channels.

    Admission is unchanged: an entry is eligible only if its embedding cosine ≥ ``threshold``. Among
    the eligible set, three rankings are fused with RRF — embedding cosine (dense), BM25 over the
    content (lexical), and recency — so an exact-keyword match the embedder under-ranked can still
    surface first. When ``query_text`` yields no lexical signal (blank query, or no shared terms)
    the function degrades to the classic ``relevance·cosine + recency·recency`` blend, keeping
    behaviour identical to the pre-hybrid path.
    """
    now_ts = datetime.now(UTC).timestamp()
    eligible: list[tuple[float, MemoryEntry]] = []
    recency: dict[str, float] = {}
    for entry in entries:
        if entry.embedding is None:
            continue
        sim = cosine_similarity(query_embedding, entry.embedding)
        if sim < threshold:
            continue
        age_seconds = max(0.0, now_ts - entry.timestamp.timestamp())
        recency[entry.id] = max(0.0, 1.0 - age_seconds / _RECENCY_WINDOW_SECONDS)
        eligible.append((sim, entry))

    if not eligible:
        return []

    dense_ranking = [e.id for _, e in sorted(eligible, key=lambda p: p[0], reverse=True)]
    by_id = {e.id: e for _, e in eligible}
    fused_order = fuse_channels(
        ids=list(by_id.keys()),
        dense_ranking=dense_ranking,
        documents={e.id: e.content for _, e in eligible},
        recency=recency,
        query_text=query_text,
        relevance_weight=relevance_weight,
        recency_weight=recency_weight,
        rrf_k=rrf_k,
    )
    if fused_order is None:
        # No lexical signal to fuse — identical to the pre-hybrid ranking.
        return _classic_blend(
            eligible,
            recency,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            limit=limit,
        )
    return [by_id[i] for i in fused_order[:limit]]
