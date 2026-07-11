from __future__ import annotations

import time
from dataclasses import dataclass

from actrone_memory.benchmark.dataset import DEFAULT_DATASET, EvalCase
from actrone_memory.config import MemoryConfig
from actrone_memory.manager import MemoryManager

"""Reproducible memory-quality eval harness.

Seeds each case's memories into a fresh (by default local, zero-service) manager,
runs the queries, and reports recall@k / precision@k / MRR plus retrieval latency.
Runs offline in CI so memory quality is a regression-gated property.

Honest by construction: the default backend uses the dependency-free hashing
embedder (keyword-overlap recall), so scores reflect *that* embedder — swap in a
real embedder (OpenAI / sentence-transformers) to measure semantic recall. The
harness is embedder- and backend-agnostic; ``run_eval`` accepts any manager.
"""


@dataclass(frozen=True)
class QueryOutcome:
    query: str
    retrieved_ids: list[str]
    relevant_ids: frozenset[str]
    latency_ms: float

    @property
    def hit_rank(self) -> int | None:
        """1-indexed rank of the first relevant result, or None if not retrieved."""
        for i, mid in enumerate(self.retrieved_ids, start=1):
            if mid in self.relevant_ids:
                return i
        return None


@dataclass(frozen=True)
class EvalReport:
    k: int
    n_queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    outcomes: list[QueryOutcome]

    def format_table(self) -> str:
        """Render an honest, copy-pasteable Markdown summary."""
        return (
            f"| metric | value |\n"
            f"| --- | --- |\n"
            f"| queries | {self.n_queries} |\n"
            f"| recall@{self.k} | {self.recall_at_k:.3f} |\n"
            f"| precision@{self.k} | {self.precision_at_k:.3f} |\n"
            f"| MRR | {self.mrr:.3f} |\n"
            f"| latency p50 | {self.latency_p50_ms:.2f} ms |\n"
            f"| latency p95 | {self.latency_p95_ms:.2f} ms |"
        )


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (0.0 for an empty input). ``pct`` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank: rank = ceil(pct/100 * n), 1-indexed, clamped to [1, n].
    rank = max(1, min(len(ordered), int(-(-len(ordered) * pct // 100))))
    return ordered[rank - 1]


async def _seed_case(mm: MemoryManager, case: EvalCase) -> dict[str, str]:
    """Inject a case's memories (+ distractors) and return dataset-id → stored-id."""
    id_map: dict[str, str] = {}
    for item in [*case.memories, *case.distractors]:
        stored = await mm.inject_memory(
            case.agent_id,
            item.content,
            importance=0.8,
            source=item.source,
            sensitivity=item.sensitivity,
        )
        id_map[item.id] = stored
    return id_map


async def run_eval(
    cases: list[EvalCase] | None = None,
    *,
    manager: MemoryManager | None = None,
    config: MemoryConfig | None = None,
    k: int = 5,
) -> EvalReport:
    """Run the eval over ``cases`` and return aggregated metrics.

    Args:
        cases: Eval cases (defaults to the bundled ``DEFAULT_DATASET``).
        manager: A ready manager to evaluate. When None, a fresh local-first
            manager is created (and closed) per run. A low relevance threshold is
            used by default so the metric reflects *ranking* quality rather than a
            provider-specific similarity cutoff.
        config: Override config for the auto-created manager.
        k: Cut-off for recall@k / precision@k and the search depth.

    Returns:
        An :class:`EvalReport` with recall@k, precision@k, MRR, and latencies.
    """
    cases = cases if cases is not None else DEFAULT_DATASET
    owns_manager = manager is None
    if manager is None:
        cfg = config or MemoryConfig(relevance_threshold=0.05, max_episodic_memories=k)
        manager = await MemoryManager.create(cfg)

    try:
        outcomes: list[QueryOutcome] = []
        for case in cases:
            id_map = await _seed_case(manager, case)
            for query in case.queries:
                relevant = frozenset(id_map[d] for d in query.relevant_ids if d in id_map)
                start = time.perf_counter()
                hits = await manager.search_memories(case.agent_id, query.text, limit=k)
                latency_ms = (time.perf_counter() - start) * 1000
                outcomes.append(
                    QueryOutcome(
                        query=query.text,
                        retrieved_ids=[h.id for h in hits],
                        relevant_ids=relevant,
                        latency_ms=latency_ms,
                    )
                )
    finally:
        if owns_manager:
            await manager.close()

    return _aggregate(outcomes, k)


def _aggregate(outcomes: list[QueryOutcome], k: int) -> EvalReport:
    n = len(outcomes)
    recall_sum = 0.0
    precision_sum = 0.0
    rr_sum = 0.0
    for o in outcomes:
        top_k = o.retrieved_ids[:k]
        hits = sum(1 for mid in top_k if mid in o.relevant_ids)
        if o.relevant_ids:
            recall_sum += hits / len(o.relevant_ids)
        precision_sum += hits / k if k else 0.0
        rank = o.hit_rank
        rr_sum += (1.0 / rank) if rank is not None and rank <= k else 0.0

    latencies = [o.latency_ms for o in outcomes]
    return EvalReport(
        k=k,
        n_queries=n,
        recall_at_k=recall_sum / n if n else 0.0,
        precision_at_k=precision_sum / n if n else 0.0,
        mrr=rr_sum / n if n else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        outcomes=outcomes,
    )
