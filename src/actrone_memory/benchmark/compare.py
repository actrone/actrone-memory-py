from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from actrone_memory.benchmark.dataset import DEFAULT_DATASET, EvalCase
from actrone_memory.benchmark.harness import EvalReport, QueryOutcome, _aggregate

"""Comparative benchmark — run the SAME dataset against multiple memory systems.

Honest by design: any system satisfying :class:`MemorySystem` (Actrone, or a thin
competitor adapter for Mem0/Zep/Letta) is evaluated on the identical dataset + metrics,
so the comparison is apples-to-apples and reproducible. Bundled runs compare Actrone
against the dependency-free baseline; competitor numbers require their libs + keys
(off the CI gate) — add an adapter and pass it in.
"""


@runtime_checkable
class MemorySystem(Protocol):
    """The minimal surface the comparative harness needs. Actrone's ``MemoryManager``
    satisfies it directly; a competitor needs a thin adapter mapping these two calls."""

    async def inject_memory(self, agent_id: str, content: str) -> str: ...

    async def search_memories(self, agent_id: str, query: str, limit: int) -> list[Any]: ...


async def run_comparison(
    systems: dict[str, MemorySystem],
    cases: list[EvalCase] | None = None,
    *,
    k: int = 5,
) -> dict[str, EvalReport]:
    """Evaluate each named system on ``cases`` and return per-system reports.

    Each system is seeded and queried independently on the same ground-truth dataset,
    so recall@k / precision@k / MRR are directly comparable.
    """
    cases = cases if cases is not None else DEFAULT_DATASET
    reports: dict[str, EvalReport] = {}

    for name, system in systems.items():
        outcomes: list[QueryOutcome] = []
        for case in cases:
            id_map: dict[str, str] = {}
            for item in case.memories:
                stored = await system.inject_memory(case.agent_id, item.content)
                id_map[item.id] = stored
            for query in case.queries:
                relevant = frozenset(id_map[d] for d in query.relevant_ids if d in id_map)
                start = time.perf_counter()
                hits = await system.search_memories(case.agent_id, query.text, k)
                latency_ms = (time.perf_counter() - start) * 1000
                outcomes.append(
                    QueryOutcome(
                        query=query.text,
                        retrieved_ids=[str(h.id) for h in hits],
                        relevant_ids=relevant,
                        latency_ms=latency_ms,
                    )
                )
        reports[name] = _aggregate(outcomes, k)

    return reports


def format_comparison(reports: dict[str, EvalReport]) -> str:
    """Render a per-system comparison as an honest Markdown table (one row per system)."""
    if not reports:
        return "(no systems)"
    k = next(iter(reports.values())).k
    lines = [
        f"| system | recall@{k} | precision@{k} | MRR | p95 latency |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, r in reports.items():
        lines.append(
            f"| {name} | {r.recall_at_k:.3f} | {r.precision_at_k:.3f} | "
            f"{r.mrr:.3f} | {r.latency_p95_ms:.2f} ms |"
        )
    return "\n".join(lines)
