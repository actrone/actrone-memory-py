from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

"""Competitor / baseline adapters for the comparative benchmark.

The comparison harness (``compare.py``) evaluates any object satisfying the tiny
:class:`MemorySystem` surface — ``inject_memory`` + ``search_memories`` — so a
competitor (Mem0 / Zep / Letta / Cognee) can be dropped in with a thin adapter
that maps those two calls onto its API. Those adapters are **not bundled**: they
pull heavy deps and API keys that would break the offline CI gate. This module
ships a dependency-free **baseline** instead, so the comparative harness is real,
reproducible, and CI-runnable out of the box (Actrone vs a naive baseline).
"""


@dataclass
class _Stored:
    """A stored item exposing ``.id`` (the shape the harness reads from results)."""

    id: str
    content: str


@dataclass
class RecencyBaseline:
    """A naive baseline: returns the most-recently-injected memories, ignoring the
    query. Represents "no semantic retrieval" — the floor any real memory system
    should beat. Dependency-free and deterministic, so it anchors the comparison.
    """

    _by_agent: dict[str, list[_Stored]] = field(default_factory=dict)

    async def inject_memory(self, agent_id: str, content: str) -> str:
        item = _Stored(id=str(uuid4()), content=content)
        self._by_agent.setdefault(agent_id, []).append(item)
        return item.id

    async def search_memories(self, agent_id: str, query: str, limit: int) -> list[_Stored]:
        items = self._by_agent.get(agent_id, [])
        # Most-recent-first, query ignored — the naive floor.
        return list(reversed(items))[:limit]
