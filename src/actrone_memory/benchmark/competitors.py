from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
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


# ── E1: real competitor adapters (opt-in — heavy deps + API keys, NOT bundled) ──────────────────
#
# Each maps the two-call MemorySystem surface onto a competitor's 2026 client API. They are
# lazily-imported and key-gated so the offline CI gate is unaffected; pass a pre-built ``client`` to
# unit-test the mapping without the real SDK. Gate E of the Memory Depth plan runs these on LOCOMO /
# LongMemEval before any "beats X" claim.


def _as_list(res: object) -> list[dict[str, Any]]:
    """Normalise a competitor response to a list of dicts. Handles a list, a
    ``{"results": [...]}`` /
    ``{"memories": [...]}`` / ``{"edges": [...]}`` wrapper, and a single bare object (a create/add
    response), so id extraction works across the vendors' shapes."""
    if isinstance(res, dict):
        for key in ("results", "memories", "edges"):
            if key in res:
                res = res[key]
                break
        else:
            return [res]  # a single object (e.g. an add/create response)
    if not isinstance(res, list):
        return []
    return [r if isinstance(r, dict) else {"id": "", "content": str(r)} for r in res]


def _first_id(res: object) -> str:
    for r in _as_list(res):
        rid = r.get("id") or r.get("uuid") or r.get("memory_id")
        if rid:
            return str(rid)
    return ""


@dataclass
class Mem0Adapter:
    """Mem0 (``pip install mem0ai`` + ``MEM0_API_KEY``).

    inject→``add``, search→``search`` by user_id.
    """

    client: Any = None

    def _ensure(self) -> Any:
        if self.client is None:
            try:
                from mem0 import MemoryClient
            except ImportError as exc:  # pragma: no cover - exercised via import guard
                raise ImportError(
                    "Mem0 adapter needs: pip install mem0ai (+ MEM0_API_KEY)"
                ) from exc
            self.client = MemoryClient()
        return self.client

    async def inject_memory(self, agent_id: str, content: str) -> str:
        client = self._ensure()
        res = await asyncio.to_thread(
            client.add, [{"role": "user", "content": content}], user_id=agent_id
        )
        return _first_id(res) or str(uuid4())

    async def search_memories(self, agent_id: str, query: str, limit: int) -> list[_Stored]:
        client = self._ensure()
        res = await asyncio.to_thread(client.search, query, user_id=agent_id, limit=limit)
        return [
            _Stored(id=str(r.get("id", "")), content=str(r.get("memory") or r.get("content", "")))
            for r in _as_list(res)
        ]


@dataclass
class ZepAdapter:
    """Zep Cloud (``pip install zep-cloud`` + ``ZEP_API_KEY``). inject→``graph.add`` (text),
    search→``graph.search`` on the user graph."""

    client: Any = None

    def _ensure(self) -> Any:
        if self.client is None:
            try:
                from zep_cloud.client import Zep
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Zep adapter needs: pip install zep-cloud (+ ZEP_API_KEY)"
                ) from exc
            self.client = Zep()
        return self.client

    async def inject_memory(self, agent_id: str, content: str) -> str:
        client = self._ensure()
        res = await asyncio.to_thread(client.graph.add, user_id=agent_id, type="text", data=content)
        return _first_id(res) or str(uuid4())

    async def search_memories(self, agent_id: str, query: str, limit: int) -> list[_Stored]:
        client = self._ensure()
        res = await asyncio.to_thread(
            client.graph.search, user_id=agent_id, query=query, limit=limit
        )
        edges = res.edges if hasattr(res, "edges") else _as_list(res)
        out: list[_Stored] = []
        for e in edges:
            content = getattr(e, "fact", None) or (e.get("fact") if isinstance(e, dict) else "")
            rid = getattr(e, "uuid_", None) or (e.get("uuid") if isinstance(e, dict) else "")
            out.append(_Stored(id=str(rid or ""), content=str(content or "")))
        return out[:limit]


@dataclass
class LettaAdapter:
    """Letta (``pip install letta-client`` + a running server/key). inject→archival
    ``passages.create``, search→``passages.search``. ``agent_id`` must be a Letta agent id."""

    client: Any = None

    def _ensure(self) -> Any:
        if self.client is None:
            try:
                from letta_client import Letta
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Letta adapter needs: pip install letta-client") from exc
            self.client = Letta()
        return self.client

    async def inject_memory(self, agent_id: str, content: str) -> str:
        client = self._ensure()
        res = await asyncio.to_thread(
            client.agents.passages.create, agent_id=agent_id, text=content
        )
        return _first_id(res) or str(uuid4())

    async def search_memories(self, agent_id: str, query: str, limit: int) -> list[_Stored]:
        client = self._ensure()
        res = await asyncio.to_thread(client.agents.passages.search, agent_id=agent_id, query=query)
        out: list[_Stored] = []
        for p in _as_list(res):
            out.append(
                _Stored(id=str(p.get("id", "")), content=str(p.get("text") or p.get("content", "")))
            )
        return out[:limit]
