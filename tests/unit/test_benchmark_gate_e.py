"""Gate E: competitor-adapter mapping (E1) + external-dataset loader (E2).

The competitor SDKs aren't installed in CI, so the adapters are tested by injecting a fake client —
verifying the two-call MemorySystem surface maps correctly onto each vendor's API shape.
"""

from __future__ import annotations

import json

import pytest

from actrone_memory.benchmark.competitors import (
    LettaAdapter,
    Mem0Adapter,
    ZepAdapter,
)
from actrone_memory.benchmark.dataset import load_cases


class _FakeMem0:
    def add(self, messages: list[dict], user_id: str) -> list[dict]:  # noqa: ARG002
        return [{"id": "m0-1"}]

    def search(self, query: str, user_id: str, limit: int) -> dict:  # noqa: ARG002
        return {"results": [{"id": "m0-1", "memory": "the capital is Paris"}]}


@pytest.mark.asyncio
async def test_mem0_adapter_maps_add_and_search() -> None:
    a = Mem0Adapter(client=_FakeMem0())
    assert await a.inject_memory("agent-1", "Paris is the capital") == "m0-1"
    hits = await a.search_memories("agent-1", "capital", limit=5)
    assert [h.id for h in hits] == ["m0-1"]
    assert "Paris" in hits[0].content


class _FakeZepGraph:
    def add(self, user_id: str, type: str, data: str) -> dict:  # noqa: A002, ARG002
        return {"uuid": "z-1"}

    def search(self, user_id: str, query: str, limit: int) -> list[dict]:  # noqa: ARG002
        return [{"uuid": "z-1", "fact": "the capital is Paris"}]


class _FakeZep:
    graph = _FakeZepGraph()


@pytest.mark.asyncio
async def test_zep_adapter_maps_graph_add_and_search() -> None:
    a = ZepAdapter(client=_FakeZep())
    assert await a.inject_memory("agent-1", "Paris is the capital") == "z-1"
    hits = await a.search_memories("agent-1", "capital", limit=5)
    assert [h.id for h in hits] == ["z-1"]
    assert "Paris" in hits[0].content


class _FakePassages:
    def create(self, agent_id: str, text: str) -> dict:  # noqa: ARG002
        return {"id": "p-1"}

    def search(self, agent_id: str, query: str) -> list[dict]:  # noqa: ARG002
        return [{"id": "p-1", "text": "the capital is Paris"}]


class _FakeLetta:
    class agents:  # noqa: N801
        passages = _FakePassages()


@pytest.mark.asyncio
async def test_letta_adapter_maps_passages() -> None:
    a = LettaAdapter(client=_FakeLetta())
    assert await a.inject_memory("agent-1", "Paris is the capital") == "p-1"
    hits = await a.search_memories("agent-1", "capital", limit=5)
    assert [h.id for h in hits] == ["p-1"]


def test_load_cases_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = [
        {
            "name": "smoke",
            "agent_id": "a1",
            "memories": [{"id": "m1", "content": "Paris is the capital of France"}],
            "queries": [{"text": "capital of France", "relevant_ids": ["m1"]}],
        }
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    cases = load_cases(str(path))
    assert len(cases) == 1
    assert cases[0].agent_id == "a1"
    assert cases[0].memories[0].id == "m1"
    assert cases[0].queries[0].relevant_ids == frozenset({"m1"})


def test_load_cases_rejects_malformed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"name": "no-agent"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="agent_id"):
        load_cases(str(path))
