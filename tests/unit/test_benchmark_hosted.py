"""The hosted adapter: the same harness, measured on Actrone's hosted memory API."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest
from tenacity import wait_none

from actrone_memory.benchmark.compare import MemorySystem, run_comparison
from actrone_memory.benchmark.dataset import DEFAULT_DATASET
from actrone_memory.benchmark.hosted import ActroneHostedMemory, HostedMemoryError


class FakeHosted:
    """An in-memory stand-in for the hosted memory API, recording each request."""

    def __init__(self, fail_first: int = 0, status: int = 503) -> None:
        self.calls: list[tuple[str, str]] = []
        self.memories: list[dict[str, str]] = []
        self.fail_first = fail_first
        self.status = status

    def __call__(self, req: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
        method, url = req.get_method(), req.full_url
        self.calls.append((method, url))
        assert req.get_header("Authorization") == "Bearer key"
        assert req.get_header("X-request-id")
        if self.fail_first > 0:
            self.fail_first -= 1
            return self.status, b""
        if method == "DELETE":
            self.memories.clear()
            return 200, b"{}"
        if method == "POST":
            body: dict[str, Any] = json.loads(req.data or b"{}")  # type: ignore[arg-type]  # urllib types data as Any
            mid = f"m{len(self.memories) + 1}"
            self.memories.append({"memory_id": mid, "content": body["content"]})
            return 201, json.dumps({"memory_id": mid}).encode()
        query = url.split("query=")[1].split("&")[0].replace("+", " ").lower()
        words = [w for w in query.split() if len(w) > 3]
        hits = [m for m in self.memories if any(w in m["content"].lower() for w in words)]
        return 200, json.dumps({"entries": hits}).encode()


def adapter(fake: FakeHosted) -> ActroneHostedMemory:
    return ActroneHostedMemory(
        base_url="https://api.example.com/", api_key="key", agent_id="bench", transport=fake
    )


async def test_cases_never_see_each_others_memories() -> None:
    fake = FakeHosted()
    hosted = adapter(fake)
    assert isinstance(hosted, MemorySystem)
    await hosted.inject_memory("case-1", "The user prefers aisle seats.")
    await hosted.inject_memory("case-1", "The user is vegetarian.")
    hits = await hosted.search_memories("case-1", "which seats", 5)
    assert [h.content for h in hits] == ["The user prefers aisle seats."]
    await hosted.inject_memory("case-2", "The user lives in Leeds.")
    assert [c for c in fake.calls if c[0] == "DELETE"] == [
        ("DELETE", "https://api.example.com/v1/agents/bench/memory"),
        ("DELETE", "https://api.example.com/v1/agents/bench/memory"),
    ], "each new case starts from an erased memory"
    assert await hosted.search_memories("case-2", "vegetarian diet", 5) == []


async def test_runs_the_bundled_harness() -> None:
    reports = await run_comparison({"actrone-hosted": adapter(FakeHosted())}, DEFAULT_DATASET)
    assert reports["actrone-hosted"].n_queries > 0


async def test_retries_a_throttle_and_refuses_a_bad_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # No real waiting in tests: the backoff is tenacity's, so the test takes it to zero.
    monkeypatch.setattr(ActroneHostedMemory._call.retry, "wait", wait_none())  # type: ignore[attr-defined]  # tenacity attaches .retry
    fake = FakeHosted(fail_first=2, status=429)
    hosted = adapter(fake)
    assert await hosted.inject_memory("c", "x") == "m1"
    refused = FakeHosted(fail_first=5, status=401)
    with pytest.raises(HostedMemoryError):
        await adapter(refused).inject_memory("c", "x")
    assert len(refused.calls) == 1, "a refusal is not retried"


def test_needs_https_a_key_and_an_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        ActroneHostedMemory(base_url="http://api.example.com", api_key="k", agent_id="a")
    with pytest.raises(ValueError):
        ActroneHostedMemory(base_url="https://api.example.com", api_key="", agent_id="a")
    ActroneHostedMemory(base_url="http://localhost:8080", api_key="k", agent_id="a")
    monkeypatch.delenv("ACTRONE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ACTRONE_API_KEY"):
        ActroneHostedMemory.from_env()
