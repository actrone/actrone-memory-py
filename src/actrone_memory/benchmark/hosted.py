"""Actrone's hosted memory, measured with this harness.

The same recall@k / precision@k / MRR harness runs against Actrone's hosted memory through its
memory API, so hosted and local memory are measured the same way on the same cases (LOCOMO and
LongMemEval converted with :func:`actrone_memory.benchmark.dataset.load_cases`).

The adapter writes to one agent kept for benchmarks and erases that agent's memory whenever a new
case starts, so cases never see each other's memories. It needs an API key that may manage that
agent's memory. Standard library HTTP only: no new dependency.

Run::

    ACTRONE_BASE_URL=https://api.actrone.com ACTRONE_API_KEY=... ACTRONE_BENCH_AGENT_ID=... \\
        python -m actrone_memory.benchmark.hosted --dataset locomo.json --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from actrone_memory.benchmark.compare import format_comparison, run_comparison
from actrone_memory.benchmark.dataset import DEFAULT_DATASET, load_cases

#: A transport sends one request and returns its status and body. The default is urllib.
Transport = Callable[[urllib.request.Request, float], tuple[int, bytes]]


class HostedMemoryError(RuntimeError):
    """The hosted memory API refused a request (not retried)."""


class _RetryableError(RuntimeError):
    """A response worth retrying: throttled or a server error."""


def _urllib_transport(req: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https base URL, checked below
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@dataclass(frozen=True)
class _Stored:
    id: str
    content: str


@dataclass
class ActroneHostedMemory:
    """Actrone's hosted memory as a :class:`~actrone_memory.benchmark.compare.MemorySystem`."""

    base_url: str
    api_key: str
    agent_id: str
    timeout: float = 30.0
    transport: Transport = _urllib_transport
    _current_case: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.base_url)
        local = parsed.hostname in {"localhost", "127.0.0.1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("base_url must be https (http only for localhost)")
        if not self.api_key or not self.agent_id:
            raise ValueError("an API key and the benchmark agent's id are required")
        self.base_url = self.base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> ActroneHostedMemory:
        """Build from ACTRONE_BASE_URL, ACTRONE_API_KEY and ACTRONE_BENCH_AGENT_ID."""
        missing = [
            n
            for n in ("ACTRONE_BASE_URL", "ACTRONE_API_KEY", "ACTRONE_BENCH_AGENT_ID")
            if not os.environ.get(n)
        ]
        if missing:
            raise ValueError("set " + ", ".join(missing))
        return cls(
            base_url=os.environ["ACTRONE_BASE_URL"],
            api_key=os.environ["ACTRONE_API_KEY"],
            agent_id=os.environ["ACTRONE_BENCH_AGENT_ID"],
        )

    @retry(
        retry=retry_if_exception_type((_RetryableError, urllib.error.URLError, TimeoutError)),
        wait=wait_random_exponential(multiplier=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)  # noqa: S310 - the scheme is checked in __post_init__ (https, or http on localhost)
        req.add_header("Authorization", "Bearer " + self.api_key)
        req.add_header("X-Request-Id", str(uuid.uuid4()))
        if data is not None:
            req.add_header("Content-Type", "application/json")
        status, raw = self.transport(req, self.timeout)
        if status == 429 or status >= 500:
            raise _RetryableError(f"{method} {path}: {status}")
        if status >= 400:
            raise HostedMemoryError(f"{method} {path}: {status}: {raw[:300]!r}")
        return json.loads(raw) if raw else None

    def _memory_path(self) -> str:
        return "/v1/agents/" + urllib.parse.quote(self.agent_id, safe="") + "/memory"

    async def _start_case(self, case_id: str) -> None:
        if case_id == self._current_case:
            return
        await asyncio.to_thread(self._call, "DELETE", self._memory_path())
        self._current_case = case_id

    async def inject_memory(self, agent_id: str, content: str) -> str:
        """Store one memory for a case; a new case first erases the benchmark agent's memory."""
        await self._start_case(agent_id)
        res = await asyncio.to_thread(
            self._call, "POST", self._memory_path(), {"content": content, "importance": 0.8}
        )
        return str((res or {}).get("memory_id") or uuid.uuid4())

    async def search_memories(self, agent_id: str, query: str, limit: int) -> list[_Stored]:
        """Search the case's memories."""
        await self._start_case(agent_id)
        qs = urllib.parse.urlencode({"query": query, "limit": limit})
        res = await asyncio.to_thread(self._call, "GET", self._memory_path() + "?" + qs)
        entries = (res or {}).get("entries") or []
        return [
            _Stored(id=str(e.get("memory_id", "")), content=str(e.get("content", "")))
            for e in entries
        ]


async def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure Actrone's hosted memory with this harness."
    )
    parser.add_argument(
        "--dataset", help="cases as JSON (load_cases); the bundled set when omitted"
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    cases = load_cases(args.dataset) if args.dataset else DEFAULT_DATASET
    hosted = ActroneHostedMemory.from_env()
    reports = await run_comparison({"actrone-hosted": hosted}, cases, k=args.k)
    print(format_comparison(reports))


if __name__ == "__main__":
    asyncio.run(_main())
