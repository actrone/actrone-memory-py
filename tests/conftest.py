from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.l2.embedder import Embedder
from actrone_memory.models import MemoryEntry, Turn

# NOTE: redis/qdrant are OPTIONAL extras, so do NOT import RedisStore/QdrantStore at collection
# time here, or every test run without the durable-backend extras (e.g. the local-first path and the
# compat-matrix jobs) fails to collect. Integration tests that need them import them lazily.

# In a compat-matrix job the framework under test IS installed, so a "<framework> not installed"
# skip there is a false pass: it is how an installed-but-unimportable framework hides (google-adk
# 1.0.0 imports `deprecated` without declaring it, and its native test was silently skipped). run.sh
# sets this variable; everywhere else these skips stay skips, so the framework-free base run is
# unaffected.
_REQUIRE_FRAMEWORK = os.environ.get("COMPAT_MATRIX_REQUIRE_FRAMEWORK") == "1"


def _skip_reason(report: pytest.CollectReport | pytest.TestReport) -> str:
    longrepr = report.longrepr
    return str(longrepr[2]) if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)


def _fail_missing_framework(report: pytest.CollectReport | pytest.TestReport) -> None:
    if _REQUIRE_FRAMEWORK and report.skipped and "not installed" in _skip_reason(report):
        report.outcome = "failed"
        report.longrepr = (
            f"compat-matrix: {_skip_reason(report)}. The framework under test must import in this "
            "job; an import error inside it (a missing or broken dependency) surfaces as this skip."
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, Any, None]:
    outcome = yield
    _fail_missing_framework(outcome.get_result())


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Generator[None, Any, None]:
    # A module-level `pytest.importorskip` skips at collection, before any test report exists.
    outcome = yield
    _fail_missing_framework(outcome.get_result())


@pytest.fixture
def fake_config() -> MemoryConfig:
    return MemoryConfig(
        redis_url="redis://localhost:6379",
        qdrant_url="http://localhost:6333",
        embedding_provider="openai",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        session_ttl_hours=1,
        max_session_turns=10,
        relevance_threshold=0.72,
        auto_summarise=False,
    )


class ConstantEmbedder(Embedder):
    """Returns a fixed embedding vector, deterministic for tests."""

    @property
    def dimensions(self) -> int:
        return 4

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture
def embedder() -> ConstantEmbedder:
    return ConstantEmbedder()


@pytest.fixture
def sample_turn() -> Turn:
    return Turn(
        session_id="sess-1",
        user_message="What is the capital of France?",
        assistant_message="The capital of France is Paris.",
        token_count=20,
    )


@pytest.fixture
def sample_memory_entry() -> MemoryEntry:
    return MemoryEntry(
        agent_id="agent-1",
        session_id="sess-1",
        content="Paris is the capital of France.",
        content_type="summary",
        embedding=[0.1, 0.2, 0.3, 0.4],
        importance_score=0.8,
        token_count=10,
    )
