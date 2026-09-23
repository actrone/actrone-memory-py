"""Calibrated admission thresholds, the offline token counter and quiet-by-default logging.

These pin three launch-review fixes. One global threshold (0.72) recalled 4% of relevant memories
with the lexical embedder and 65% with bge-small; the default token counter downloaded tiktoken's
encoding on first use, contradicting "no egress"; and the library printed its debug events to stdout
in any program that imported it.
"""

from __future__ import annotations

import builtins
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from actrone_memory import MemoryManager
from actrone_memory.config import (
    DEFAULT_RELEVANCE_THRESHOLD,
    MemoryConfig,
    resolve_relevance_threshold,
)
from actrone_memory.exceptions import ConfigurationError
from actrone_memory.l2.embedder import (
    BGE_SMALL_RELEVANCE_THRESHOLD,
    LEXICAL_RELEVANCE_THRESHOLD,
    MINILM_RELEVANCE_THRESHOLD,
    CachedEmbedder,
    Embedder,
    FastEmbedEmbedder,
    HashingEmbedder,
    LocalEmbedder,
)
from actrone_memory.logging import bind_logger, configure_json_logging
from actrone_memory.tokens import build_token_counter, heuristic_token_counter


class _Declared(Embedder):
    """A custom embedder over the lexical vectors, optionally declaring a threshold."""

    def __init__(self, threshold: float | None) -> None:
        self._inner = HashingEmbedder()
        self._threshold = threshold

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def relevance_threshold(self) -> float | None:
        return self._threshold

    async def embed(self, text: str) -> list[float]:
        return await self._inner.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed_batch(texts)


def _hashing_config(**overrides: Any) -> MemoryConfig:  # noqa: ANN401
    return MemoryConfig(embedding_provider="hashing", **overrides)  # type: ignore[call-arg]


# ── Threshold precedence ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("configured", "declared", "expected"),
    [
        (0.5, 0.63, 0.5),  # an explicit config value wins
        (None, 0.63, 0.63),  # else the embedder's calibration
        (None, None, DEFAULT_RELEVANCE_THRESHOLD),  # else the library default
    ],
)
def test_resolve_relevance_threshold(
    configured: float | None, declared: float | None, expected: float
) -> None:
    assert resolve_relevance_threshold(configured, declared) == expected


def test_relevance_threshold_is_unset_by_default_and_validated_when_set() -> None:
    cfg = _hashing_config()
    assert cfg.relevance_threshold is None
    cfg.validate_runtime()  # unset is valid
    with pytest.raises(ConfigurationError, match="relevance_threshold"):
        _hashing_config(relevance_threshold=1.5).validate_runtime()


def test_built_in_embedders_declare_their_calibration() -> None:
    assert HashingEmbedder().relevance_threshold == LEXICAL_RELEVANCE_THRESHOLD
    # Constructing these would load a model; the declared value depends only on the model name.
    bge = object.__new__(FastEmbedEmbedder)
    bge._model_name = "BAAI/bge-small-en-v1.5"
    assert bge.relevance_threshold == BGE_SMALL_RELEVANCE_THRESHOLD
    other = object.__new__(FastEmbedEmbedder)
    other._model_name = "BAAI/bge-base-en-v1.5"
    assert other.relevance_threshold is None  # unmeasured model: the library default applies
    assert object.__new__(LocalEmbedder).relevance_threshold == MINILM_RELEVANCE_THRESHOLD


def test_cached_embedder_keeps_the_inner_calibration() -> None:
    cached = CachedEmbedder(HashingEmbedder(), cache=None)  # type: ignore[arg-type]
    assert cached.relevance_threshold == LEXICAL_RELEVANCE_THRESHOLD


async def test_manager_applies_the_resolved_threshold() -> None:
    lexical = await MemoryManager.create(_hashing_config())
    assert lexical.relevance_threshold == LEXICAL_RELEVANCE_THRESHOLD
    explicit = await MemoryManager.create(_hashing_config(relevance_threshold=0.55))
    assert explicit.relevance_threshold == 0.55
    declared = await MemoryManager.create(_hashing_config(), embedder=_Declared(0.42))
    assert declared.relevance_threshold == 0.42
    undeclared = await MemoryManager.create(_hashing_config(), embedder=_Declared(None))
    assert undeclared.relevance_threshold == DEFAULT_RELEVANCE_THRESHOLD
    for mm in (lexical, explicit, declared, undeclared):
        await mm.close()


async def test_lexical_default_recalls_a_matching_fact_and_not_an_unrelated_one() -> None:
    # The same case as the TypeScript README quickstart, so both libraries behave alike.
    mm = await MemoryManager.create(_hashing_config())
    try:
        await mm.inject_memory(
            "support-bot", "The customer is on the Enterprise plan.", importance=0.9
        )
        hit = await mm.retrieve_context(
            "support-bot", "sess-1", "Which plan is the customer on?", token_budget=4096
        )
        assert [m.content for m in hit.episodic_memories] == [
            "The customer is on the Enterprise plan."
        ]
        miss = await mm.retrieve_context(
            "support-bot", "sess-1", "How long is shipping to Kenya?", token_budget=4096
        )
        assert miss.episodic_memories == []
    finally:
        await mm.close()


# ── Token counting ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"), [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2), ("x" * 400, 100)]
)
def test_heuristic_counter_matches_the_typescript_library(text: str, expected: int) -> None:
    # TypeScript: text.length === 0 ? 0 : Math.max(1, Math.ceil(text.length / 4))
    assert heuristic_token_counter(text) == expected


@pytest.fixture
def no_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import tiktoken` fail, as on a bare `pip install actrone-memory`."""
    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if name == "tiktoken" or name.startswith("tiktoken."):
            raise ImportError("No module named 'tiktoken'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.usefixtures("no_tiktoken")
async def test_default_path_never_needs_tiktoken() -> None:
    mm = await MemoryManager.create(_hashing_config())
    try:
        await mm.store_turn("a", "s", "hello there", "general kenobi")
        ctx = await mm.retrieve_context("a", "s", "hello", token_budget=4096)
        assert ctx.recent_turns[0].token_count == heuristic_token_counter(
            "hello there\ngeneral kenobi"
        )
    finally:
        await mm.close()


@pytest.mark.usefixtures("no_tiktoken")
def test_tiktoken_counter_without_the_extra_says_how_to_install_it() -> None:
    with pytest.raises(ConfigurationError, match=r"actrone-memory\[tiktoken\]"):
        build_token_counter("tiktoken")


# ── Logging ──────────────────────────────────────────────────────────────────


@pytest.fixture
def pristine_logging() -> Iterator[None]:
    """Start from an application that has configured neither structlog nor logging."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    structlog.reset_defaults()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    yield
    structlog.reset_defaults()
    root.handlers[:], root.level = saved[0], saved[1]


@pytest.mark.usefixtures("pristine_logging")
def test_unconfigured_app_sees_no_debug_or_info_output(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    log = bind_logger("actrone_memory.test")
    log.debug("memory.test.debug", detail="noise")
    log.info("memory.test.info", detail="noise")
    with caplog.at_level(logging.WARNING, logger="actrone_memory.test"):
        log.warning("memory.test.warning", detail="worth seeing")
    out = capsys.readouterr().out
    assert out == "", "nothing may be printed to stdout by default"
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "memory.test.warning" in caplog.records[0].getMessage()


@pytest.mark.usefixtures("pristine_logging")
def test_a_configured_app_gets_the_library_events_in_its_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_json_logging(logging.INFO)
    bind_logger("actrone_memory.test").info("memory.test.info", detail="visible")
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    event = json.loads(lines[-1])
    assert event["event"] == "memory.test.info"
    assert event["service"] == "actrone-memory"
    assert event["detail"] == "visible"
