from __future__ import annotations

from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import ConfigurationError


def test_config_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTRONE_REDIS_URL", "redis://myhost:6379")
    monkeypatch.setenv("ACTRONE_OPENAI_API_KEY", "sk-abc")
    cfg = MemoryConfig()
    assert cfg.redis_url == "redis://myhost:6379"
    assert cfg.openai_api_key is not None


def test_validate_runtime_raises_without_openai_key() -> None:
    cfg = MemoryConfig(embedding_provider="openai")
    with pytest.raises(ConfigurationError, match="ACTRONE_OPENAI_API_KEY"):
        cfg.validate_runtime()


def test_validate_runtime_passes_for_local_embedder() -> None:
    cfg = MemoryConfig(embedding_provider="local")
    cfg.validate_runtime()  # must not raise


def test_validate_runtime_raises_on_bad_budget() -> None:
    cfg = MemoryConfig(
        embedding_provider="local",
        budget_fraction_system=0.5,
        budget_fraction_episodic=0.5,
        budget_fraction_session=0.5,
        budget_fraction_current_turn=0.1,
    )
    with pytest.raises(ConfigurationError, match="sum to 1.0"):
        cfg.validate_runtime()


def test_validate_runtime_raises_on_bad_threshold() -> None:
    cfg = MemoryConfig(embedding_provider="local", relevance_threshold=1.5)
    with pytest.raises(ConfigurationError, match="relevance_threshold"):
        cfg.validate_runtime()


def test_validate_runtime_raises_on_zero_pool() -> None:
    cfg = MemoryConfig(embedding_provider="local", redis_max_connections=0)
    with pytest.raises(ConfigurationError, match="redis_max_connections"):
        cfg.validate_runtime()


def test_config_error_carries_details() -> None:
    cfg = MemoryConfig(embedding_provider="openai")
    with pytest.raises(ConfigurationError) as exc_info:
        cfg.validate_runtime()
    assert "embedding_provider" in exc_info.value.details


def test_validate_runtime_raises_on_unbalanced_rank_weights() -> None:
    """The documented blend is relevance × cosine + recency × recency, so the
    weights must sum to 1.0 rather than silently rescaling every score."""
    cfg = MemoryConfig(embedding_provider="local", relevance_weight=0.9, recency_weight=0.9)
    with pytest.raises(ConfigurationError, match="sum to 1.0"):
        cfg.validate_runtime()


def test_validate_runtime_accepts_balanced_rank_weights() -> None:
    cfg = MemoryConfig(embedding_provider="local", relevance_weight=0.6, recency_weight=0.4)
    cfg.validate_runtime()


@pytest.mark.parametrize("field", ["max_session_turns", "max_episodic_memories", "rerank_top_k"])
def test_validate_runtime_rejects_non_positive_limits(field: str) -> None:
    limits: dict[str, Any] = {field: 0}
    cfg = MemoryConfig(embedding_provider="local", **limits)
    with pytest.raises(ConfigurationError, match=field):
        cfg.validate_runtime()
