from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import ConfigurationError


def test_config_loads_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ACTRONE_REDIS_URL", "redis://myhost:6379")
    monkeypatch.setenv("ACTRONE_OPENAI_API_KEY", "sk-abc")
    cfg = MemoryConfig()  # type: ignore[call-arg]
    assert cfg.redis_url == "redis://myhost:6379"
    assert cfg.openai_api_key is not None


def test_validate_runtime_raises_without_openai_key():
    cfg = MemoryConfig(embedding_provider="openai")  # type: ignore[call-arg]
    with pytest.raises(ConfigurationError, match="ACTRONE_OPENAI_API_KEY"):
        cfg.validate_runtime()


def test_validate_runtime_passes_for_local_embedder():
    cfg = MemoryConfig(embedding_provider="local")  # type: ignore[call-arg]
    cfg.validate_runtime()  # must not raise


def test_validate_runtime_raises_on_bad_budget():
    cfg = MemoryConfig(
        embedding_provider="local",
        budget_fraction_system=0.5,
        budget_fraction_episodic=0.5,
        budget_fraction_session=0.5,
        budget_fraction_current_turn=0.1,
    )  # type: ignore[call-arg]
    with pytest.raises(ConfigurationError, match="sum to 1.0"):
        cfg.validate_runtime()


def test_validate_runtime_raises_on_bad_threshold():
    cfg = MemoryConfig(embedding_provider="local", relevance_threshold=1.5)  # type: ignore[call-arg]
    with pytest.raises(ConfigurationError, match="relevance_threshold"):
        cfg.validate_runtime()


def test_validate_runtime_raises_on_zero_pool():
    cfg = MemoryConfig(embedding_provider="local", redis_max_connections=0)  # type: ignore[call-arg]
    with pytest.raises(ConfigurationError, match="redis_max_connections"):
        cfg.validate_runtime()


def test_config_error_carries_details():
    cfg = MemoryConfig(embedding_provider="openai")  # type: ignore[call-arg]
    with pytest.raises(ConfigurationError) as exc_info:
        cfg.validate_runtime()
    assert "embedding_provider" in exc_info.value.details
