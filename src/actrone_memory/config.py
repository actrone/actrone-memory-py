from __future__ import annotations

from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from actrone_memory.exceptions import ConfigurationError


class MemoryConfig(BaseSettings):
    """All configuration for actrone-memory.

    Values are read from environment variables with the ``ACTRONE_`` prefix.
    You can also pass values directly when constructing this object.

    Required when ``embedding_provider="openai"`` (the default):
        ACTRONE_OPENAI_API_KEY

    All other settings have sensible defaults.
    """

    model_config = SettingsConfigDict(env_prefix="ACTRONE_", env_file=".env", extra="ignore")

    # ── Backend ──────────────────────────────────────────────────────────
    # "memory" (default) is the zero-service, in-process local-first backend —
    # no Redis, no Qdrant, no API key — parity with the TypeScript on-ramp.
    # "redis_qdrant" is the durable, horizontally-scalable production backend.
    backend: Literal["memory", "redis_qdrant"] = "memory"

    # ── Store connections (used only when backend="redis_qdrant") ─────────
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    # ── Redis connection pool ────────────────────────────────────────────
    # Raise max_connections under high concurrency (many concurrent agents).
    redis_max_connections: int = 10
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0

    # ── Qdrant client ────────────────────────────────────────────────────
    qdrant_timeout: float = 10.0

    # ── Embedding ────────────────────────────────────────────────────────
    # "hashing" (default): dependency-free, deterministic, offline, no API key.
    # "openai": text-embedding-3-small (needs ACTRONE_OPENAI_API_KEY).
    # "local": sentence-transformers all-MiniLM-L6-v2 (needs the [local] extra).
    embedding_provider: Literal["openai", "local", "hashing"] = "hashing"
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    # Vector width for the hashing embedder (matches the TS LocalEmbedder default).
    hashing_dimensions: int = 256
    embedding_cache_ttl_seconds: int = 604800  # 7 days

    # ── Summarisation ────────────────────────────────────────────────────
    # Model used for LLM-based conversation summarisation (OpenAI only).
    # GPT-4o-mini is cheap (~$0.00003 per call) and fast enough for background use.
    summarisation_model: str = "gpt-4o-mini"

    # ── Session (L1 Redis) ───────────────────────────────────────────────
    session_ttl_hours: int = 24
    max_session_turns: int = 50

    # ── Episodic memory (L2 Qdrant) ──────────────────────────────────────
    qdrant_collection: str = "agent_memories"
    max_episodic_memories: int = 500
    relevance_threshold: float = 0.72
    relevance_weight: float = 0.7  # recency_weight = 1 - relevance_weight
    recency_weight: float = 0.3

    # ── Fact extraction (LLM-gated, opt-in) ──────────────────────────────
    # When True *and* an LLM embedder/summariser is configured (OpenAI provider),
    # durable atomic facts are extracted from a session's turns and stored as
    # first-class memories (content_type="fact", source="extracted"). Off by
    # default because it costs one LLM call per extraction. Rides the same cadence
    # as auto-summarisation to stay cost-bounded.
    extract_facts: bool = False

    # ── Auto-summarisation ───────────────────────────────────────────────
    auto_summarise: bool = True
    summarise_after_turns: int = 20
    # L1 caps the turn list at max_session_turns, so once a session reaches
    # summarise_after_turns the count stays above the threshold and would
    # otherwise spawn a summarisation on *every* subsequent turn. This cooldown
    # is the TTL of a Redis SET-NX lock that both deduplicates concurrent/fleet
    # summarisations of the same session and throttles re-summarisation.
    summarise_cooldown_seconds: int = 300

    # ── Background-task error handling ────────────────────────────────────
    # When True, background-task failures (e.g. _summarise_session) re-raise
    # so test suites catch regressions deterministically. Production code
    # should leave this False and rely on metrics / structured logs.
    strict_background_errors: bool = False

    # ── Graceful shutdown ─────────────────────────────────────────────────
    # Maximum time MemoryManager.close() waits for in-flight background
    # tasks to drain before cancelling them. CLAUDE.md §6.1 mandates a
    # configurable grace period; 30 s matches the workspace default.
    shutdown_grace_seconds: float = 30.0

    # ── Token budget fractions (must sum to 1.0) ─────────────────────────
    budget_fraction_system: float = 0.30
    budget_fraction_episodic: float = 0.25
    budget_fraction_session: float = 0.35
    budget_fraction_current_turn: float = 0.10

    @field_validator("embedding_provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        return v  # cross-field validation done in validate_runtime()

    def validate_runtime(self) -> None:
        """Validate settings that require cross-field checks.

        Call this before opening connections. Raises ConfigurationError with a
        clear message if any required setting is missing or inconsistent.
        """
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "ACTRONE_OPENAI_API_KEY is required when embedding_provider='openai'.",
                details={"embedding_provider": self.embedding_provider},
            )

        budget_total = (
            self.budget_fraction_system
            + self.budget_fraction_episodic
            + self.budget_fraction_session
            + self.budget_fraction_current_turn
        )
        if abs(budget_total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"Token budget fractions must sum to 1.0, got {budget_total:.4f}.",
                details={"sum": budget_total},
            )

        if not (0.0 < self.relevance_threshold < 1.0):
            raise ConfigurationError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}.",
                details={"relevance_threshold": self.relevance_threshold},
            )

        if self.redis_max_connections < 1:
            raise ConfigurationError(
                "redis_max_connections must be ≥ 1.",
                details={"redis_max_connections": self.redis_max_connections},
            )
