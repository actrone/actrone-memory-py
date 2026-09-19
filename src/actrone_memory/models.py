from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


ContentType = Literal["turn", "summary", "tool_result", "injected", "fact"]

# ── Provenance-typing v1 (the governance seed that graduates to hosted) ──────
# Every stored fact carries *where it came from* and *how sensitive it is*, so a
# memory can be filtered, attributed, and erased by policy, even in the free,
# local library. These vocabularies are the language-neutral memory spec shared
# with the TypeScript lib and the hosted engine; keep the two enums in lockstep.

# Origin/attribution of a memory. Free-form callers may also pass a namespaced
# string (e.g. "tool:web_search", "import:crm"), the typed values are the
# canonical set; anything else is accepted as an opaque source label.
MemorySource = Literal[
    "user",  # stated by the end user
    "assistant",  # asserted by the agent
    "tool",  # produced by a tool call
    "summary",  # distilled from a conversation summary
    "injected",  # seeded directly via inject_memory()
    "extracted",  # derived by fact extraction
    "reflection",  # synthesised by a reflection pass
    "imported",  # loaded from an external system
    "unknown",  # provenance not recorded
]

# Sensitivity classification for governance / right-to-erasure. Ordered from
# least to most sensitive. Mirrors the hosted DPE tiers conceptually so a fact's
# handling policy is consistent from the OSS wedge up to the governed platform.
Sensitivity = Literal[
    "none",  # non-personal, freely retained
    "low",  # mildly personal / preference data
    "pii",  # personally identifiable information
    "sensitive",  # special-category / regulated (health, financial, credentials)
]


class ToolResult(BaseModel):
    tool_name: str
    params: dict[str, object]
    result: object
    success: bool
    duration_ms: int | None = None


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    session_id: str
    content: str
    content_type: ContentType
    embedding: list[float] | None = None
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    topic_tags: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=_utcnow)
    source_turn_ids: list[str] = Field(default_factory=list)
    # ── Provenance-typing v1 ────────────────────────────────────────────
    # Where the fact came from (attribution) and how sensitive it is. Defaults
    # are backwards-compatible: pre-existing/untagged memories read as
    # source="unknown", sensitivity="none".
    source: str = "unknown"
    sensitivity: Sensitivity = "none"


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_message: str
    assistant_message: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)
    token_count: int = Field(default=0, ge=0)


class RetrievedContext(BaseModel):
    """Output of MemoryManager.retrieve_context, ready to inject into an LLM prompt."""

    recent_turns: list[Turn]
    episodic_memories: list[MemoryEntry]
    total_tokens_used: int
    token_budget: int
    retrieval_duration_ms: float

    @property
    def budget_utilisation(self) -> float:
        if self.token_budget == 0:
            return 0.0
        return self.total_tokens_used / self.token_budget


class SessionMetadata(BaseModel):
    agent_id: str
    session_id: str
    turn_count: int
    created_at: datetime | None = None
    last_active: datetime | None = None
