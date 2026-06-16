from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


ContentType = Literal["turn", "summary", "tool_result", "injected"]


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


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_message: str
    assistant_message: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)
    token_count: int = Field(default=0, ge=0)


class RetrievedContext(BaseModel):
    """Output of MemoryManager.retrieve_context — ready to inject into an LLM prompt."""

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
