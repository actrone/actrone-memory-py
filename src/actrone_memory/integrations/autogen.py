"""AutoGen 0.4 Memory protocol adapter for actrone-memory.

Implements the ``autogen_core.memory.Memory`` protocol so any AutoGen 0.4
agent can use actrone-memory as its persistent backend with no code changes.

Install::

    pip install actrone-memory[autogen]

Usage::

    from actrone_memory.integrations.autogen import ActroneAutoGenMemory

    memory = ActroneAutoGenMemory(agent_id="research-agent", session_id="session-1")
    agent = AssistantAgent("researcher", memory=[memory])

Requires autogen-agentchat >= 0.4 and autogen-core >= 0.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from actrone_memory.config import MemoryConfig
from actrone_memory.manager import MemoryManager

if TYPE_CHECKING:
    from autogen_core.memory import (
        MemoryContent,
        MemoryQueryResult,
        UpdateContextResult,
    )
    from autogen_core.model_context import ChatCompletionContext


def _require_autogen() -> None:
    try:
        import autogen_core.memory  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install AutoGen extras: pip install actrone-memory[autogen]") from exc


class ActroneAutoGenMemory:
    """AutoGen 0.4 Memory protocol implementation backed by actrone-memory.

    Implements ``autogen_core.memory.Memory`` — drop-in for any AutoGen 0.4 agent.

    Args:
        agent_id: Unique identifier for the agent. Used as the memory namespace.
        session_id: Conversation/thread identifier for L1 Redis session memory.
        token_budget: Maximum tokens injected into the model context per turn.
        config: Optional ``MemoryConfig``; falls back to environment variables.
        memory_manager: Pre-constructed ``MemoryManager`` (avoids re-initialisation).
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str = "default",
        token_budget: int = 4096,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        _require_autogen()
        self._agent_id = agent_id
        self._session_id = session_id
        self._token_budget = token_budget
        self._config = config
        self._mm: MemoryManager | None = memory_manager

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Any | None = None,  # noqa: ANN401
    ) -> None:
        """Persist a memory entry. Called by AutoGen after each agent turn."""
        from autogen_core.memory import MemoryMimeType

        if content.mime_type not in (MemoryMimeType.TEXT, None):
            # Non-text content (images, etc.) is not supported by actrone-memory.
            return

        text = content.content if isinstance(content.content, str) else str(content.content)
        importance: float = float((content.metadata or {}).get("importance", 0.5))
        tags: list[str] = list((content.metadata or {}).get("tags", []))

        mm = await self._get_manager()
        await mm.inject_memory(self._agent_id, text, importance=importance, topic_tags=tags)

    async def query(
        self,
        query: Any,  # noqa: ANN401  — autogen_core.memory.MemoryQuery
        cancellation_token: Any | None = None,  # noqa: ANN401
    ) -> MemoryQueryResult:
        """Return memories relevant to the query text."""
        from autogen_core.memory import MemoryContent, MemoryMimeType, MemoryQueryResult

        query_text: str = query.text if hasattr(query, "text") else str(query)
        n_results: int = getattr(query, "n_results", 10)

        mm = await self._get_manager()
        memories = await mm.search_memories(self._agent_id, query_text, limit=n_results)

        results: list[MemoryContent] = [
            MemoryContent(
                content=m.content,
                mime_type=MemoryMimeType.TEXT,
                metadata={
                    "memory_id": m.id,
                    "importance_score": m.importance_score,
                    "timestamp": m.timestamp.isoformat(),
                    "topic_tags": m.topic_tags,
                },
            )
            for m in memories
        ]
        return MemoryQueryResult(results=results)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject relevant memories into the model context as system messages.

        Called by AutoGen before each model call. Retrieves the most recent turn
        from the context as the query, then injects top-k episodic memories.
        """
        from autogen_core.memory import UpdateContextResult
        from autogen_core.models import SystemMessage

        messages = await model_context.get_messages()
        query = ""
        if messages:
            last = messages[-1]
            query = last.content if isinstance(last.content, str) else str(last.content)

        mm = await self._get_manager()
        ctx = await mm.retrieve_context(
            self._agent_id, self._session_id, query or "context", self._token_budget
        )

        injected = 0
        if ctx.episodic_memories:
            memory_text = "\n".join(f"[Memory] {m.content}" for m in ctx.episodic_memories)
            await model_context.add_message(SystemMessage(content=memory_text))
            injected += len(ctx.episodic_memories)

        return UpdateContextResult(memories_added=injected)

    async def clear(self) -> None:
        """Clear all session memory for this agent."""
        mm = await self._get_manager()
        await mm.clear_session(self._agent_id, self._session_id)

    async def close(self) -> None:
        """Release resources. No-op when MemoryManager is managed externally."""
