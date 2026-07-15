"""CrewAI memory backend adapter for actrone-memory."""

from __future__ import annotations

from typing import Any

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import governed_context
from actrone_memory.manager import MemoryManager


class ActroneCrewMemory:
    """
    Drop-in CrewAI memory backend. Requires: pip install actrone-memory[crewai]

    Usage:
        from actrone_memory.integrations.crewai import ActroneCrewMemory
        memory = ActroneCrewMemory(agent_id="research-crew")
        agent = Agent(role="Researcher", memory=True, memory_backend=memory)
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str = "default",
        token_budget: int = 4096,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        try:
            import crewai  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install crewai extras: pip install actrone-memory[crewai]") from exc

        self.agent_id = agent_id
        self.session_id = session_id
        self.token_budget = token_budget
        self._config = config
        self._mm: MemoryManager | None = memory_manager

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(self, query: str, *, token_budget: int | None = None) -> str:
        """Governed system-context string for ``query`` (Tier 1, framework-free).

        The universally-correct path: prepend the returned block to a task description or agent
        backstory regardless of framework. Returns ``""`` when nothing is relevant. Complements
        the native CrewAI ``save`` / ``search`` methods below.
        """
        mm = await self._get_manager()
        return await governed_context(
            mm, self.agent_id, self.session_id, query, token_budget or self.token_budget
        )

    async def save(self, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Called by CrewAI to persist agent output."""
        content = str(value)
        task_input: str = (metadata or {}).get("task_input", "")
        mm = await self._get_manager()
        await mm.store_turn(
            agent_id=self.agent_id,
            session_id=self.session_id,
            user_message=task_input,
            assistant_message=content,
        )

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Called by CrewAI to retrieve relevant memories."""
        mm = await self._get_manager()
        memories = await mm.search_memories(self.agent_id, query, limit=limit)
        return [
            {
                "id": m.id,
                "content": m.content,
                "score": m.importance_score,
                "metadata": {
                    "session_id": m.session_id,
                    "content_type": m.content_type,
                    "timestamp": m.timestamp.isoformat(),
                    "topic_tags": m.topic_tags,
                },
            }
            for m in memories
        ]

    async def reset(self) -> None:
        """Clear all memory for this agent/session."""
        mm = await self._get_manager()
        await mm.clear_session(self.agent_id, self.session_id)
