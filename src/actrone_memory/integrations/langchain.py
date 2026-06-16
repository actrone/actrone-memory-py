"""LangChain BaseMemory adapter for actrone-memory."""

from __future__ import annotations

from typing import Any

from actrone_memory.config import MemoryConfig
from actrone_memory.manager import MemoryManager


class ActroneMemory:
    """
    Drop-in LangChain memory backend. Requires: pip install actrone-memory[langchain]

    Usage:
        from actrone_memory.integrations.langchain import ActroneMemory
        memory = ActroneMemory(agent_id="my-agent", session_id="session-1")
        chain = ConversationChain(llm=llm, memory=memory)
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        token_budget: int = 4096,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        try:
            from langchain_core.memory import BaseMemory  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install langchain extras: pip install actrone-memory[langchain]"
            ) from exc

        self.agent_id = agent_id
        self.session_id = session_id
        self.token_budget = token_budget
        self._config = config
        self._mm: MemoryManager | None = memory_manager
        self._memory_variables = ["history"]

    @property
    def memory_variables(self) -> list[str]:
        return self._memory_variables

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Called by LangChain before each LLM call to inject memory context."""
        query = inputs.get("input", inputs.get("human_input", ""))
        mm = await self._get_manager()
        ctx = await mm.retrieve_context(self.agent_id, self.session_id, query, self.token_budget)

        history_lines: list[str] = []
        for turn in ctx.recent_turns:
            history_lines.append(f"Human: {turn.user_message}")
            history_lines.append(f"AI: {turn.assistant_message}")

        if ctx.episodic_memories:
            history_lines.append("\n[Relevant memories]")
            for mem in ctx.episodic_memories:
                history_lines.append(f"- {mem.content}")

        return {"history": "\n".join(history_lines)}

    async def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Called by LangChain after each LLM call to persist the turn."""
        user_msg = inputs.get("input", inputs.get("human_input", ""))
        assistant_msg = outputs.get("response", outputs.get("output", ""))
        mm = await self._get_manager()
        await mm.store_turn(self.agent_id, self.session_id, user_msg, assistant_msg)

    async def clear(self) -> None:
        mm = await self._get_manager()
        await mm.clear_session(self.agent_id, self.session_id)
