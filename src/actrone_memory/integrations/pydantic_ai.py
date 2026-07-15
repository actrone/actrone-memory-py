"""Pydantic AI memory adapter for actrone-memory (Tier 1).

Pydantic AI has no memory object — an agent's context comes from its system prompt (static or
a ``@agent.system_prompt`` dynamic function) plus ``message_history``. This adapter returns the
governed Actrone memory as a system-prompt string to register as a dynamic system prompt, and
persists the completed turn.

Framework-free: nothing from ``pydantic-ai`` is imported — this runs against a local,
service-free MemoryManager.
"""

from __future__ import annotations

from actrone_memory.integrations._context import BaseActroneMemory


class ActronePydanticAIMemory(BaseActroneMemory):
    """Governed memory for a Pydantic AI ``Agent`` via a dynamic system prompt.

    Usage::

        from pydantic_ai import Agent, RunContext
        from actrone_memory.integrations.pydantic_ai import ActronePydanticAIMemory

        memory = ActronePydanticAIMemory(agent_id="support-bot", session_id="s1")
        agent = Agent("openai:gpt-4o-mini")

        @agent.system_prompt
        async def with_memory(ctx: RunContext[str]) -> str:
            return await memory.system_prompt(ctx.deps)   # ctx.deps carries the user input

        result = await agent.run(user_input, deps=user_input)
        await memory.remember(user_input, result.output)
    """

    async def system_prompt(self, query: str, *, token_budget: int | None = None) -> str:
        """Return the governed memory context as a system-prompt string (``""`` when empty)."""
        return await self.build_context(query, token_budget=token_budget)
