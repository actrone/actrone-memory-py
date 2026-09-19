"""OpenAI Agents SDK memory adapter for actrone-memory (Tier 1).

The OpenAI Agents SDK carries an agent's context in its ``instructions``. This adapter returns
the ``instructions`` to give an ``Agent`` for a turn, the developer's base instructions with
governed Actrone memory prepended, and persists the completed turn. The Agents SDK has no
formal memory object to conform to (its ``Session`` is a transcript store, not an injectable
memory interface), so the system-instructions path is the idiomatic integration.

Framework-free: nothing from ``openai-agents`` is imported, this runs against a local,
service-free MemoryManager. Mirrors the TypeScript ``openaiAgentsMemory`` adapter.
"""

from __future__ import annotations

from actrone_memory.integrations._context import BaseActroneMemory


class ActroneOpenAIAgentsMemory(BaseActroneMemory):
    """Governed memory for an OpenAI Agents SDK ``Agent``.

    Usage::

        from agents import Agent, Runner
        from actrone_memory.integrations.openai_agents import ActroneOpenAIAgentsMemory

        memory = ActroneOpenAIAgentsMemory(agent_id="support-bot", session_id="s1")
        instructions = await memory.instructions_for("You are a support agent.", user_input)
        result = await Runner.run(Agent(name="support", instructions=instructions), user_input)
        await memory.remember(user_input, result.final_output)
    """

    async def instructions_for(
        self, base_instructions: str, query: str, *, token_budget: int | None = None
    ) -> str:
        """Return the agent ``instructions`` for the turn: governed memory prepended to
        ``base_instructions`` (or just the memory when there are no base instructions)."""
        context = await self.build_context(query, token_budget=token_budget)
        if not context:
            return base_instructions
        return f"{context}\n\n{base_instructions}" if base_instructions else context
