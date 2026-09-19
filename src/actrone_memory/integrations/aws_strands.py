"""AWS Strands memory adapter for actrone-memory (Tier 1).

Strands drives an ``Agent`` from a ``system_prompt``. This adapter returns the ``system_prompt`` to
give the agent for a turn: the developer's base system prompt with governed Actrone memory
appended, and persists the completed turn. Strands' session/memory is a storage subsystem rather
than a simple injectable interface, so the system-prompt path is the idiomatic Tier-1 integration.

Framework-free: nothing from ``strands`` is imported, this runs against a local, service-free
MemoryManager.
"""

from __future__ import annotations

from actrone_memory.integrations._context import BaseActroneMemory


class ActroneStrandsMemory(BaseActroneMemory):
    """Governed memory for an AWS Strands ``Agent``.

    Usage::

        from strands import Agent
        from actrone_memory.integrations.aws_strands import ActroneStrandsMemory

        memory = ActroneStrandsMemory(agent_id="support-bot", session_id="s1")
        system_prompt = await memory.system_prompt("You are a support agent.", user_input)
        agent = Agent(model=model, system_prompt=system_prompt)
        result = agent(user_input)
        await memory.remember(user_input, str(result))
    """

    async def system_prompt(
        self, base_system_prompt: str, query: str, *, token_budget: int | None = None
    ) -> str:
        """Return the turn's system prompt: governed memory appended to ``base_system_prompt``
        (or just the memory when there is no base system prompt)."""
        context = await self.build_context(query, token_budget=token_budget)
        if not context:
            return base_system_prompt
        return f"{base_system_prompt}\n\n{context}" if base_system_prompt else context
