"""Agno (ex-Phidata) memory adapter for actrone-memory (Tier 1).

Agno drives an ``Agent`` from ``instructions`` + ``additional_context``. This adapter returns the
governed Actrone memory as an ``additional_context`` string to hand the agent for a turn, and
persists the completed turn. Agno's own memory/db is a storage subsystem rather than a simple
injectable interface, so the ``additional_context`` path is the idiomatic Tier-1 integration.

Framework-free: nothing from ``agno`` is imported — this runs against a local, service-free
MemoryManager.
"""

from __future__ import annotations

from actrone_memory.integrations._context import BaseActroneMemory


class ActroneAgnoMemory(BaseActroneMemory):
    """Governed memory for an Agno ``Agent``.

    Usage::

        from agno.agent import Agent
        from actrone_memory.integrations.agno import ActroneAgnoMemory

        memory = ActroneAgnoMemory(agent_id="support-bot", session_id="s1")
        context = await memory.additional_context(user_input)
        agent = Agent(model=model, additional_context=context)
        response = await agent.arun(user_input)
        await memory.remember(user_input, response.content)
    """

    async def additional_context(self, query: str, *, token_budget: int | None = None) -> str:
        """Return the governed memory context to pass as an Agno agent's ``additional_context``
        for the turn (``""`` when nothing relevant)."""
        return await self.build_context(query, token_budget=token_budget)
