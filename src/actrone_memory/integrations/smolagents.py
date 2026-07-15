"""smolagents (Hugging Face) memory adapter for actrone-memory (Tier 1).

smolagents runs a `CodeAgent` / `ToolCallingAgent` over a task string. This adapter returns the
governed Actrone memory as a context block to prepend to the task (or add via `additional_args`) for
a run, and persists the completed turn. smolagents keeps its own step memory rather than exposing a
simple injectable memory interface, so the task-context path is the idiomatic Tier-1 integration.

Framework-free: nothing from ``smolagents`` is imported — this runs against a local, service-free
MemoryManager.
"""

from __future__ import annotations

from actrone_memory.integrations._context import BaseActroneMemory


class ActroneSmolagentsMemory(BaseActroneMemory):
    """Governed memory for a smolagents agent.

    Usage::

        from smolagents import CodeAgent
        from actrone_memory.integrations.smolagents import ActroneSmolagentsMemory

        memory = ActroneSmolagentsMemory(agent_id="support-bot", session_id="s1")
        task = await memory.task_context(user_input) + user_input
        result = agent.run(task)
        await memory.remember(user_input, str(result))
    """

    async def task_context(self, query: str, *, token_budget: int | None = None) -> str:
        """Return a governed memory block to prepend to the agent's task (``""`` when empty).

        Includes a trailing separator when non-empty so it composes cleanly ahead of the task text.
        """
        context = await self.build_context(query, token_budget=token_budget)
        return f"{context}\n\n---\n\n" if context else ""
