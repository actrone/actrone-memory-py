"""Semantic Kernel memory adapter for actrone-memory (Tier 1 + Tier 2).

Semantic Kernel drives a chat agent from a ``ChatHistory``. This adapter offers:

- **Tier 1**, :meth:`ActroneSemanticKernelMemory.system_message`: the governed memory as a
  string to add as a system message before invoking the kernel.
- **Tier 2**, :meth:`ActroneSemanticKernelMemory.add_to_chat_history`: populate a real SK
  ``ChatHistory`` in place via its ``add_system_message`` method (duck-typed, no import of
  ``semantic-kernel`` at runtime, so the adapter stays dependency-free and testable with a fake).

Then persist the completed turn with :meth:`remember`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from actrone_memory.integrations._context import BaseActroneMemory


@runtime_checkable
class _SupportsSystemMessage(Protocol):
    """The subset of Semantic Kernel's ``ChatHistory`` this adapter needs (duck-typed)."""

    def add_system_message(self, content: str) -> Any: ...


class ActroneSemanticKernelMemory(BaseActroneMemory):
    """Governed memory for a Semantic Kernel chat agent.

    Usage::

        from semantic_kernel.contents import ChatHistory
        from actrone_memory.integrations.semantic_kernel import ActroneSemanticKernelMemory

        memory = ActroneSemanticKernelMemory(agent_id="support-bot", session_id="s1")
        history = ChatHistory()
        await memory.add_to_chat_history(history, user_input)   # Tier 2, native ChatHistory
        history.add_user_message(user_input)
        # ...invoke the kernel/agent with `history`...
        await memory.remember(user_input, answer)
    """

    async def system_message(self, query: str, *, token_budget: int | None = None) -> str:
        """Tier 1: the governed memory context as a system-message string (``""`` when empty)."""
        return await self.build_context(query, token_budget=token_budget)

    async def add_to_chat_history(
        self, chat_history: _SupportsSystemMessage, query: str, *, token_budget: int | None = None
    ) -> bool:
        """Tier 2: add the governed memory to a Semantic Kernel ``ChatHistory`` as a system
        message, in place. Returns ``True`` when a message was added, ``False`` when there was
        nothing relevant to add (so the caller can branch)."""
        context = await self.build_context(query, token_budget=token_budget)
        if not context:
            return False
        chat_history.add_system_message(context)
        return True
