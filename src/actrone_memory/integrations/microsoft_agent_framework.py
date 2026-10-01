"""Microsoft Agent Framework memory adapter for actrone-memory (Tier 1 + Tier 2).

Microsoft Agent Framework (the Semantic Kernel + AutoGen successor) injects memory through a
``ContextProvider``. Around each ``agent.run(...)`` the framework calls the provider's
``before_run(...)`` with a ``SessionContext`` the provider can add instructions to, and
``after_run(...)`` once ``context.response`` holds the reply. This adapter offers:

- **Tier 1**, :meth:`ActroneAgentFrameworkMemory.build_context`: the governed memory as a
  string (inherited from the shared base).
- **Tier 2**, :meth:`ActroneAgentFrameworkMemory.as_context_provider`: a real
  ``ContextProvider`` backed by the Actrone store, ready to pass as ``Agent(...,
  context_providers=[…])``. ``agent-framework`` is imported lazily inside that method only.

Targets the Agent Framework 1.x provider API (``before_run`` / ``after_run`` on a
``SessionContext``), which is stable from 1.0.0. The pre-1.0 preview API (``invoking`` /
``invoked`` returning a ``Context``) was removed at 1.0 and is not supported.
"""

from __future__ import annotations

from typing import Any

from actrone_memory.integrations._context import BaseActroneMemory

#: Default ``source_id`` for the provider, which Agent Framework uses to attribute the
#: instructions it adds (other providers can filter on it).
DEFAULT_SOURCE_ID = "actrone-memory"


class ActroneAgentFrameworkMemory(BaseActroneMemory):
    """Governed memory for a Microsoft Agent Framework ``Agent``.

    Usage (Tier 2, native ``ContextProvider``)::

        from agent_framework import Agent
        from actrone_memory.integrations.microsoft_agent_framework import (
            ActroneAgentFrameworkMemory,
        )

        memory = ActroneAgentFrameworkMemory(agent_id="support-bot", session_id="s1")
        agent = Agent(client, context_providers=[memory.as_context_provider()])
    """

    def as_context_provider(self, source_id: str = DEFAULT_SOURCE_ID) -> Any:
        """Return an Agent Framework ``ContextProvider`` backed by the Actrone store (Tier 2).

        Before each run it adds the governed memory for the latest user message as instructions;
        after a successful run it stores the user and assistant turns.

        Args:
            source_id: How Agent Framework attributes the instructions this provider adds.

        Requires ``agent-framework`` 1.x (``pip install
        actrone-memory[microsoft_agent_framework]``); imported lazily here so the module and the
        Tier-1 path work without it.
        """
        try:
            from agent_framework import ContextProvider
        except ImportError as exc:  # pragma: no cover - exercised via import guard test
            raise ImportError(
                "Install agent-framework extras: "
                "pip install actrone-memory[microsoft_agent_framework]"
            ) from exc

        outer = self

        class _ActroneContextProvider(ContextProvider):  # type: ignore[misc]
            """Adds governed memory before a run; persists the exchange after it."""

            async def before_run(
                self, *, agent: Any, session: Any, context: Any, state: dict[str, Any]
            ) -> None:
                query = _last_role_text(getattr(context, "input_messages", None), "user")
                memory = await outer.build_context(query) if query else ""
                if memory:
                    context.extend_instructions(self.source_id, memory)

            async def after_run(
                self, *, agent: Any, session: Any, context: Any, state: dict[str, Any]
            ) -> None:
                response = getattr(context, "response", None)
                if response is None:  # the run failed or produced nothing to remember
                    return
                user_text = _last_role_text(getattr(context, "input_messages", None), "user")
                assistant_text = _last_role_text(
                    getattr(response, "messages", None), "assistant"
                ) or str(getattr(response, "text", "") or "")
                if user_text and assistant_text:
                    await outer.remember(user_text, assistant_text)

        return _ActroneContextProvider(source_id)


def _message_text(message: Any) -> str:
    """Flatten an Agent Framework ``ChatMessage`` (``.text`` or ``.contents``) into text. Pure."""
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    contents = getattr(message, "contents", None) or []
    parts = [str(getattr(c, "text", "") or "") for c in contents]
    return " ".join(p for p in parts if p).strip()


def _role_value(message: Any) -> str:
    """Normalise a message's role to a lowercase string (handles enum-like ``role.value``)."""
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "").lower()


def _last_role_text(messages: Any, role: str) -> str:
    """Return the text of the last message with ``role``. Accepts a single message or a list."""
    if messages is None:
        return ""
    items = messages if isinstance(messages, list) else [messages]
    for message in reversed(items):
        if _role_value(message) == role:
            return _message_text(message)
    return ""
