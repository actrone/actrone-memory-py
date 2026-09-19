"""Microsoft Agent Framework memory adapter for actrone-memory (Tier 1 + Tier 2).

Microsoft Agent Framework (the Semantic Kernel + AutoGen successor) injects memory through a
``ContextProvider``: before each ``agent.run(...)`` the framework calls ``invoking(messages=…)``
to fetch context (returned as a ``Context`` with ``instructions``), and after the response calls
``invoked(...)`` to persist the exchange. This adapter offers:

- **Tier 1**, :meth:`ActroneAgentFrameworkMemory.build_context`: the governed memory as a
  string (inherited from the shared base).
- **Tier 2**, :meth:`ActroneAgentFrameworkMemory.as_context_provider`: a real
  ``ContextProvider`` backed by the Actrone store, ready to pass as ``ChatAgent(...,
  context_providers=[…])``. ``agent-framework`` is imported lazily inside that method only.
"""

from __future__ import annotations

from typing import Any

from actrone_memory.integrations._context import BaseActroneMemory


class ActroneAgentFrameworkMemory(BaseActroneMemory):
    """Governed memory for a Microsoft Agent Framework ``ChatAgent``.

    Usage (Tier 2, native ``ContextProvider``)::

        from agent_framework import ChatAgent
        from actrone_memory.integrations.microsoft_agent_framework import (
            ActroneAgentFrameworkMemory,
        )

        memory = ActroneAgentFrameworkMemory(agent_id="support-bot", session_id="s1")
        agent = ChatAgent(chat_client=client, context_providers=[memory.as_context_provider()])
    """

    def as_context_provider(self) -> Any:
        """Return an Agent Framework ``ContextProvider`` backed by the Actrone store (Tier 2).

        Requires ``agent-framework`` (``pip install
        actrone-memory[microsoft_agent_framework]``); imported lazily here so the module and the
        Tier-1 path work without it.
        """
        try:
            from agent_framework import Context, ContextProvider
        except ImportError as exc:  # pragma: no cover - exercised via import guard test
            raise ImportError(
                "Install agent-framework extras: "
                "pip install actrone-memory[microsoft_agent_framework]"
            ) from exc

        outer = self

        class _ActroneContextProvider(ContextProvider):  # type: ignore[misc]
            """Fetches governed memory before invocation; persists the exchange after."""

            async def invoking(self, messages: Any, **_kwargs: Any) -> Any:
                query = _last_role_text(messages, "user")
                context = await outer.build_context(query) if query else ""
                return Context(instructions=context or None)

            async def invoked(
                self,
                request_messages: Any,
                response_messages: Any = None,
                invoke_exception: BaseException | None = None,
                **_kwargs: Any,
            ) -> None:
                if invoke_exception is not None:
                    return
                user_text = _last_role_text(request_messages, "user")
                assistant_text = _last_role_text(response_messages, "assistant")
                if user_text and assistant_text:
                    await outer.remember(user_text, assistant_text)

        return _ActroneContextProvider()


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
