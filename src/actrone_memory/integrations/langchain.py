"""LangChain adapters for actrone-memory.

Two Tier-2 surfaces, because LangChain changed shape between major lines:

* :class:`ActroneMemory` implements ``langchain_core.memory.BaseMemory``. That module was
  **removed in LangChain 1.x**, so this class only works on the 0.x line and raises a clear
  error elsewhere.
* :class:`ActroneChatMessageHistory` implements the ``BaseChatMessageHistory`` surface, which
  is unchanged across 0.x and 1.x. Use it with ``RunnableWithMessageHistory``. This is the
  adapter to reach for on a current LangChain install.

Both are structural (duck-typed) rather than subclasses, so importing this module never
imports LangChain. Passing them to APIs typed against the abstract classes may need a cast.
The framework-free ``build_context()`` on each works on every LangChain version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import governed_context
from actrone_memory.manager import MemoryManager

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

_SYNC_UNSUPPORTED = (
    "{name} is async-only because the underlying store is async. "
    "Use `await {alt}` instead, and drive LangChain through its async API "
    "(ainvoke / astream)."
)


class ActroneMemory:
    """Drop-in LangChain ``BaseMemory`` backend. Requires ``actrone-memory[langchain]``.

    **LangChain 0.x only.** ``langchain_core.memory`` was removed in LangChain 1.x, which
    retired the ``BaseMemory`` abstraction, so there is nothing for this class to implement
    there. On 1.x use :class:`ActroneChatMessageHistory`, the LangGraph checkpointer adapter
    (``actrone_memory.integrations.langgraph``), or ``build_context()`` below, which is
    framework-free and works on every version.

    Usage::

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
            # Distinguish "LangChain absent" from "LangChain 1.x, which dropped BaseMemory",
            # because the fix is completely different for each. The version probe is kept out
            # of the raise so this except clause cannot swallow the error we are raising.
            try:
                import langchain_core

                installed: str | None = getattr(langchain_core, "__version__", "unknown")
            except ImportError:
                installed = None

            if installed is None:
                raise ImportError(
                    "Install langchain extras: pip install actrone-memory[langchain]"
                ) from exc
            raise ImportError(
                f"langchain_core {installed} has no `langchain_core.memory`: LangChain "
                "removed the BaseMemory abstraction in 1.x, so ActroneMemory cannot be "
                "used on this version. Use ActroneChatMessageHistory (same module) with "
                "RunnableWithMessageHistory, the LangGraph checkpointer adapter, or the "
                "framework-free build_context(). Pin langchain-core<1 to keep using "
                "ActroneMemory."
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

    async def build_context(self, query: str, *, token_budget: int | None = None) -> str:
        """Governed system-context string for ``query`` (Tier 1, framework-free).

        The universally-correct path: prepend the returned block to any prompt regardless of
        framework. Returns ``""`` when nothing is relevant. Complements the native LangChain
        ``BaseMemory`` methods below.
        """
        mm = await self._get_manager()
        return await governed_context(
            mm, self.agent_id, self.session_id, query, token_budget or self.token_budget
        )

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


def _message_role(message: Any) -> str:  # noqa: ANN401 - duck-typed LangChain message
    """Best-effort role for a LangChain message across versions.

    ``BaseMessage.type`` is the documented attribute; ``_get_type()`` is the older private
    accessor some community message classes still define. Falls back to the class name so an
    unknown message is skipped rather than mis-stored.
    """
    role = getattr(message, "type", None)
    if isinstance(role, str) and role:
        return role
    getter = getattr(message, "_get_type", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001 - a broken message must not break persistence
            value = None
        if isinstance(value, str) and value:
            return value
    name = type(message).__name__.lower()
    if "human" in name or "user" in name:
        return "human"
    if "ai" in name or "assistant" in name:
        return "ai"
    return "unknown"


def _message_text(message: Any) -> str:  # noqa: ANN401 - duck-typed LangChain message
    """Flatten a message's content to text, including 1.x content-block lists."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # LangChain 1.x allows a list of content blocks; keep only the text parts.
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


class ActroneChatMessageHistory:
    """Governed ``BaseChatMessageHistory`` for ``RunnableWithMessageHistory``.

    Works on **both** LangChain 0.x and 1.x: ``langchain_core.chat_history`` is unchanged
    across the major bump, unlike ``langchain_core.memory``. Mirrors the TypeScript
    ``langchainChatHistory`` adapter so both libraries behave identically.

    Turn pairing: a human message is held until the next AI message, then the pair is stored
    as one turn. Two human messages in a row store the first with an empty response rather
    than dropping it. System and tool messages are not persisted as turns.

    Async-only, because the store is. Use it through LangChain's async API and the
    ``aget_messages`` / ``aadd_messages`` / ``aclear`` methods; the synchronous members raise
    with a pointer to their async counterpart instead of silently blocking an event loop.

    Usage::

        history = ActroneChatMessageHistory(agent_id="my-agent", session_id="session-1")
        chain = RunnableWithMessageHistory(runnable, lambda _: history)  # may need a cast
        await chain.ainvoke({"input": "hello"}, config={"configurable": {"session_id": "s"}})
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
            from langchain_core.chat_history import BaseChatMessageHistory  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install langchain extras: pip install actrone-memory[langchain]"
            ) from exc

        self.agent_id = agent_id
        self.session_id = session_id
        self.token_budget = token_budget
        self._config = config
        self._mm: MemoryManager | None = memory_manager
        # A human message with no paired AI reply yet; flushed on the next AI message.
        self._pending_human: str | None = None

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(self, query: str, *, token_budget: int | None = None) -> str:
        """Governed system-context string for ``query`` (Tier 1, framework-free)."""
        mm = await self._get_manager()
        return await governed_context(
            mm, self.agent_id, self.session_id, query, token_budget or self.token_budget
        )

    # ── async surface (the supported one) ────────────────────────────────

    async def aget_messages(self) -> list[Any]:
        """Recent session turns as alternating Human/AI messages, oldest first."""
        from langchain_core.messages import AIMessage, HumanMessage

        mm = await self._get_manager()
        turns = await mm.get_recent_turns(self.agent_id, self.session_id)
        messages: list[Any] = []
        for turn in turns:
            messages.append(HumanMessage(content=turn.user_message))
            if turn.assistant_message:
                messages.append(AIMessage(content=turn.assistant_message))
        return messages

    async def aadd_messages(self, messages: Sequence[Any]) -> None:
        """Persist messages, pairing each human with the AI reply that follows it."""
        for message in messages:
            await self._add_one(message)

    async def aadd_message(self, message: Any) -> None:  # noqa: ANN401 - duck-typed
        """Persist a single message. Convenience alias for one-item ``aadd_messages``."""
        await self._add_one(message)

    async def aclear(self) -> None:
        """Drop the session's turns and any unpaired human message."""
        self._pending_human = None
        mm = await self._get_manager()
        await mm.clear_session(self.agent_id, self.session_id)

    async def _add_one(self, message: Any) -> None:  # noqa: ANN401 - duck-typed
        role = _message_role(message)
        text = _message_text(message)
        mm = await self._get_manager()

        if role == "human":
            if self._pending_human is not None:
                # Two humans in a row: store the first alone rather than losing it.
                await mm.store_turn(self.agent_id, self.session_id, self._pending_human, "")
            self._pending_human = text
        elif role == "ai":
            await mm.store_turn(self.agent_id, self.session_id, self._pending_human or "", text)
            self._pending_human = None
        # System and tool messages are context, not conversation turns.

    # ── synchronous surface (unsupported, fails loudly) ──────────────────

    @property
    def messages(self) -> list[Any]:
        raise NotImplementedError(
            _SYNC_UNSUPPORTED.format(
                name="ActroneChatMessageHistory.messages", alt="aget_messages()"
            )
        )

    def add_messages(self, messages: Sequence[Any]) -> None:
        raise NotImplementedError(
            _SYNC_UNSUPPORTED.format(
                name="ActroneChatMessageHistory.add_messages", alt="aadd_messages(...)"
            )
        )

    def add_message(self, message: Any) -> None:  # noqa: ANN401 - duck-typed
        raise NotImplementedError(
            _SYNC_UNSUPPORTED.format(
                name="ActroneChatMessageHistory.add_message", alt="aadd_messages([...])"
            )
        )

    def clear(self) -> None:
        raise NotImplementedError(
            _SYNC_UNSUPPORTED.format(name="ActroneChatMessageHistory.clear", alt="aclear()")
        )
