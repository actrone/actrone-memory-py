"""LangGraph checkpointer adapter for actrone-memory.

``ActroneCheckpointer`` is a real LangGraph ``BaseCheckpointSaver``, so it is accepted by
``builder.compile(checkpointer=...)`` on every supported LangGraph version. It works in two layers:

- **Graph state** (checkpoints, pending writes, versions) is handled by a wrapped LangGraph saver,
  so resume, interrupts and time travel behave exactly as LangGraph defines them. The default is
  LangGraph's in-memory saver; pass ``saver=`` a durable one (for example a Postgres saver) to keep
  graph state across restarts.
- **Memory**: after each checkpoint written through the async API (``ainvoke`` / ``astream``), the
  latest completed user and AI exchange in the ``messages`` channel is stored in actrone-memory, so
  it becomes governed long-term memory that :meth:`ActroneCheckpointer.build_context` recalls.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from typing import Any

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import governed_context
from actrone_memory.manager import MemoryManager

_INSTALL_HINT = "Install langgraph extras: pip install actrone-memory[langgraph]"

# The async checkpoint write hook, wrapped to also record the exchange in memory.
_RECORDING_METHODS = frozenset({"aput"})


class ActroneCheckpointer:
    """Drop-in LangGraph checkpointer that also records conversation turns in actrone-memory.

    Requires ``pip install actrone-memory[langgraph]``.

    Usage::

        from actrone_memory.integrations.langgraph import ActroneCheckpointer

        checkpointer = ActroneCheckpointer(agent_id="my-agent")
        graph = builder.compile(checkpointer=checkpointer)
        await graph.ainvoke(state, config={"configurable": {"thread_id": "s1"}})

    Constructing it returns an instance of a ``BaseCheckpointSaver`` subclass (built on first use,
    so this module imports without LangGraph installed). Turns are recorded on the async path; the
    sync API (``invoke``) keeps graph state only, because actrone-memory is async.
    """

    agent_id: str
    token_budget: int
    _config: MemoryConfig | None
    _mm: MemoryManager | None
    _saver: Any
    _recorded: dict[str, str]

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is ActroneCheckpointer:
            cls = _checkpoint_saver_class()
        return super().__new__(cls)

    def __init__(
        self,
        agent_id: str,
        token_budget: int = 4096,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
        saver: Any | None = None,
    ) -> None:
        """Wrap ``saver`` (LangGraph's in-memory saver by default); record turns for ``agent_id``.

        Args:
            agent_id: The actrone-memory scope that recorded turns and recalled memory belong to.
            token_budget: Default token budget for :meth:`build_context`.
            config: Memory configuration, used when no ``memory_manager`` is given.
            memory_manager: An existing manager to share; one is created on first use otherwise.
            saver: The LangGraph ``BaseCheckpointSaver`` that stores graph state.
        """
        saver_base, default_saver = _langgraph_savers()
        self._saver = saver if saver is not None else default_saver()
        if not isinstance(self._saver, saver_base):
            raise TypeError(
                f"saver must be a LangGraph BaseCheckpointSaver, got {type(self._saver).__name__}"
            )
        # BaseCheckpointSaver.__init__ sets the serializer; reuse the wrapped saver's so both agree.
        super().__init__(serde=getattr(self._saver, "serde", None))  # type: ignore[call-arg]
        self.agent_id = agent_id
        self.token_budget = token_budget
        self._config = config
        self._mm = memory_manager
        self._recorded = {}

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(
        self, query: str, *, thread_id: str = "default", token_budget: int | None = None
    ) -> str:
        """Governed system-context string for ``query`` (Tier 1, framework-free).

        Prepend the returned block to a graph node's prompt. ``thread_id`` scopes recent-turn
        recency (LangGraph's session equivalent). Returns ``""`` when nothing is relevant.
        """
        mm = await self._get_manager()
        return await governed_context(
            mm, self.agent_id, thread_id, query, token_budget or self.token_budget
        )

    async def _record_exchange(self, config: dict[str, Any], checkpoint: dict[str, Any]) -> None:
        """Store the newest completed user and AI exchange in this checkpoint, once per thread."""
        thread_id = str((config.get("configurable") or {}).get("thread_id", "default"))
        messages = (checkpoint.get("channel_values") or {}).get("messages") or []
        exchange = _latest_exchange(messages)
        if exchange is None:
            return
        user_text, ai_text = exchange
        # LangGraph writes a checkpoint per step, so the same exchange is seen several times.
        fingerprint = hashlib.sha256(f"{user_text}\x00{ai_text}".encode()).hexdigest()
        if self._recorded.get(thread_id) == fingerprint:
            return
        self._recorded[thread_id] = fingerprint
        mm = await self._get_manager()
        await mm.store_turn(self.agent_id, thread_id, user_text, ai_text)


def _langgraph_savers() -> tuple[type[Any], Callable[[], Any]]:
    """LangGraph's ``BaseCheckpointSaver`` and its in-memory saver (renamed across versions)."""
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    try:
        from langgraph.checkpoint.memory import InMemorySaver as DefaultSaver
    except ImportError:  # older LangGraph only has the MemorySaver name
        from langgraph.checkpoint.memory import MemorySaver as DefaultSaver
    return BaseCheckpointSaver, DefaultSaver


_SAVER_CLASS: type[Any] | None = None


def _checkpoint_saver_class() -> type[Any]:
    """Build (once) the ``BaseCheckpointSaver`` subclass that ``ActroneCheckpointer()`` returns.

    Every public method the installed LangGraph declares on ``BaseCheckpointSaver`` is forwarded to
    the wrapped saver, so the adapter keeps working as LangGraph adds or changes checkpoint methods
    (0.1 had no ``new_versions`` argument, later versions added thread deletion and more).
    Forwarding also keeps the concrete saver's own optimised implementations in charge of graph
    state.
    """
    global _SAVER_CLASS
    if _SAVER_CLASS is not None:
        return _SAVER_CLASS
    saver_base, _ = _langgraph_savers()

    def forward(name: str) -> Any:
        if name in _RECORDING_METHODS:

            async def recording(
                self: Any, config: Any, checkpoint: Any, *args: Any, **kwargs: Any
            ) -> Any:
                result = await getattr(self._saver, name)(config, checkpoint, *args, **kwargs)
                await self._record_exchange(config, checkpoint)
                return result

            return recording

        def delegate(self: Any, *args: Any, **kwargs: Any) -> Any:
            return getattr(self._saver, name)(*args, **kwargs)

        return delegate

    namespace: dict[str, Any] = {}
    for name in dir(saver_base):
        if name.startswith("_"):
            continue
        member = getattr(saver_base, name)
        if isinstance(member, property):
            namespace[name] = _forwarded_property(name)
        elif callable(member) and not isinstance(member, type):
            namespace[name] = forward(name)
    if hasattr(saver_base, "with_allowlist"):
        namespace["with_allowlist"] = _with_allowlist
    namespace["__doc__"] = ActroneCheckpointer.__doc__
    namespace["__module__"] = __name__
    _SAVER_CLASS = type("ActroneCheckpointSaver", (ActroneCheckpointer, saver_base), namespace)
    return _SAVER_CLASS


def _forwarded_property(name: str) -> property:
    """A read-only property that reads ``name`` from the wrapped saver (e.g. ``config_specs``)."""

    def getter(self: Any) -> Any:
        return getattr(self._saver, name)

    return property(getter)


def _with_allowlist(self: Any, extra_allowlist: Any) -> Any:
    """LangGraph's serializer allowlist hook, applied to the wrapped saver that serializes.

    LangGraph calls it at compile time so a graph's own state types can be deserialized. The base
    implementation clones the saver with a new ``serde``; forwarding it keeps that serializer on
    the saver that actually reads and writes checkpoints, not on this wrapper.
    """
    inner = self._saver.with_allowlist(extra_allowlist)
    if inner is self._saver:
        return self
    clone = copy.copy(self)
    clone._saver = inner
    clone.serde = inner.serde
    return clone


def _message_role_and_text(message: Any) -> tuple[str, str]:
    """``(role, text)`` for a LangChain message, a ``(role, text)`` tuple or a role/content dict."""
    if isinstance(message, tuple) and len(message) == 2:
        role, content = message
    elif isinstance(message, dict):
        role, content = message.get("role") or message.get("type", ""), message.get("content", "")
    else:
        role, content = getattr(message, "type", ""), getattr(message, "content", "")
    if isinstance(content, list):  # multimodal content blocks: keep the text parts
        content = " ".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    role = {"human": "user", "ai": "assistant"}.get(str(role).lower(), str(role).lower())
    return role, str(content or "").strip()


def _latest_exchange(messages: list[Any]) -> tuple[str, str] | None:
    """The newest ``(user, assistant)`` pair, or ``None`` until the last message is an AI reply."""
    parsed = [_message_role_and_text(m) for m in messages]
    if not parsed or parsed[-1][0] != "assistant" or not parsed[-1][1]:
        return None
    for role, text in reversed(parsed[:-1]):
        if role == "user" and text:
            return text, parsed[-1][1]
    return None
