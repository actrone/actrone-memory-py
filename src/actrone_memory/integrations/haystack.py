"""Haystack v2 component adapters for actrone-memory.

Provides two Haystack v2 components:

- ``ActroneRetriever``, retrieves relevant memories as Haystack ``Document`` objects. Drop-in for
  any Haystack RAG pipeline.
- ``ActroneWriter``, stores a user/assistant conversation turn. Connect after your LLM component
  to persist conversations.

Install::

    pip install actrone-memory[haystack]

Usage::

    from actrone_memory.integrations.haystack import ActroneRetriever, ActroneWriter
    from haystack import Pipeline

    pipeline = Pipeline()
    pipeline.add_component("retriever", ActroneRetriever(agent_id="research-agent"))
    pipeline.add_component("writer", ActroneWriter(agent_id="research-agent"))

Requires haystack-ai >= 2.0.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable, Coroutine
from typing import Any, Optional, TypeVar

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import format_memories
from actrone_memory.manager import MemoryManager

_T = TypeVar("_T")


def _require_haystack() -> None:
    try:
        from haystack import component  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install Haystack extras: pip install actrone-memory[haystack]") from exc


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` from Haystack's sync executor, on a worker thread if a loop is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


class ActroneRetriever:
    """Haystack v2 component that retrieves memories as ``Document`` objects.

    Input ``query`` (and optional ``top_k``); output ``documents``, ranked by actrone-memory's
    relevance and recency blend. Constructing it returns a real Haystack component (a class built
    with Haystack's ``@component`` on first use, so this module imports without Haystack), ready for
    ``Pipeline.add_component``. Both ``run`` and ``run_async`` are provided.

    Args:
        agent_id: Unique identifier for the agent (memory namespace).
        top_k: Default number of memories to retrieve.
        config: Optional ``MemoryConfig``; falls back to environment variables.
        memory_manager: Pre-constructed ``MemoryManager`` (avoids re-initialisation).
    """

    _initialised: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is ActroneRetriever:
            # Haystack's component metaclass must run the construction, so build through it.
            return _component_class("retriever")(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        agent_id: str,
        top_k: int = 10,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        _require_haystack()
        if self._initialised:  # already built through Haystack's metaclass by __new__
            return
        self._initialised = True
        self._agent_id = agent_id
        self._top_k = top_k
        self._config = config
        self._mm: MemoryManager | None = memory_manager

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(self, query: str, *, top_k: int | None = None) -> str:
        """Governed long-term-memory block for ``query`` (Tier 1, framework-free).

        The universally-correct path: prepend the returned block to any prompt regardless of
        framework. Renders the same relevance-ranked memories as the component's ``run`` into a
        single system-prompt block; returns ``""`` when nothing is relevant.
        """
        mm = await self._get_manager()
        memories = await mm.search_memories(self._agent_id, query, limit=top_k or self._top_k)
        return format_memories(memories)

    async def _retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        from haystack.dataclasses import Document

        k = top_k if top_k is not None else self._top_k
        mm = await self._get_manager()
        memories = await mm.search_memories(self._agent_id, query, limit=k)
        documents = [
            Document(
                content=m.content,
                meta={
                    "memory_id": m.id,
                    "agent_id": m.agent_id,
                    "session_id": m.session_id,
                    "content_type": m.content_type,
                    "importance_score": m.importance_score,
                    "timestamp": m.timestamp.isoformat(),
                    "topic_tags": m.topic_tags,
                },
                score=m.importance_score,
            )
            for m in memories
        ]
        return {"documents": documents}


class ActroneWriter:
    """Haystack v2 component that persists a conversation turn to memory.

    Inputs ``user_message`` and ``assistant_message``; output ``memories_written``. Connect it after
    your LLM component. Constructing it returns a real Haystack component (see
    :class:`ActroneRetriever`), with both ``run`` and ``run_async``.

    Args:
        agent_id: Unique identifier for the agent.
        session_id: Conversation/thread identifier.
        config: Optional ``MemoryConfig``; falls back to environment variables.
        memory_manager: Pre-constructed ``MemoryManager`` (avoids re-initialisation).
    """

    _initialised: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is ActroneWriter:
            return _component_class("writer")(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        agent_id: str,
        session_id: str = "default",
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        _require_haystack()
        if self._initialised:
            return
        self._initialised = True
        self._agent_id = agent_id
        self._session_id = session_id
        self._config = config
        self._mm: MemoryManager | None = memory_manager

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def _write(self, user_message: str, assistant_message: str) -> dict[str, Any]:
        mm = await self._get_manager()
        await mm.store_turn(self._agent_id, self._session_id, user_message, assistant_message)
        return {"memories_written": 1}


_COMPONENT_CLASSES: dict[str, type[Any]] = {}


def _retriever_methods() -> dict[str, Callable[..., Any]]:
    def run(self: Any, query: str, top_k: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = _run_sync(self._retrieve(query, top_k))
        return result

    async def run_async(self: Any, query: str, top_k: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = await self._retrieve(query, top_k)
        return result

    return {"run": run, "run_async": run_async}


def _writer_methods() -> dict[str, Callable[..., Any]]:
    def run(self: Any, user_message: str, assistant_message: str) -> dict[str, Any]:
        result: dict[str, Any] = _run_sync(self._write(user_message, assistant_message))
        return result

    async def run_async(self: Any, user_message: str, assistant_message: str) -> dict[str, Any]:
        result: dict[str, Any] = await self._write(user_message, assistant_message)
        return result

    return {"run": run, "run_async": run_async}


def _component_class(kind: str) -> type[Any]:
    """Build (once) the Haystack ``@component`` class the two adapters return when constructed.

    Haystack reads a component's input sockets from the ``run`` signature and its output sockets
    from ``@component.output_types``, and only a class passed through ``@component`` can be added to
    a pipeline. This module uses postponed annotations, which Haystack does not resolve, so the
    socket types are set as concrete annotations here. The class is built on first use so the module
    still imports without Haystack.
    """
    if kind in _COMPONENT_CLASSES:
        return _COMPONENT_CLASSES[kind]
    _require_haystack()
    from haystack import component
    from haystack.dataclasses import Document

    base: type[Any]
    if kind == "retriever":
        base = ActroneRetriever
        methods = _retriever_methods()
        inputs: dict[str, Any] = {"query": str, "top_k": Optional[int]}  # noqa: UP045 - Haystack 2.0 does not resolve X | None
        outputs: dict[str, Any] = {"documents": list[Document]}
    else:
        base = ActroneWriter
        methods = _writer_methods()
        inputs = {"user_message": str, "assistant_message": str}
        outputs = {"memories_written": int}

    namespace: dict[str, Any] = {"__module__": __name__, "__doc__": base.__doc__}
    for name, method in methods.items():
        method.__annotations__ = {**inputs, "return": dict[str, Any]}
        namespace[name] = component.output_types(**outputs)(method)
    built: type[Any] = component(type(base.__name__, (base,), namespace))
    _COMPONENT_CLASSES[kind] = built
    return built
