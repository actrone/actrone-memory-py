"""Haystack v2 component adapters for actrone-memory.

Provides two Haystack v2 ``@component``-decorated classes:

- ``ActroneRetriever``, retrieves relevant memories from Qdrant L2 as Haystack
  ``Document`` objects. Drop-in for any Haystack RAG pipeline.
- ``ActroneWriter``, stores a user/assistant conversation turn to Redis L1 and
  Qdrant L2. Connect after your LLM component to auto-persist conversations.

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

from typing import Any

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import format_memories
from actrone_memory.manager import MemoryManager


def _require_haystack() -> None:
    try:
        from haystack import component  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install Haystack extras: pip install actrone-memory[haystack]") from exc


class ActroneRetriever:
    """Haystack v2 ``@component`` that retrieves memories from Qdrant L2.

    Accepts a ``query`` string and returns a list of Haystack ``Document`` objects
    ranked by 0.7 × relevance + 0.3 × recency (the actrone-memory default).

    Args:
        agent_id: Unique identifier for the agent (memory namespace).
        top_k: Default number of memories to retrieve.
        config: Optional ``MemoryConfig``; falls back to environment variables.
        memory_manager: Pre-constructed ``MemoryManager`` (avoids re-initialisation).
    """

    def __init__(
        self,
        agent_id: str,
        top_k: int = 10,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        _require_haystack()
        from haystack import component

        self._agent_id = agent_id
        self._top_k = top_k
        self._config = config
        self._mm: MemoryManager | None = memory_manager

        # Haystack registers I/O types via the @component decorator at class level.
        # Because we can't apply it to a class that also needs __init__ args, we
        # call component.output_types() manually as a mixin-style registration.
        component.set_output_type(self, "documents", list)

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(self, query: str, *, top_k: int | None = None) -> str:
        """Governed long-term-memory block for ``query`` (Tier 1, framework-free).

        The universally-correct path: prepend the returned block to any prompt regardless of
        framework. Renders the same relevance-ranked memories as :meth:`run_async` into a single
        system-prompt block; returns ``""`` when nothing is relevant.
        """
        mm = await self._get_manager()
        memories = await mm.search_memories(self._agent_id, query, limit=top_k or self._top_k)
        return format_memories(memories)

    async def run_async(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Async version of ``run`` for use in async Haystack pipelines."""
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

    def run(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Synchronous entrypoint, Haystack's default pipeline executor calls this."""
        import asyncio

        try:
            # Probe for a running loop; the binding is unused, we only branch on
            # whether the call raises.
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(query, top_k))

        # Existing event loop, run in a thread to avoid blocking it.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self.run_async(query, top_k))
            return future.result()


class ActroneWriter:
    """Haystack v2 ``@component`` that persists a conversation turn to memory.

    Connect after your LLM component. Pass ``user_message`` and ``assistant_message``
    from the pipeline context to automatically store turns in Redis L1 + Qdrant L2.

    Args:
        agent_id: Unique identifier for the agent.
        session_id: Conversation/thread identifier.
        config: Optional ``MemoryConfig``; falls back to environment variables.
        memory_manager: Pre-constructed ``MemoryManager`` (avoids re-initialisation).
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str = "default",
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        _require_haystack()
        from haystack import component

        self._agent_id = agent_id
        self._session_id = session_id
        self._config = config
        self._mm: MemoryManager | None = memory_manager

        component.set_output_type(self, "memories_written", int)

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def run_async(self, user_message: str, assistant_message: str) -> dict[str, Any]:
        """Async version of ``run`` for use in async Haystack pipelines."""
        mm = await self._get_manager()
        await mm.store_turn(
            self._agent_id,
            self._session_id,
            user_message,
            assistant_message,
        )
        return {"memories_written": 1}

    def run(self, user_message: str, assistant_message: str) -> dict[str, Any]:
        """Synchronous entrypoint called by the Haystack pipeline executor."""
        import asyncio
        import concurrent.futures

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, self.run_async(user_message, assistant_message)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.run_async(user_message, assistant_message))
