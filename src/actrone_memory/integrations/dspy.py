"""DSPy Retrieve module adapter for actrone-memory.

Wraps Qdrant L2 semantic search as a DSPy ``Retrieve``-compatible module.
Plug it into any DSPy program that calls ``dspy.Retrieve`` to get persistent,
relevance-ranked context instead of a stateless retrieval backend.

Install::

    pip install actrone-memory[dspy]

Usage::

    import dspy
    from actrone_memory.integrations.dspy import ActroneRM

    # Register as the global DSPy retriever.
    rm = ActroneRM(agent_id="research-agent", k=10)
    dspy.settings.configure(rm=rm)

    # Or use directly in a DSPy Module.
    class RAGModule(dspy.Module):
        def __init__(self):
            self.retrieve = ActroneRM(agent_id="research-agent", k=5)

        def forward(self, question: str) -> dspy.Prediction:
            context = self.retrieve(question).passages
            ...

    # Store turns after each inference to keep the memory up to date.
    await rm.store_turn(agent_id, session_id, user_msg, assistant_msg)

Requires dspy-ai >= 2.4.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import format_memories
from actrone_memory.manager import MemoryManager


def _require_dspy() -> None:
    try:
        import dspy  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install DSPy extras: pip install actrone-memory[dspy]") from exc


def _run_async(coro: Any) -> Any:  # noqa: ANN401
    """Run an async coroutine from sync context without blocking an existing event loop."""
    try:
        asyncio.get_running_loop()
        # Event loop is running (e.g. Jupyter, FastAPI), offload to a thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


class ActroneRM:
    """DSPy Retrieve-compatible module backed by Qdrant L2 semantic search.

    Compatible with ``dspy.settings.configure(rm=...)`` and direct module use.
    Returns a ``dspy.Prediction`` with a ``passages`` attribute, a list of
    content strings ranked by 0.7 × relevance + 0.3 × recency.

    Args:
        agent_id: Unique identifier for the agent (memory namespace).
        k: Default number of passages to retrieve.
        config: Optional ``MemoryConfig``; falls back to environment variables.
        memory_manager: Pre-constructed ``MemoryManager`` (avoids re-initialisation).
    """

    def __init__(
        self,
        agent_id: str,
        k: int = 5,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        _require_dspy()
        self._agent_id = agent_id
        self.k = k
        self._config = config
        self._mm: MemoryManager | None = memory_manager

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def _search(self, query: str, k: int) -> list[str]:
        mm = await self._get_manager()
        memories = await mm.search_memories(self._agent_id, query, limit=k)
        return [m.content for m in memories]

    async def build_context(self, query: str, *, limit: int | None = None) -> str:
        """Governed long-term-memory block for ``query`` (Tier 1, framework-free).

        The universally-correct path: prepend the returned block to any prompt regardless of
        framework. Renders the same relevance-ranked memories as :meth:`forward` into a single
        system-prompt block; returns ``""`` when nothing is relevant.
        """
        mm = await self._get_manager()
        memories = await mm.search_memories(self._agent_id, query, limit=limit or self.k)
        return format_memories(memories)

    def forward(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
        **_kwargs: Any,
    ) -> Any:  # noqa: ANN401, returns dspy.Prediction
        """Retrieve passages for one or more queries.

        Called by DSPy's ``Retrieve`` mechanism. Runs async search synchronously
        using a thread pool to avoid event loop conflicts in both Jupyter and
        production async contexts.

        Args:
            query_or_queries: A single query string or a list of query strings.
            k: Number of passages per query. Defaults to ``self.k``.

        Returns:
            ``dspy.Prediction(passages=[...])`` where passages is a flat list
            of content strings from Qdrant L2.
        """
        import dspy

        num = k if k is not None else self.k

        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = list(query_or_queries)

        async def _fetch_all() -> list[list[str]]:
            return list(await asyncio.gather(*[self._search(q, num) for q in queries]))

        per_query: list[list[str]] = _run_async(_fetch_all())

        if len(queries) == 1:
            passages = per_query[0]
        else:
            # Interleave results: first passage from each query, then second, etc.
            max_len = max((len(p) for p in per_query), default=0)
            passages = [
                per_query[qi][pi]
                for pi in range(max_len)
                for qi in range(len(queries))
                if pi < len(per_query[qi])
            ]

        return dspy.Prediction(passages=passages)

    def __call__(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        """Allow direct calls: ``rm("my query")``."""
        return self.forward(query_or_queries, k=k, **kwargs)

    async def store_turn(
        self,
        agent_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Persist a conversation turn after DSPy inference.

        Call this after each ``program(input)`` call to keep the memory store
        up to date with the agent's conversation history.

        Args:
            agent_id: Must match the ``agent_id`` this retriever was initialised with.
            session_id: Conversation thread identifier.
            user_message: The user's input to the DSPy program.
            assistant_message: The program's output / final answer.
        """
        mm = await self._get_manager()
        await mm.store_turn(agent_id, session_id, user_message, assistant_message)

    async def search_memories(self, query: str, limit: int | None = None) -> list[str]:
        """Raw async semantic search, returns a list of content strings."""
        return await self._search(query, limit if limit is not None else self.k)
