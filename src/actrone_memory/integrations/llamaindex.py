"""LlamaIndex BaseMemory adapter for actrone-memory.

Wraps ``MemoryManager`` as a LlamaIndex ``BaseMemory`` subclass so any
LlamaIndex agent or chat engine can use actrone-memory as its persistent
conversation + episodic memory store.

Install::

    pip install actrone-memory[llamaindex]

Usage::

    from actrone_memory.integrations.llamaindex import ActroneLlamaMemory
    from llama_index.core.chat_engine import SimpleChatEngine

    memory = ActroneLlamaMemory(agent_id="research-agent", session_id="session-1")
    engine = SimpleChatEngine.from_defaults(memory=memory)

Requires llama-index-core >= 0.10.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import governed_context
from actrone_memory.manager import MemoryManager

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import ChatMessage


def _run_sync(coro: Any) -> Any:  # noqa: ANN401
    """Run an async coroutine from a synchronous context.

    When no event loop is running (the common case in scripts and tests) this
    calls ``asyncio.run()``.  When a loop *is* running (Jupyter, FastAPI,
    async test runners) it submits the coroutine to a fresh thread so the
    caller does not block the running loop, matching the same pattern used in
    ``haystack.py`` and ``dspy.py``.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def _llamaindex_tokenizer() -> Callable[[str], list[Any]]:
    """LlamaIndex's default tokenizer, or whitespace splitting if it cannot be loaded offline."""
    try:
        from llama_index.core.utils import get_tokenizer

        tokenizer: Callable[[str], list[Any]] = get_tokenizer()
        return tokenizer
    except (ImportError, OSError):  # no tokenizer module, or its data cannot be fetched offline
        return str.split


def _require_llamaindex() -> None:
    try:
        import llama_index.core.memory  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install LlamaIndex extras: pip install actrone-memory[llamaindex]"
        ) from exc


class ActroneLlamaMemory:
    """LlamaIndex ``BaseMemory`` implementation backed by actrone-memory.

    Short-term session turns are stored in Redis L1 (<1 ms reads).
    Episodic memories are retrieved from Qdrant L2 (~10 ms semantic search).

    Args:
        agent_id: Unique identifier for the agent.
        session_id: Conversation thread identifier.
        token_budget: Maximum tokens returned from ``aget()``.
        config: Optional ``MemoryConfig``; falls back to environment variables.
        memory_manager: Pre-constructed ``MemoryManager`` (avoids re-initialisation).
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str = "default",
        token_budget: int = 4096,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        _require_llamaindex()
        self._agent_id = agent_id
        self._session_id = session_id
        self._token_budget = token_budget
        self._config = config
        self._mm: MemoryManager | None = memory_manager

        # Buffer for the current turn being assembled (mirrors LlamaIndex's pattern).
        self._pending_user_msg: str = ""

        # LlamaIndex chat engines count prompt tokens with ``memory.tokenizer_fn``; 0.10 reads it
        # unconditionally, so without it every chat crashed. Use LlamaIndex's own tokenizer so the
        # counts match the framework's.
        self.tokenizer_fn: Callable[[str], list[Any]] = _llamaindex_tokenizer()

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(self, query: str, *, token_budget: int | None = None) -> str:
        """Governed system-context string for ``query`` (Tier 1, framework-free).

        The universally-correct path: prepend the returned block to any prompt regardless of
        framework. Returns ``""`` when nothing is relevant. Complements the native LlamaIndex
        ``BaseMemory`` interface below.
        """
        mm = await self._get_manager()
        return await governed_context(
            mm, self._agent_id, self._session_id, query, token_budget or self._token_budget
        )

    # ── LlamaIndex BaseMemory interface ──────────────────────────────────────

    async def aget(
        self,
        input: str | None = None,
        initial_token_count: int = 0,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """The conversation for the next LLM call: memories, recent turns, then the new message.

        LlamaIndex chat engines ``aput`` the user's message first and then call ``aget`` with no
        ``input`` to build the prompt, relying on the memory to return the whole conversation. So
        the pending (not yet answered) user message is both the recall query and the last message
        returned; without it the model would receive neither the memories nor the question.
        """
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        mm = await self._get_manager()
        ctx = await mm.retrieve_context(
            self._agent_id,
            self._session_id,
            input or self._pending_user_msg,
            self._token_budget - initial_token_count,
        )

        messages: list[ChatMessage] = []

        # Episodic memories injected as a system message.
        if ctx.episodic_memories:
            memory_block = "\n".join(f"- {m.content}" for m in ctx.episodic_memories)
            messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"[Relevant memories]\n{memory_block}",
                )
            )

        # Recent session turns as user/assistant pairs.
        for turn in ctx.recent_turns:
            messages.append(ChatMessage(role=MessageRole.USER, content=turn.user_message))
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=turn.assistant_message))

        # The message being answered now: put, but not yet a stored turn.
        if self._pending_user_msg:
            messages.append(ChatMessage(role=MessageRole.USER, content=self._pending_user_msg))

        return messages

    # Sync shims required by LlamaIndex's synchronous chat engines.
    # Uses a thread pool when an event loop is already running (e.g. Jupyter, FastAPI)
    # to avoid "asyncio.run() cannot be called when another loop is running".
    def get(
        self,
        input: str | None = None,
        initial_token_count: int = 0,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        # _run_sync executes the coroutine and returns its (typed) result; mypy
        # only sees Any through the event-loop bridge.
        return _run_sync(self.aget(input, initial_token_count, **kwargs))  # type: ignore[no-any-return]

    async def aput(self, message: ChatMessage) -> None:
        """Buffer a message. Flushes to memory when a complete turn is assembled."""
        from llama_index.core.base.llms.types import MessageRole

        if message.role == MessageRole.USER:
            self._pending_user_msg = str(message.content)
        elif message.role == MessageRole.ASSISTANT and self._pending_user_msg:
            mm = await self._get_manager()
            await mm.store_turn(
                self._agent_id,
                self._session_id,
                self._pending_user_msg,
                str(message.content),
            )
            self._pending_user_msg = ""

    def put(self, message: ChatMessage) -> None:
        _run_sync(self.aput(message))

    async def aset(self, messages: list[ChatMessage]) -> None:
        """Replace the session's conversation with ``messages``.

        Chat engines call this when a caller passes ``chat_history``. The session is cleared, each
        user message is paired with the assistant reply that follows it, and a trailing user
        message stays pending for the next reply.
        """
        await self.areset()
        for message in messages:
            await self.aput(message)

    def set(self, messages: list[ChatMessage]) -> None:
        _run_sync(self.aset(messages))

    async def areset(self) -> None:
        """Clear all memory for this agent/session."""
        mm = await self._get_manager()
        await mm.clear_session(self._agent_id, self._session_id)
        self._pending_user_msg = ""

    def reset(self) -> None:
        _run_sync(self.areset())

    async def aget_all(self) -> list[ChatMessage]:
        """Return all known messages. Delegates to ``aget`` with no query."""
        return await self.aget()

    def get_all(self) -> list[ChatMessage]:
        result: list[ChatMessage] = _run_sync(self.aget_all())
        return result
