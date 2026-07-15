"""Framework-agnostic building blocks shared by the actrone-memory adapters.

Every framework adapter needs the same two things: turn a query into a governed
**system-context string** (Tier 1 — the universally-correct, framework-free path), and
persist a completed turn. :class:`BaseActroneMemory` provides both plus lazy
:class:`~actrone_memory.manager.MemoryManager` lifecycle, so each per-framework adapter only
adds that framework's native method names (and, where the framework has a formal memory
interface, its Tier-2 conformance).

Nothing here imports any agent framework — these helpers run against a local in-memory /
hashing MemoryManager with no services and no API key.
"""

from __future__ import annotations

from collections.abc import Iterable

from actrone_memory.config import MemoryConfig
from actrone_memory.manager import MemoryManager
from actrone_memory.models import MemoryEntry, RetrievedContext

DEFAULT_TOKEN_BUDGET = 4096


def format_memories(memories: Iterable[MemoryEntry]) -> str:
    """Render episodic search results into a governed long-term-memory block.

    The Tier-1 rendering for retrieval-shaped adapters (DSPy ``Retrieve``, the Haystack
    retriever) that rank by relevance without a turn/session model. Returns ``""`` when there
    is nothing to inject, so callers can cheaply skip prepending an empty block.
    """
    facts = [f"- {m.content}" for m in memories]
    if not facts:
        return ""
    return "Relevant long-term memory:\n" + "\n".join(facts)


async def governed_context(
    manager: MemoryManager,
    agent_id: str,
    session_id: str,
    query: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """Tier-1 governed system-context string for a conversational adapter.

    Framework-free: retrieves recent turns + episodic memory for ``query`` and renders the
    single system-prompt block every adapter can prepend to its instructions. Returns ``""``
    when nothing is relevant. This is the universally-correct path shared by the legacy
    adapters' :meth:`build_context` and :meth:`BaseActroneMemory.build_context`.
    """
    ctx = await manager.retrieve_context(agent_id, session_id, query, token_budget)
    return format_memory_context(ctx)


def format_memory_context(ctx: RetrievedContext) -> str:
    """Render a retrieved context into a single governed system-prompt string.

    Returns ``""`` when there is nothing relevant, so callers can cheaply skip prepending an
    empty block. Long-term (episodic) memory leads, followed by the recent conversation —
    the shape every framework's ``system`` / ``instructions`` slot expects.
    """
    parts: list[str] = []
    if ctx.episodic_memories:
        facts = "\n".join(f"- {m.content}" for m in ctx.episodic_memories)
        parts.append(f"Relevant long-term memory:\n{facts}")
    if ctx.recent_turns:
        convo = "\n".join(
            f"User: {t.user_message}\nAssistant: {t.assistant_message}" for t in ctx.recent_turns
        )
        parts.append(f"Recent conversation:\n{convo}")
    return "\n\n".join(parts)


class BaseActroneMemory:
    """Shared base for the framework memory adapters (Tier 1: system-context string).

    Subclasses add the framework's native method names (e.g. ``instructions_for`` /
    ``search_memory`` / ``invoking``); this base owns the manager lifecycle and the two
    primitives every adapter needs — :meth:`build_context` and :meth:`remember`.

    Args:
        agent_id:       Agent whose memory namespace is used.
        session_id:     Conversation/session id (L1 recency scope).
        token_budget:   Context token budget for retrieval (default 4096).
        config:         Optional :class:`MemoryConfig`; used to lazily create a manager.
        memory_manager: An existing manager to reuse (e.g. shared across agents). When
            omitted, one is created on first use from ``config`` — local-first, no services.
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.token_budget = token_budget
        self._config = config
        self._mm = memory_manager

    async def manager(self) -> MemoryManager:
        """Return the (lazily created) MemoryManager backing this adapter."""
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(self, query: str, *, token_budget: int | None = None) -> str:
        """Governed system-context string for ``query`` (Tier 1). ``""`` when nothing relevant."""
        mm = await self.manager()
        return await governed_context(
            mm, self.agent_id, self.session_id, query, token_budget or self.token_budget
        )

    async def remember(self, user_message: str, assistant_message: str) -> str:
        """Persist a completed turn; returns the stored turn id."""
        mm = await self.manager()
        return await mm.store_turn(self.agent_id, self.session_id, user_message, assistant_message)

    async def clear(self) -> None:
        """Clear this session's recent-turn buffer."""
        mm = await self.manager()
        await mm.clear_session(self.agent_id, self.session_id)
