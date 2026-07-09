"""Compiled/type-checked docs examples for ``actrone-memory`` (Public-Domain Cutover Runbook Phase 6,
item 5).

CI type-checks this file with ``mypy --strict`` against the CURRENT package source, so an API change
that breaks a documented snippet fails the build. The ``# region`` blocks are extracted verbatim into
the docs by ``scripts/extract_snippets.py`` — the guide never hand-types these, so they cannot drift.

Note the correct construction: ``MemoryManager`` is created via the async ``create()`` factory (the
class docstring forbids direct instantiation), so these snippets are the authoritative usage.
"""

from __future__ import annotations

# region memory-py-quickstart
from actrone_memory import MemoryManager


async def quickstart() -> None:
    memory = await MemoryManager.create()

    await memory.store_turn(
        agent_id="research-agent",
        session_id="session-42",
        user_message="Summarise Q4 earnings for AAPL",
        assistant_message="Apple reported revenue of $119.6B in Q4 2024, up 6% YoY...",
    )

    context = await memory.retrieve_context(
        agent_id="research-agent",
        session_id="session-42",
        query="Apple revenue Q4",
        token_budget=2000,
    )

    # context.recent_turns          – recent session turns (L1)
    # context.episodic_memories     – semantically relevant long-term memories (L2)
    # context.total_tokens_used     – tokens consumed across both tiers
    print(context.total_tokens_used)
# endregion memory-py-quickstart
