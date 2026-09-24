"""Use case: a coding assistant (actrone.com/use-cases/coding-assistants).

CI type-checks this file with ``mypy --strict``, and ``tests/unit/test_use_cases.py`` runs it and
asserts what the page claims: conventions stay with their repository and the recalled context never
exceeds the budget. The ``# region`` block is what the page shows.
"""

# region memory-py-use-case-coding
from dataclasses import dataclass

from actrone_memory import MemoryManager


def scope_for(repo: str) -> str:
    """One scope per repository, so one codebase's conventions never leak into another's."""
    return f"repo:{repo}"


async def learn_convention(memory: MemoryManager, repo: str, convention: str) -> str:
    """Record a convention once, when the developer or a code review states it."""
    return await memory.inject_memory(
        agent_id=scope_for(repo),
        content=convention,
        importance=0.9,
        session_id="conventions",
        topic_tags=["convention"],
        source="user",
    )


@dataclass
class TaskContext:
    conventions: list[str]
    tokens_used: int


async def conventions_for(
    memory: MemoryManager, repo: str, session_id: str, task: str
) -> TaskContext:
    """Before each request, recall this task's conventions within a token budget."""
    context = await memory.retrieve_context(
        agent_id=scope_for(repo), session_id=session_id, query=task, token_budget=800
    )
    return TaskContext(
        conventions=[m.content for m in context.episodic_memories],
        tokens_used=context.total_tokens_used,
    )


async def record_exchange(
    memory: MemoryManager, repo: str, session_id: str, request: str, answer: str
) -> None:
    """Keep the exchange, so the next request in this session sees it."""
    await memory.store_turn(
        agent_id=scope_for(repo),
        session_id=session_id,
        user_message=request,
        assistant_message=answer,
    )


# endregion memory-py-use-case-coding
