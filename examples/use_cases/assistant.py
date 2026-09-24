"""Use case: a personal assistant (actrone.com/use-cases/personal-assistants).

CI type-checks this file with ``mypy --strict``, and ``tests/unit/test_use_cases.py`` runs it and
asserts what the page claims: preferences outlive the conversation that produced them, and each one
arrives tagged with its sensitivity. The ``# region`` block is what the page shows.
"""

# region memory-py-use-case-assistant
from dataclasses import dataclass

from actrone_memory import MemoryManager, Sensitivity


def scope_for(user_id: str) -> str:
    """One scope per user: everything below is recalled for this user only."""
    return f"user:{user_id}"


async def remember_about_user(
    memory: MemoryManager, user_id: str, fact: str, sensitivity: Sensitivity = "low"
) -> str:
    """Keep something the user told you, with the sensitivity you judge it to have."""
    return await memory.inject_memory(
        agent_id=scope_for(user_id),
        content=fact,
        importance=0.9,
        session_id="profile",
        source="user",
        sensitivity=sensitivity,
    )


async def record_turn(
    memory: MemoryManager, user_id: str, conversation_id: str, user_message: str, reply: str
) -> None:
    """Each conversation is a session: its turns feed the next reply in the same conversation."""
    await memory.store_turn(
        agent_id=scope_for(user_id),
        session_id=conversation_id,
        user_message=user_message,
        assistant_message=reply,
    )


@dataclass
class Recall:
    facts: list[tuple[str, Sensitivity]]
    recent_turns: int


async def recall_for(
    memory: MemoryManager, user_id: str, conversation_id: str, message: str
) -> Recall:
    """Recall for a new message; each fact keeps its tag, so you decide what the model sees."""
    context = await memory.retrieve_context(
        agent_id=scope_for(user_id), session_id=conversation_id, query=message, token_budget=1500
    )
    return Recall(
        facts=[(m.content, m.sensitivity) for m in context.episodic_memories],
        recent_turns=len(context.recent_turns),
    )


async def end_conversation(memory: MemoryManager, user_id: str, conversation_id: str) -> None:
    """Closing a conversation drops its turns. What you remembered about the user stays."""
    await memory.clear_session(scope_for(user_id), conversation_id)


# endregion memory-py-use-case-assistant
