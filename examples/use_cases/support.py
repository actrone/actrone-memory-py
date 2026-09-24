"""Use case: a customer support agent (actrone.com/use-cases/customer-support).

CI type-checks this file with ``mypy --strict``, and ``tests/unit/test_use_cases.py`` runs it and
asserts what the page claims: customers stay apart, personal data stays out of the prompt, and one
call erases a customer. The ``# region`` block is what the page shows.
"""

# region memory-py-use-case-support
from collections.abc import Awaitable, Callable

from actrone_memory import MemoryManager

CallModel = Callable[[str, list[str]], Awaitable[str]]


def scope_for(customer_id: str) -> str:
    """Long-term memory is scoped by agent_id, so one scope per customer keeps them apart."""
    return f"customer:{customer_id}"


async def import_account(memory: MemoryManager, customer_id: str) -> None:
    """Seed what your CRM already knows, tagged with its source and how sensitive it is."""
    scope = scope_for(customer_id)
    await memory.inject_memory(
        agent_id=scope,
        content="The customer is on the Business plan and renews in March.",
        importance=0.9,
        session_id="crm",
        topic_tags=["plan"],
        source="import:crm",
        sensitivity="none",
    )
    await memory.inject_memory(
        agent_id=scope,
        content="The customer billing email is dana@example.com.",
        importance=0.6,
        session_id="crm",
        topic_tags=["contact"],
        source="import:crm",
        sensitivity="pii",
    )


async def answer_ticket(
    memory: MemoryManager, customer_id: str, ticket_id: str, question: str, call_model: CallModel
) -> str:
    """Answer a ticket with what you know about this customer, minus personal data."""
    scope = scope_for(customer_id)
    context = await memory.retrieve_context(
        agent_id=scope, session_id=ticket_id, query=question, token_budget=2000
    )
    facts = [m.content for m in context.episodic_memories if m.sensitivity in ("none", "low")]
    reply = await call_model(question, facts)
    await memory.store_turn(
        agent_id=scope, session_id=ticket_id, user_message=question, assistant_message=reply
    )
    return reply


async def forget_customer(memory: MemoryManager, customer_id: str, ticket_id: str) -> None:
    """A deletion request: erase the customer's memories and this ticket's turns."""
    await memory.erase_agent_memories(scope_for(customer_id), session_id=ticket_id)


# endregion memory-py-use-case-support
