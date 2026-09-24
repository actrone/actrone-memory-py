"""Runs the use-case examples (``examples/use_cases/``) and asserts every behaviour their pages
on actrone.com claim. The pages render the examples' ``# region`` blocks verbatim, so a claim
that stops being true fails here before it can be published. Mirrors ``test/use-cases.test.ts``
in the TypeScript library.

The manager is pinned to the keyword embedder, so the run is deterministic and needs no model
download; every query below shares real words with the fact it should recall.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import pytest_asyncio

from actrone_memory import MemoryConfig, MemoryManager

_USE_CASES = Path(__file__).resolve().parents[2] / "examples" / "use_cases"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"use_case_{name}", _USE_CASES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = _load("support")
coding = _load("coding")
assistant = _load("assistant")


@pytest_asyncio.fixture
async def memory() -> AsyncIterator[MemoryManager]:
    manager = await MemoryManager.create(
        MemoryConfig(embedding_provider="hashing", auto_summarise=False)
    )
    try:
        yield manager
    finally:
        await manager.close()


class RecordingModel:
    """A model stand-in that records the facts it was given."""

    def __init__(self, reply: str = "Thanks, that is sorted.") -> None:
        self.reply = reply
        self.calls: list[list[str]] = []

    async def __call__(self, question: str, facts: list[str]) -> str:
        self.calls.append(facts)
        return self.reply


# ── Customer support ──────────────────────────────────────────────────────────


async def test_support_answers_from_own_facts_without_personal_data(memory: MemoryManager) -> None:
    await support.import_account(memory, "c-1")
    model = RecordingModel("You are on the Business plan.")

    await support.answer_ticket(memory, "c-1", "t-1", "Which plan is the customer on?", model)

    facts = model.calls[0]
    assert "The customer is on the Business plan and renews in March." in facts
    assert "@" not in " ".join(facts)
    assert len(await memory.get_recent_turns("customer:c-1", "t-1")) == 1


async def test_support_keeps_customers_apart(memory: MemoryManager) -> None:
    await support.import_account(memory, "c-1")
    model = RecordingModel()

    await support.answer_ticket(memory, "c-2", "t-9", "Which plan is the customer on?", model)

    assert model.calls[0] == []


async def test_support_erases_a_customer_in_one_call(memory: MemoryManager) -> None:
    await support.import_account(memory, "c-1")
    await support.answer_ticket(
        memory, "c-1", "t-1", "Which plan is the customer on?", RecordingModel()
    )

    await support.forget_customer(memory, "c-1", "t-1")

    after = RecordingModel()
    await support.answer_ticket(memory, "c-1", "t-2", "Which plan is the customer on?", after)
    assert after.calls[0] == []
    assert await memory.get_recent_turns("customer:c-1", "t-1") == []


# ── Coding assistant ──────────────────────────────────────────────────────────

ZOD_RULE = "Route handlers live in src/routes and validate the request body with zod."


async def _seed_conventions(memory: MemoryManager) -> None:
    await coding.learn_convention(memory, "web", ZOD_RULE)
    await coding.learn_convention(memory, "web", "Run the tests with pnpm vitest, not npm test.")
    await coding.learn_convention(memory, "web", "Dates use date-fns; moment is not allowed.")
    await coding.learn_convention(
        memory, "billing-service", "Route handlers in this service validate with pydantic."
    )


async def test_coding_recalls_the_relevant_convention_from_this_repo_only(
    memory: MemoryManager,
) -> None:
    await _seed_conventions(memory)

    result = await coding.conventions_for(
        memory, "web", "s-1", "Add a route handler that validates the request body"
    )

    assert ZOD_RULE in result.conventions
    assert "pydantic" not in " ".join(result.conventions)


async def test_coding_never_exceeds_the_budget(memory: MemoryManager) -> None:
    await _seed_conventions(memory)
    for i in range(20):
        await coding.record_exchange(
            memory, "web", "s-1", f"Request {i}: tidy the route handlers", "Done, see the diff."
        )

    result = await coding.conventions_for(memory, "web", "s-1", "Add a route handler")

    assert result.tokens_used <= 800


# ── Personal assistant ────────────────────────────────────────────────────────

PREFERENCE = "Prefers answers in British English without a preamble."
ALLERGY = "Is allergic to peanuts."


async def _seed_user(memory: MemoryManager) -> None:
    await assistant.remember_about_user(memory, "u-1", PREFERENCE)
    await assistant.remember_about_user(memory, "u-1", ALLERGY, "sensitive")


async def test_assistant_remembers_after_the_conversation_ends(memory: MemoryManager) -> None:
    await _seed_user(memory)
    await assistant.record_turn(
        memory, "u-1", "conv-1", "Book me a table for Friday", "Booked for 7pm."
    )
    await assistant.end_conversation(memory, "u-1", "conv-1")

    recall = await assistant.recall_for(
        memory, "u-1", "conv-2", "Write a short note in British English"
    )

    assert recall.recent_turns == 0
    assert (PREFERENCE, "low") in recall.facts


async def test_assistant_returns_each_fact_with_its_sensitivity(memory: MemoryManager) -> None:
    await _seed_user(memory)

    recall = await assistant.recall_for(
        memory, "u-1", "conv-2", "Suggest a dinner place, I am allergic to peanuts"
    )

    assert (ALLERGY, "sensitive") in recall.facts


async def test_assistant_keeps_users_apart(memory: MemoryManager) -> None:
    await _seed_user(memory)

    recall = await assistant.recall_for(
        memory, "u-2", "conv-1", "Suggest a dinner place, I am allergic to peanuts"
    )

    assert recall.facts == []
