from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from actrone_memory.models import Sensitivity

"""Bundled, reproducible memory-quality dataset (LongMemEval-style, offline).

Each case seeds a set of durable memories for one agent, then poses queries whose
*relevant* memory ids are known ground truth. The harness (``harness.py``) measures
recall@k / precision@k / MRR over these — so memory quality is a measurable,
regression-gated property that ships **in the library**. Deliberately small,
keyword-overlap-friendly, and dependency-free so it runs in CI with the default
local backend (in-memory store + hashing embedder) — no services, no API key.
"""


@dataclass(frozen=True)
class MemoryItem:
    """A durable fact to seed, with a dataset-local id used as ground truth."""

    id: str
    content: str
    sensitivity: Sensitivity = "none"
    source: str = "imported"


@dataclass(frozen=True)
class Query:
    """A retrieval probe and the set of memory ids that *should* be recalled."""

    text: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class EvalCase:
    agent_id: str
    memories: list[MemoryItem]
    queries: list[Query]
    name: str = ""
    distractors: list[MemoryItem] = field(default_factory=list)


# ── Case 1: personal assistant long-term memory ──────────────────────────────
_ASSISTANT = EvalCase(
    name="personal_assistant",
    agent_id="assistant",
    memories=[
        MemoryItem("a1", "The user's name is Alex Rivera.", sensitivity="pii"),
        MemoryItem("a2", "The user lives in Cape Town, South Africa.", sensitivity="pii"),
        MemoryItem("a3", "The user is allergic to penicillin.", sensitivity="sensitive"),
        MemoryItem("a4", "The user prefers concise answers with no preamble.", sensitivity="low"),
        MemoryItem("a5", "The user is building a crypto trading bot.", sensitivity="low"),
        MemoryItem("a6", "The user's preferred programming language is Rust.", sensitivity="low"),
        MemoryItem("a7", "The user meets the finance team every Monday.", sensitivity="low"),
        MemoryItem("a8", "The user's daughter Maya is six years old.", sensitivity="pii"),
    ],
    queries=[
        Query("what is the user's name", frozenset({"a1"})),
        Query("where does the user live city country", frozenset({"a2"})),
        Query("does the user have any drug allergies penicillin", frozenset({"a3"})),
        Query("how does the user prefer answers concise", frozenset({"a4"})),
        Query("what project is the user building crypto trading bot", frozenset({"a5"})),
        Query("preferred programming language Rust", frozenset({"a6"})),
        Query("the user's daughter Maya age", frozenset({"a8"})),
    ],
)

# ── Case 2: customer-support agent knowledge ─────────────────────────────────
_SUPPORT = EvalCase(
    name="support_agent",
    agent_id="support-bot",
    memories=[
        MemoryItem("s1", "Refunds are processed within five business days."),
        MemoryItem("s2", "The premium plan costs ninety nine dollars per month."),
        MemoryItem("s3", "Password resets are sent to the account email address."),
        MemoryItem("s4", "The API rate limit is one thousand requests per minute."),
        MemoryItem("s5", "Enterprise customers get a dedicated support engineer."),
        MemoryItem("s6", "Data is stored in the EU region for European customers."),
        MemoryItem("s7", "The free trial lasts fourteen days with no credit card required."),
        MemoryItem("s8", "Invoices can be downloaded from the billing settings page."),
    ],
    queries=[
        Query("how long do refunds take business days", frozenset({"s1"})),
        Query("premium plan price per month cost", frozenset({"s2"})),
        Query("how do password resets work account email", frozenset({"s3"})),
        Query("what is the API rate limit requests per minute", frozenset({"s4"})),
        Query("where is data stored EU region European", frozenset({"s6"})),
        Query("free trial length days credit card", frozenset({"s7"})),
        Query("where to download invoices billing settings", frozenset({"s8"})),
    ],
)

#: The default bundled benchmark. Import and extend for domain-specific evals.
DEFAULT_DATASET: list[EvalCase] = [_ASSISTANT, _SUPPORT]


def load_cases(path: str) -> list[EvalCase]:
    """Load evaluation cases from a JSON file so external benchmarks (LOCOMO, LongMemEval) can drive
    the same recall@k / precision@k / MRR harness (Gate E). The file is a JSON list of case
    objects::

        [{"name": "...", "agent_id": "...",
          "memories":  [{"id": "m1", "content": "...", "sensitivity": "none"}],
          "queries":   [{"text": "...", "relevant_ids": ["m1"]}],
          "distractors": [{"id": "d1", "content": "..."}]}]

    A tiny converter turns each upstream benchmark (whose evidence/answer ids become
    ``relevant_ids``)
    into this shape, keeping the metric + the harness identical across datasets — the reproducible
    results table Gate E requires. Raises ``ValueError`` on a malformed file.
    """
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("dataset file must be a JSON list of case objects")
    cases: list[EvalCase] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict) or "agent_id" not in c:
            raise ValueError(f"case {i}: must be an object with an 'agent_id'")
        cases.append(_case_from_dict(c, i))
    return cases


def _mem_from_dict(d: dict[str, Any], where: str) -> MemoryItem:
    if "id" not in d or "content" not in d:
        raise ValueError(f"{where}: memory needs 'id' and 'content'")
    return MemoryItem(
        id=str(d["id"]),
        content=str(d["content"]),
        sensitivity=d.get("sensitivity", "none"),
        source=str(d.get("source", "imported")),
    )


def _case_from_dict(c: dict[str, Any], i: int) -> EvalCase:
    memories = [_mem_from_dict(m, f"case {i}") for m in c.get("memories", [])]
    distractors = [_mem_from_dict(m, f"case {i} distractor") for m in c.get("distractors", [])]
    queries = []
    for q in c.get("queries", []):
        if "text" not in q or "relevant_ids" not in q:
            raise ValueError(f"case {i}: query needs 'text' and 'relevant_ids'")
        queries.append(
            Query(text=str(q["text"]), relevant_ids=frozenset(map(str, q["relevant_ids"])))
        )
    return EvalCase(
        agent_id=str(c["agent_id"]),
        memories=memories,
        queries=queries,
        name=str(c.get("name", "")),
        distractors=distractors,
    )
