from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from actrone_memory.logging import bind_logger
from actrone_memory.models import Sensitivity

log = bind_logger(__name__)

# ── Shared extraction spec v1 (language-neutral; keep in lockstep with the TS lib) ──
# The canonical reference lives at docs/memory-spec/extraction.v1.md. The prompt and
# the output JSON shape are the contract both OSS libs and the hosted engine conform
# to, so "improve once" = update the spec + eval and both implementations follow.
EXTRACTION_SPEC_VERSION = "1.0"

# Each fact the model returns:
#   { "content": str, "sensitivity": "none"|"low"|"pii"|"sensitive",
#     "topic_tags": [str], "importance": float(0..1) }
EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable, atomic facts from a conversation so an AI agent can "
    "remember them across sessions. Return ONLY facts worth remembering long term: "
    "stable user attributes, preferences, decisions, commitments, and key entities. "
    "Ignore small talk, transient state, and anything already obvious.\n"
    "For each fact, classify its sensitivity: 'none' (non-personal), 'low' (mild "
    "preference), 'pii' (personally identifiable, names, emails, phone, address, "
    "account numbers), or 'sensitive' (health, financial, credentials, special "
    "category). Assign an importance from 0.0 to 1.0.\n"
    'Respond with strict JSON of the form {"facts": [{"content": "...", '
    '"sensitivity": "none", "topic_tags": ["..."], "importance": 0.7}]}. '
    "Write each fact as a self-contained sentence. Return an empty list if there is "
    "nothing durable to remember."
)


class ExtractedFact(BaseModel):
    """One atomic fact extracted from conversation, conforming to the shared spec."""

    content: str
    sensitivity: Sensitivity = "none"
    topic_tags: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)


@runtime_checkable
class FactExtractor(Protocol):
    """Seam for turning conversation text into durable facts (turns → facts).

    Implementations are LLM-backed; the library treats extraction as best-effort
    enrichment (a failure returns no facts and never breaks the write path).
    """

    async def extract(self, text: str) -> list[ExtractedFact]: ...


# Hard caps so a misbehaving model can't blow up the store or the token budget.
_MAX_FACTS = 20
_MAX_FACT_CHARS = 2_000


def parse_facts(raw: str) -> list[ExtractedFact]:
    """Parse a model's JSON response into validated facts, defensively.

    Tolerates a bare list or a ``{"facts": [...]}`` envelope, skips malformed
    entries, clamps oversize content, and bounds the count. Never raises, a
    completely unparseable response yields ``[]``.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log.warning("extraction.parse.invalid_json")
        return []

    items = data.get("facts") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    facts: list[ExtractedFact] = []
    for item in items[:_MAX_FACTS]:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            fact = ExtractedFact(
                content=content.strip()[:_MAX_FACT_CHARS],
                sensitivity=item.get("sensitivity", "none"),
                topic_tags=[str(t) for t in item.get("topic_tags", []) if isinstance(t, str)][:20],
                importance=float(item.get("importance", 0.6)),
            )
        except (ValueError, TypeError):
            # Bad sensitivity enum / importance out of range, skip this one fact.
            continue
        facts.append(fact)
    return facts


class OpenAIFactExtractor:
    """LLM fact extractor using OpenAI JSON mode, conforming to the shared spec.

    Cost: one cheap completion (e.g. gpt-4o-mini) per extraction call. Extraction
    is opt-in (``MemoryConfig.extract_facts``) precisely because it costs tokens.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def extract(self, text: str) -> list[ExtractedFact]:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                max_tokens=800,
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            return parse_facts(content)
        except Exception as exc:
            # Best-effort enrichment: log and return nothing, never break the caller.
            log.warning("extraction.llm.failed", error=str(exc), model=self._model)
            return []
