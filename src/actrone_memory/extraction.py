from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from actrone_memory.logging import bind_logger
from actrone_memory.models import Sensitivity

if TYPE_CHECKING:
    from openai import AsyncOpenAI

log = bind_logger(__name__)

# ── Shared extraction spec (language-neutral; keep in lockstep with the TS lib) ──
# The canonical reference lives at docs/memory-spec/extraction.v1.md. The prompt, the
# input framing and the output JSON shape are the contract both OSS libs and the hosted
# engine conform to, so "improve once" = update the spec + eval and both follow.
#
# 1.1: small models (a 3B model on Ollama, measured) returned "{}" for nearly every real
# exchange under 1.0, because they read the assistant's reply as part of what to mine.
# 1.1 frames the conversation, says whose facts to record, sends a JSON schema, and
# gives two worked examples: one with facts and one where the right answer is none.
EXTRACTION_SPEC_VERSION = "1.1"

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
    "nothing durable to remember.\n"
    "The conversation is between a user and an AI assistant. Extract facts about the user "
    "and their world from what the user says; use the assistant's replies only as context, "
    "never as a source of facts.\n"
    "Write one fact per piece of information: a name and a job are two facts. Classify each "
    "fact by the most sensitive detail it contains: a person's name, email address, phone "
    "number or postal address is 'pii'; health, emotions or mental state, money and "
    "credentials are 'sensitive'; a preference is 'low'; everything else is 'none'.\n"
    "Only record facts the user states about themselves, their work or their world. Never "
    "record facts about the conversation itself (such as what the user asked), about the "
    "assistant, or general knowledge from the assistant's answers. If the user only makes "
    "small talk, thanks the assistant, "
    'or asks a general question, return {"facts": []}.\n'
    "Example, not part of the conversation you are given:\n"
    "User: I'm Sam, a nurse, and I've been struggling with insomnia. Email me at "
    "sam@example.org. I like short replies.\n"
    "Assistant: Thanks Sam, noted.\n"
    'Output: {"facts": ['
    '{"content": "The user\'s name is Sam.", "sensitivity": "pii", '
    '"topic_tags": ["identity"], "importance": 0.8}, '
    '{"content": "The user works as a nurse.", "sensitivity": "none", '
    '"topic_tags": ["role"], "importance": 0.6}, '
    '{"content": "The user has been struggling with insomnia.", "sensitivity": "sensitive", '
    '"topic_tags": ["health"], "importance": 0.7}, '
    '{"content": "The user\'s email address is sam@example.org.", "sensitivity": "pii", '
    '"topic_tags": ["contact"], "importance": 0.8}, '
    '{"content": "The user prefers short replies.", "sensitivity": "low", '
    '"topic_tags": ["preference"], "importance": 0.5}]}\n'
    "Second example, also not part of the conversation:\n"
    "User: Thanks, that helps!\n"
    "Assistant: Glad to help. The Moon is about 384,000 km away, by the way.\n"
    'Output: {"facts": []}'
)

#: JSON schema for the extraction output, sent as a structured-output ``response_format``
#: so constrained decoding keeps even a small model on the contract. Strict-mode shaped
#: (every property required, no extra properties) so OpenAI accepts it with ``strict``.
EXTRACTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "sensitivity": {"type": "string", "enum": ["none", "low", "pii", "sensitive"]},
                    "topic_tags": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number"},
                },
                "required": ["content", "sensitivity", "topic_tags", "importance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def format_extraction_input(conversation: str) -> str:
    """Frame a conversation (``User: ...`` / ``Assistant: ...`` lines) as the extraction
    request's user message, per the shared spec. Custom ``FactExtractor`` implementations
    that call a model should send this rather than the raw conversation."""
    return f"Conversation:\n\n{conversation}\n\nExtract the durable facts from this conversation."


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


_SCHEMA_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_facts",
        "strict": True,
        "schema": EXTRACTION_RESPONSE_SCHEMA,
    },
}
_JSON_FORMAT: dict[str, Any] = {"type": "json_object"}


class OpenAIFactExtractor:
    """LLM fact extractor for any OpenAI-compatible chat API, conforming to the shared spec.

    Works with OpenAI itself and with local or self-hosted servers that speak the same API
    (Ollama, vLLM, LM Studio): pass ``base_url``, or a ready ``AsyncOpenAI`` as ``client``.
    It asks for a JSON-schema structured output and, if a server rejects that, falls back to
    plain JSON mode for the rest of its life.

    Cost: one completion per extraction call. Extraction is opt-in
    (``MemoryConfig.extract_facts``) precisely because it costs tokens.

    Args:
        api_key: API key; when omitted, the ``openai`` SDK reads ``OPENAI_API_KEY``.
        model: The chat model to extract with.
        base_url: An OpenAI-compatible endpoint, for example ``http://localhost:11434/v1``.
        client: An existing ``AsyncOpenAI`` (or compatible) client; ``api_key`` and
            ``base_url`` are ignored when it is given.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        *,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self._model = model
        self._use_schema = True

    async def extract(self, text: str) -> list[ExtractedFact]:
        """Extract facts from ``text`` (``User:`` / ``Assistant:`` lines). Never raises."""
        try:
            if not self._use_schema:
                return await self._complete(text, _JSON_FORMAT)
            try:
                return await self._complete(text, _SCHEMA_FORMAT)
            except Exception as exc:
                # Only a server that rejects the request (400/422) lacks json_schema support;
                # anything else (timeouts, 5xx) is not a reason to give up the schema.
                if getattr(exc, "status_code", None) not in (400, 422):
                    raise
                log.info("extraction.schema.unsupported", error=str(exc), model=self._model)
            facts = await self._complete(text, _JSON_FORMAT)
            self._use_schema = False
            return facts
        except Exception as exc:
            # Best-effort enrichment: log and return nothing, never break the caller.
            log.warning("extraction.llm.failed", error=str(exc), model=self._model)
            return []

    async def _complete(self, text: str, response_format: dict[str, Any]) -> list[ExtractedFact]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": format_extraction_input(text)},
            ],
            "response_format": response_format,
            "max_tokens": 800,
            "temperature": 0.1,
        }
        response = await self._client.chat.completions.create(**request)
        return parse_facts(response.choices[0].message.content or "{}")
