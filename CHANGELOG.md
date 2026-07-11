# Changelog

All notable changes to `actrone-memory` follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- **Local-first, zero-service default backend.** `MemoryManager.create()` now runs
  fully in-process with **no Redis, no Qdrant, and no API key** — parity with the
  TypeScript `@actrone/memory` on-ramp. "Memory that never phones home."
  - New `InMemoryStore` (`actrone_memory.in_memory`) implements **both** the L1
    (hot session) and L2 (cold semantic) tiers with the same blended
    relevance+recency ranking as the Qdrant backend.
  - New dependency-free `HashingEmbedder` (`actrone_memory.l2.embedder`) — a
    deterministic hashing vectorizer (`embedding_provider="hashing"`), the new
    default. No model download, no external call.
  - New `L1Store` / `L2Store` `Protocol`s (`actrone_memory.protocols`) — the
    manager now depends on the store seam, not concrete backends.
  - New config: `backend: "memory" | "redis_qdrant"` and `hashing_dimensions`.

- **Provenance-typed facts v1** — every stored memory now carries **`source`**
  (attribution: `user`/`assistant`/`tool`/`summary`/`injected`/`extracted`/
  `reflection`/`imported`/`unknown`, or a namespaced string like `"import:crm"`)
  and **`sensitivity`** (`none`/`low`/`pii`/`sensitive`). `inject_memory()` accepts
  both; summaries are tagged `source="summary"`. Defaults are backwards-compatible
  (`unknown`/`none`). New `MemorySource`/`Sensitivity` types exported.
- **Local right-to-erasure** — `MemoryManager.erase_agent_memories(agent_id,
  session_id=None)` hard-deletes an agent's long-term memories (and optionally a
  session's turns). The governance seed that graduates to hosted *provable*
  erasure. `L2Store` gains `delete_agent_memories`.

- **Optional LLM fact extraction** (turns → durable facts), to the shared
  extraction spec v1 (`docs/memory-spec/extraction.v1.md`). New
  `actrone_memory.extraction` (`FactExtractor` seam, `OpenAIFactExtractor`,
  `ExtractedFact`, `parse_facts`, `EXTRACTION_SPEC_VERSION`). Opt-in via
  `MemoryConfig.extract_facts=True` (OpenAI provider); stores each fact as
  `content_type="fact"`, `source="extracted"` with a classified `sensitivity`.
  Exposed as `MemoryManager.extract_memories()` and auto-run on the summarise
  cadence (cost-bounded). New `"fact"` content type.

### Changed

- **Default `backend` is now `"memory"`** (was implicitly Redis + Qdrant) and the
  **default `embedding_provider` is now `"hashing"`** (was `"openai"`). Existing
  deployments must set `ACTRONE_BACKEND=redis_qdrant` and
  `ACTRONE_EMBEDDING_PROVIDER=openai` (+ `ACTRONE_OPENAI_API_KEY`) to keep the
  previous behaviour. No code changes are required — the switches are read at
  startup.
- Qdrant collection dimensionality is now sized from the embedder in use
  (`embedder.dimensions`) rather than a fixed `embedding_dimensions`, so the
  hashing/local embedders provision correctly.

---

## [0.2.0] — 2026-05-20

### Added

- **AutoGen 0.4 adapter** (`integrations/autogen.py`) — implements `autogen_core.memory.Memory` protocol.
  - `add()` stores text content as an injected memory entry.
  - `query()` performs Qdrant L2 semantic search and returns `MemoryQueryResult`.
  - `update_context()` injects episodic memories into the AutoGen model context as system messages.
  - Install: `pip install actrone-memory[autogen]`

- **LlamaIndex adapter** (`integrations/llamaindex.py`) — implements the `BaseMemory` interface.
  - `aget()` / `get()` returns recent turns + episodic memories as `ChatMessage` list.
  - `aput()` / `put()` buffers user messages and flushes to Redis L1 on assistant response.
  - `areset()` / `reset()` clears session memory.
  - Install: `pip install actrone-memory[llamaindex]`

- **Haystack v2 adapters** (`integrations/haystack.py`) — two `@component`-compatible classes.
  - `ActroneRetriever` — retrieves memories as Haystack `Document` objects with metadata.
  - `ActroneWriter` — stores conversation turns after LLM inference.
  - Install: `pip install actrone-memory[haystack]`

- **DSPy adapter** (`integrations/dspy.py`) — `ActroneRM` is a DSPy `Retrieve`-compatible module.
  - `forward()` performs semantic search and returns `dspy.Prediction(passages=[...])`.
  - Supports single query and multi-query (results interleaved by rank position).
  - Works from both sync and async contexts via thread-pool isolation.
  - `store_turn()` persists turns after DSPy program inference.
  - Install: `pip install actrone-memory[dspy]`

- **New optional extras**: `autogen`, `llamaindex`, `haystack`, `dspy` (also included in `all`).
- **Contract test suite** (`tests/contract/`) — each adapter has a full contract test file.

### Changed

- Version bumped `0.1.0 → 0.2.0`.
- `pyproject.toml` keywords expanded to include all supported frameworks.
- `integrations/__init__.py` updated with install instructions for all extras.

---

## [0.1.0] — 2026-05-18

### Added

- Two-tier persistent memory: Redis L1 (hot, <1 ms) + Qdrant L2 (semantic, ~10 ms).
- `MemoryManager` public API: `store_turn`, `retrieve_context`, `inject_memory`, `delete_memory`, `clear_session`, `search_memories`.
- 4-phase retrieval pipeline: parallel L1+L2 fetch → budget allocation → relevance filter → pruning.
- Auto-summarisation: background compression after N turns (default 20) via GPT-4o-mini.
- LangChain adapter (`integrations/langchain.py`) — `BaseChatMemory` drop-in.
- LangGraph checkpointer (`integrations/langgraph.py`) — `BaseCheckpointSaver` implementation.
- CrewAI backend (`integrations/crewai.py`) — `MemoryBackend` implementation.
- OpenAI embedder (`text-embedding-3-small`) and local embedder (`all-MiniLM-L6-v2`).
- Embedding cache: Redis key `embed:{sha256(text)}` with 7-day TTL.
- Full test suite: unit (≥80%), integration (testcontainers), contract.
- CI: ruff, mypy --strict, pip-audit, Docker build, PyPI publish on tag.
