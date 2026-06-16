# Changelog

All notable changes to `actrone-memory` follow [Semantic Versioning](https://semver.org/).

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
