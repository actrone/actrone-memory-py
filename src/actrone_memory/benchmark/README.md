# actrone-memory: memory quality benchmark & eval harness

**The only memory library that ships its own quality eval.** Memory quality is a
measurable, regression-gated property here, not a vibe. This harness runs in CI on
every change, fully offline (no services, no API key), so a change that degrades
recall fails the build.

## Run it

```bash
python -m actrone_memory.benchmark
```

Output (default local backend, in-memory store + dependency-free hashing embedder):

```text
| metric        | value   |
| ------------- | ------- |
| queries       | 14      |
| recall@5      | 0.929   |
| precision@5   | 0.186   |
| MRR           | 0.929   |
| latency p50   | 0.22 ms |
| latency p95   | 0.61 ms |
```

## What it measures

A bundled, LongMemEval-style dataset (`dataset.py`) seeds durable memories for an
agent, then poses queries whose *relevant* memory ids are known ground truth. The
harness reports standard information-retrieval metrics:

- **recall@k**, of the memories that *should* be recalled, what fraction land in
  the top-k.
- **precision@k**, of the top-k returned, what fraction are relevant. (With one
  relevant memory per query, the ceiling is `1/k`, so ~0.2 at k=5 is near-perfect,
  not a weakness.)
- **MRR**, mean reciprocal rank of the first relevant hit.
- **latency p50/p95**, per-query retrieval time.

## Be honest about what this shows

The **default** scores reflect the **hashing embedder** (keyword overlap), not
semantic understanding. That is the point of the free, zero-service default, and
its limits are real: paraphrases with no shared words score low. Where we **win**
today is latency, cost, local-first/zero-egress, and governance (provenance +
erasure). Where we **lose** today is raw semantic recall depth versus dense-embedding
systems. To measure semantic recall, evaluate with a real embedder:

```python
import asyncio
from actrone_memory import MemoryConfig
from actrone_memory.benchmark import run_eval

cfg = MemoryConfig(embedding_provider="openai", openai_api_key="sk-...",
                   relevance_threshold=0.2)
print(asyncio.run(run_eval(config=cfg)).format_table())
```

## Head-to-head comparison (incl. competitors)

`run_comparison({...})` runs the **same dataset** against multiple systems and
returns a per-system report, so comparisons are apples-to-apples and reproducible.
The bundled run compares Actrone against a dependency-free naive recency baseline:

```text
| system           | recall@5 | precision@5 | MRR   | p95 latency |
| ---------------- | -------- | ----------- | ----- | ----------- |
| actrone          | 0.929    | 0.186       | 0.929 | 0.44 ms     |
| recency-baseline | 0.571    | 0.114       | 0.273 | 0.01 ms     |
```

Any object satisfying the tiny `MemorySystem` protocol is comparable, just two
async methods:

```python
class MemorySystem(Protocol):
    async def inject_memory(self, agent_id: str, content: str) -> str: ...
    async def search_memories(self, agent_id: str, query: str, limit: int) -> list: ...  # items with .id
```

To benchmark **Mem0 / Zep / Letta / Cognee**, write a thin adapter mapping those two
calls onto its API and pass it in:

```python
import asyncio
from actrone_memory import MemoryConfig, MemoryManager
from actrone_memory.benchmark import run_comparison, format_comparison

class Mem0Adapter:  # sketch, wrap the real mem0 client
    def __init__(self, client): self._c = client
    async def inject_memory(self, agent_id, content):
        return self._c.add(content, user_id=agent_id)["id"]
    async def search_memories(self, agent_id, query, limit):
        return [type("H", (), {"id": r["id"]}) for r in self._c.search(query, user_id=agent_id, limit=limit)]

async def main():
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.2))
    reports = await run_comparison({"actrone": mm, "mem0": Mem0Adapter(mem0_client)})
    print(format_comparison(reports))

asyncio.run(main())
```

Competitor adapters are intentionally **not** bundled: they pull heavy deps and API
keys, which would break the offline CI gate. The harness + a runnable baseline ship
so the comparison is real and reproducible out of the box.

## CI gate

`tests/unit/test_benchmark.py` asserts `recall@5 ≥ 0.85` and `MRR ≥ 0.80` against
the default dataset. Raise the floors when a stronger embedder becomes the default.
