from __future__ import annotations

import asyncio

from actrone_memory.benchmark.compare import format_comparison, run_comparison
from actrone_memory.benchmark.competitors import RecencyBaseline
from actrone_memory.benchmark.harness import run_eval
from actrone_memory.config import MemoryConfig
from actrone_memory.manager import MemoryManager


async def _main() -> None:
    report = await run_eval()
    print("actrone-memory — memory quality benchmark (local backend, hashing embedder)\n")
    print(report.format_table())

    # Comparative run: Actrone vs the dependency-free naive baseline, same dataset.
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))
    try:
        reports = await run_comparison({"actrone": mm, "recency-baseline": RecencyBaseline()})
    finally:
        await mm.close()
    print("\nComparative (same dataset):\n")
    print(format_comparison(reports))
    print(
        "\nNote: scores reflect the dependency-free hashing embedder (keyword overlap); the "
        "baseline is naive recency. Add a Mem0/Zep/Letta adapter (competitors.py docstring) for a "
        "head-to-head. See benchmark/README.md."
    )


if __name__ == "__main__":
    asyncio.run(_main())
