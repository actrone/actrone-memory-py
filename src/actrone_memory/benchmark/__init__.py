"""Shipped memory-quality benchmark + eval harness.

Run it with ``python -m actrone_memory.benchmark``. Import :func:`run_eval` to
evaluate any manager (yours or a competitor adapter) against the bundled dataset.
"""

from actrone_memory.benchmark.compare import (
    MemorySystem,
    format_comparison,
    run_comparison,
)
from actrone_memory.benchmark.competitors import RecencyBaseline
from actrone_memory.benchmark.dataset import (
    DEFAULT_DATASET,
    EvalCase,
    MemoryItem,
    Query,
)
from actrone_memory.benchmark.harness import (
    EvalReport,
    QueryOutcome,
    run_eval,
)

__all__ = [
    "DEFAULT_DATASET",
    "EvalCase",
    "MemoryItem",
    "Query",
    "EvalReport",
    "QueryOutcome",
    "run_eval",
    "run_comparison",
    "format_comparison",
    "MemorySystem",
    "RecencyBaseline",
]
