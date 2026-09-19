"""Prometheus metrics for actrone-memory.

Defines a private CollectorRegistry so embedding applications can pick a
single exposition strategy (mounting on their own HTTP server, push-gateway,
multiprocess shared dir, etc.) without colliding with the global registry.

To expose alongside your FastAPI app::

    from prometheus_client import generate_latest
    from actrone_memory.metrics import REGISTRY

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type="text/plain")
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# ── L1 / L2 store operations ────────────────────────────────────────────────
store_op_duration_seconds = Histogram(
    "actrone_memory_store_op_duration_seconds",
    "Latency of L1/L2 store operations in seconds.",
    labelnames=("tier", "operation", "outcome"),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

store_op_total = Counter(
    "actrone_memory_store_op_total",
    "Count of L1/L2 store operations.",
    labelnames=("tier", "operation", "outcome"),
    registry=REGISTRY,
)

# ── Embedding cache ─────────────────────────────────────────────────────────
embedding_cache_hits_total = Counter(
    "actrone_memory_embedding_cache_hits_total",
    "Embedding cache hit count.",
    registry=REGISTRY,
)

embedding_cache_misses_total = Counter(
    "actrone_memory_embedding_cache_misses_total",
    "Embedding cache miss count.",
    registry=REGISTRY,
)

# ── Background tasks ────────────────────────────────────────────────────────
background_task_in_flight = Gauge(
    "actrone_memory_background_task_in_flight",
    "Number of background tasks (e.g. session summarisation) currently running.",
    labelnames=("task",),
    registry=REGISTRY,
)

background_task_failed_total = Counter(
    "actrone_memory_background_task_failed_total",
    "Background task failures by task name and exception class.",
    labelnames=("task", "exception"),
    registry=REGISTRY,
)

# ── Retrieval pipeline ──────────────────────────────────────────────────────
retrieve_context_duration_seconds = Histogram(
    "actrone_memory_retrieve_context_duration_seconds",
    "End-to-end latency of MemoryManager.retrieve_context().",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# The store Protocols declare `async def`, so the decorator must hand back a
# coroutine function. A plain Awaitable return would not satisfy them.
_StoreMethod = Callable[_P, Coroutine[Any, Any, _R]]


def track_store_op(
    tier: str, operation: str
) -> Callable[[_StoreMethod[_P, _R]], _StoreMethod[_P, _R]]:
    """Decorate a store method so its latency and outcome are recorded.

    Without this, ``store_op_duration_seconds`` and ``store_op_total`` are
    declared but never observed, so a ``/metrics`` scrape reports zero for every
    store operation. The wrapped call is labelled ``outcome="error"`` whenever it
    raises, and the exception always propagates unchanged.

    Apply it below the retry decorator so each attempt is measured separately,
    which is what an error rate should count: work actually sent to the store.
    """

    def decorate(fn: _StoreMethod[_P, _R]) -> _StoreMethod[_P, _R]:
        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            start = time.perf_counter()
            outcome = "success"
            try:
                return await fn(*args, **kwargs)
            except BaseException:
                outcome = "error"
                raise
            finally:
                elapsed = time.perf_counter() - start
                labels = {"tier": tier, "operation": operation, "outcome": outcome}
                store_op_duration_seconds.labels(**labels).observe(elapsed)
                store_op_total.labels(**labels).inc()

        return wrapper

    return decorate


__all__ = (
    "REGISTRY",
    "background_task_failed_total",
    "background_task_in_flight",
    "embedding_cache_hits_total",
    "embedding_cache_misses_total",
    "retrieve_context_duration_seconds",
    "store_op_duration_seconds",
    "store_op_total",
    "track_store_op",
)
