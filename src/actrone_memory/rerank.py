"""Optional cross-encoder reranking over the top-K candidates (Axis A4).

A bi-encoder (the embedder) scores query and document independently; a **cross-encoder** scores the
*pair* jointly and is markedly more precise — but O(K) model calls, so it is only worth running over
a small over-fetched candidate set. **Honest constraint:** reranking lifts *precision*, not recall —
it can only reorder what retrieval already fetched, so it must sit *after* an over-fetch. Off by
default (one model download + K inferences per query); enable with ``rerank_enabled=True``.

Uses ``fastembed``'s ONNX ``TextCrossEncoder`` (the ``[onnx]`` extra, no torch), so it keeps the
local-first / zero-egress promise: one model download, offline thereafter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import structlog

from actrone_memory.models import MemoryEntry

log = structlog.get_logger(__name__)

# A small, widely-used cross-encoder (~90 MB ONNX). Override via MemoryConfig.rerank_model.
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks memories by a cross-encoder relevance score of ``(query, content)`` pairs.

    Requires ``pip install actrone-memory[onnx]``. Model files download on first construction and
    are cached (offline thereafter). CPU-bound inference is offloaded to a thread pool.
    """

    def __init__(
        self, model_name: str = DEFAULT_RERANK_MODEL, cache_dir: str | None = None
    ) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ImportError("Install onnx extras: pip install actrone-memory[onnx]") from exc
        self._model_name = model_name
        self._encoder = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self, query: str, entries: Sequence[MemoryEntry], *, top_k: int | None = None
    ) -> list[MemoryEntry]:
        """Return ``entries`` reordered by descending cross-encoder relevance to ``query``.

        Only the first ``top_k`` entries are rescored (retrieval is assumed to have already surfaced
        the strongest candidates first); any remainder is appended in its original order, so the
        result is always a permutation of the input with the same length.
        """
        if not entries or not query.strip():
            return list(entries)

        head = list(entries) if top_k is None else list(entries[:top_k])
        tail = [] if top_k is None else list(entries[top_k:])
        documents = [e.content for e in head]

        loop = asyncio.get_running_loop()

        def _score() -> list[float]:
            return list(self._encoder.rerank(query, documents))

        scores = await loop.run_in_executor(None, _score)
        order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
        return [head[i] for i in order] + tail


def build_reranker(
    *, enabled: bool, model_name: str = DEFAULT_RERANK_MODEL
) -> CrossEncoderReranker | None:
    """Construct the reranker when enabled, degrading to ``None`` if its dependency is missing.

    Returning ``None`` (rather than raising) keeps ``MemoryManager.create()`` robust: an operator
    who enables reranking without installing the ``[onnx]`` extra gets a logged warning and the
    un-reranked (still correct) retrieval order, not a hard startup failure.
    """
    if not enabled:
        return None
    try:
        reranker = CrossEncoderReranker(model_name)
        log.info("memory.rerank.enabled", model=reranker.model_name)
        return reranker
    except Exception as exc:  # ImportError (extra absent) OR model-fetch failure (air-gapped)
        log.warning(
            "memory.rerank.unavailable",
            reason=str(exc),
            note="rerank_enabled=True but the cross-encoder could not load; "
            "install actrone-memory[onnx]. Falling back to un-reranked order.",
        )
        return None
