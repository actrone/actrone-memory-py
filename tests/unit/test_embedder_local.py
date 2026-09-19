"""Unit tests for the in-process ONNX embedder and the graceful local chain.

These stub ``fastembed`` so they run in the base venv (no heavy onnxruntime download) and cover:
the ONNX wrapper's embed/dimensions logic, and ``build_local_embedder`` degrading ONNX →
sentence-transformers → hashing so ``embedding_provider="local"`` (the new default) always yields a
working, offline, no-API-key embedder.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from actrone_memory.l2.embedder import (
    FastEmbedEmbedder,
    HashingEmbedder,
    build_local_embedder,
)


class _FakeArray:
    """Minimal stand-in for a numpy vector (avoids a numpy test dependency)."""

    def __init__(self, data: list[float]) -> None:
        self._data = data

    def tolist(self) -> list[float]:
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _install_fake_fastembed(
    monkeypatch: pytest.MonkeyPatch, *, catalogue_dim: int | None = 384
) -> None:
    """Inject a fake ``fastembed`` module exposing a deterministic ``TextEmbedding``."""

    class _FakeTextEmbedding:
        def __init__(
            self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: Any = None
        ) -> None:
            self.model_name = model_name

        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            if catalogue_dim is None:
                return []
            return [{"model": "BAAI/bge-small-en-v1.5", "dim": catalogue_dim}]

        def embed(self, texts: list[str]) -> Any:
            for t in texts:
                yield _FakeArray([float(len(t)), 1.0, 2.0])

    module = types.ModuleType("fastembed")
    module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)


@pytest.mark.asyncio
async def test_fastembed_embedder_uses_catalogue_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed(monkeypatch, catalogue_dim=384)
    emb = FastEmbedEmbedder()
    assert emb.model_name == "BAAI/bge-small-en-v1.5"
    assert emb.dimensions == 384  # resolved from the catalogue, not the probe vector length

    vecs = await emb.embed_batch(["hello", "world!!"])
    assert len(vecs) == 2
    assert all(isinstance(v, list) for v in vecs)
    single = await emb.embed("x")
    assert isinstance(single, list)


@pytest.mark.asyncio
async def test_fastembed_embedder_probes_when_model_absent_from_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unknown model → not in catalogue → dimension resolved by probing one embedding (len 3 here).
    _install_fake_fastembed(monkeypatch, catalogue_dim=None)
    emb = FastEmbedEmbedder(model_name="some/custom-model")
    assert emb.dimensions == 3


async def test_fastembed_embedder_empty_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed(monkeypatch)
    emb = FastEmbedEmbedder()
    assert await emb.embed_batch([]) == []


def test_build_local_embedder_prefers_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed(monkeypatch, catalogue_dim=384)
    emb = build_local_embedder()
    assert isinstance(emb, FastEmbedEmbedder)
    assert emb.dimensions == 384


def test_build_local_embedder_falls_back_to_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neither fastembed nor sentence-transformers importable → hashing, honouring the passed width.
    monkeypatch.setitem(sys.modules, "fastembed", None)  # forces ImportError on `import fastembed`
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    emb = build_local_embedder(hashing_dimensions=128)
    assert isinstance(emb, HashingEmbedder)
    assert emb.dimensions == 128


def test_build_local_embedder_falls_through_on_model_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fastembed importable but construction raises (e.g. air-gapped first run) → falls through.
    class _Boom:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("model download failed (air-gapped)")

    module = types.ModuleType("fastembed")
    module.TextEmbedding = _Boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    emb = build_local_embedder(hashing_dimensions=64)
    assert isinstance(emb, HashingEmbedder)
    assert emb.dimensions == 64
