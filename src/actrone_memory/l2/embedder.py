from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

import structlog
from redis.asyncio import Redis
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from actrone_memory.exceptions import EmbeddingError

log = structlog.get_logger(__name__)

_CACHE_PREFIX = "embed:"


_RetryableFn = TypeVar("_RetryableFn", bound=Callable[..., Awaitable[Any]])


def _build_openai_retry() -> Callable[[_RetryableFn], _RetryableFn]:
    """Construct an OpenAI-specific retry decorator at import time.

    Importing ``openai`` lazily lets the rest of the module load when the
    optional dependency is missing (e.g. local-embedder-only deployments).
    The decorator allowlists only transient OpenAI client errors so that
    permanent failures (bad model name, invalid API key, schema mismatch)
    surface immediately instead of burning three retries.
    """
    transient: tuple[type[BaseException], ...]
    try:
        import openai

        transient = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
    except ImportError:
        # Fall back to network-level transients only. The decorator is still
        # required by the abstract Embedder API.
        transient = (ConnectionError, TimeoutError)

    return retry(
        retry=retry_if_exception_type(transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        reraise=True,
    )


_EMBED_RETRY = _build_openai_retry()


class Embedder(ABC):
    """Abstract interface all embedding providers must implement."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts.

        The returned list is always the same length as `texts`.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors this provider produces."""


class CachedEmbedder(Embedder):
    """Wraps any Embedder with a Redis-backed cache (default TTL: 7 days).

    On a cache hit, returns the stored vector in < 1 ms with no API call.
    On a miss, calls the inner embedder and writes the result to the cache.
    """

    def __init__(self, inner: Embedder, cache: Redis, ttl_seconds: int = 604800) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_seconds

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"{_CACHE_PREFIX}{digest}"

    async def embed(self, text: str) -> list[float]:
        import json

        key = self._cache_key(text)
        cached = await self._cache.get(key)
        if cached:
            return json.loads(cached)  # type: ignore[no-any-return]

        vector = await self._inner.embed(text)
        await self._cache.set(key, json.dumps(vector), ex=self._ttl)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed with per-item cache lookup. O(n) cache checks, one API call for misses."""
        import json

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = await self._cache.get(self._cache_key(text))
            if cached:
                results[i] = json.loads(cached)
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            vectors = await self._inner.embed_batch(miss_texts)

            if len(vectors) != len(miss_texts):
                raise EmbeddingError(
                    f"Embedder returned {len(vectors)} vectors for {len(miss_texts)} texts."
                )

            for idx, text, vector in zip(miss_indices, miss_texts, vectors, strict=True):
                results[idx] = vector
                await self._cache.set(self._cache_key(text), json.dumps(vector), ex=self._ttl)

        # Contract: returned list is always the same length as `texts`.
        # Any None slot indicates a corrupt cache entry or inner-embedder bug — fail loudly.
        for i, slot in enumerate(results):
            if slot is None:
                raise EmbeddingError(
                    f"CachedEmbedder produced no vector for index {i} "
                    f"(text length {len(texts[i])})."
                )

        return cast("list[list[float]]", results)


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small (1536-dim) with automatic retry.

    Retries up to 3 times on any exception using exponential backoff with
    jitter, starting at 1 second and capped at 10 seconds.
    """

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimensions: int = 1536
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    @_EMBED_RETRY
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            log.error(
                "openai.embed.failed", error=str(exc), model=self._model, batch_size=len(texts)
            )
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc


class LocalEmbedder(Embedder):
    """sentence-transformers/all-MiniLM-L6-v2 for offline / free usage (384-dim).

    Requires: pip install actrone-memory[local]
    CPU-bound encoding is offloaded to a thread pool to avoid blocking the event loop.
    """

    _MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    _DIMENSIONS = 384

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install local extras: pip install actrone-memory[local]") from exc

        self._model = SentenceTransformer(self._MODEL_NAME)

    @property
    def dimensions(self) -> int:
        return self._DIMENSIONS

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        from functools import partial

        # functools.partial keeps the bound encode() call typed by the
        # SentenceTransformer stubs — a plain lambda would erase the return
        # type to Any and fail mypy --strict downstream.
        loop = asyncio.get_running_loop()
        encode = partial(self._model.encode, texts, convert_to_numpy=True)
        embeddings = await loop.run_in_executor(None, encode)
        return [e.tolist() for e in embeddings]
