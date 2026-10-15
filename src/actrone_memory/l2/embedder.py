from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from actrone_memory.exceptions import EmbeddingError
from actrone_memory.logging import bind_logger
from actrone_memory.metrics import embedding_cache_hits_total, embedding_cache_misses_total

if TYPE_CHECKING:
    # `redis` is an OPTIONAL extra: only the CachedEmbedder (redis_qdrant backend) uses it,
    # and then only as a constructor type hint. Keeping it out of runtime imports means the
    # local-first install (`pip install actrone-memory`) pulls no redis.
    from redis.asyncio import Redis

log = bind_logger(__name__)

_CACHE_PREFIX = "embed:"
_WORD_RE = re.compile(r"[a-z0-9]+")


P = ParamSpec("P")
R = TypeVar("R", bound=Awaitable[Any])


def _build_openai_retry() -> Callable[[Callable[P, R]], Callable[P, R]]:
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
