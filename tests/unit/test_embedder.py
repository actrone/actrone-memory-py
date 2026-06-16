from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.exceptions import EmbeddingError
from actrone_memory.l2.embedder import CachedEmbedder, OpenAIEmbedder
from tests.conftest import ConstantEmbedder


@pytest.mark.asyncio
async def test_cached_embedder_returns_vector():
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = None

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    result = await embedder.embed("hello world")
    assert result == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_cached_embedder_uses_cache_on_hit():
    cached_value = json.dumps([0.9, 0.8, 0.7, 0.6]).encode()
    cache = AsyncMock()
    cache.get.return_value = cached_value

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    result = await embedder.embed("hello world")
    # Should return cached value, not inner embedder's value
    assert result == [0.9, 0.8, 0.7, 0.6]


@pytest.mark.asyncio
async def test_cached_embedder_batch_partial_cache_hit():
    cache = AsyncMock()
    cache.get.side_effect = [json.dumps([0.9, 0.8, 0.7, 0.6]).encode(), None]
    cache.set.return_value = None

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    results = await embedder.embed_batch(["cached text", "uncached text"])
    assert len(results) == 2
    assert results[0] == [0.9, 0.8, 0.7, 0.6]
    assert results[1] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_cached_embedder_writes_to_cache_on_miss():
    cache = AsyncMock()
    cache.get.return_value = None

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    await embedder.embed("new text")
    cache.set.assert_called_once()


def test_cached_embedder_dimensions_delegates_to_inner():
    cache = AsyncMock()
    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    assert embedder.dimensions == 4


# ------------------------------------------------------------------
# OpenAIEmbedder (mocked openai client)
# ------------------------------------------------------------------

def _make_openai_embedder() -> tuple[OpenAIEmbedder, MagicMock]:
    """Construct OpenAIEmbedder then swap out the client with a mock."""
    embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small", dimensions=1536)
    mock_client = MagicMock()
    embedder._client = mock_client
    return embedder, mock_client


def test_openai_embedder_dimensions():
    embedder, _ = _make_openai_embedder()
    assert embedder.dimensions == 1536


@pytest.mark.asyncio
async def test_openai_embedder_embed_returns_vector():
    embedder, mock_client = _make_openai_embedder()

    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = [mock_item]
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await embedder.embed("hello world")
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_openai_embedder_embed_batch_returns_vectors():
    embedder, mock_client = _make_openai_embedder()

    items = [MagicMock(embedding=[float(i)]) for i in range(3)]
    mock_response = MagicMock()
    mock_response.data = items
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    results = await embedder.embed_batch(["a", "b", "c"])
    assert len(results) == 3


@pytest.mark.asyncio
async def test_openai_embedder_embed_batch_empty_returns_empty():
    embedder, _ = _make_openai_embedder()
    results = await embedder.embed_batch([])
    assert results == []


@pytest.mark.asyncio
async def test_openai_embedder_raises_embedding_error_on_failure():
    embedder, mock_client = _make_openai_embedder()
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=Exception("API error"))

    with pytest.raises(EmbeddingError, match="OpenAI embedding failed"):
        await embedder.embed("fail")
