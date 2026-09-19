"""Contract test for the Google ADK memory adapter (Tier 1 + Tier-2 import guard).

Tier 1 (``build_context`` / ``remember``) runs against a real local MemoryManager. The Tier-2
``as_memory_service`` path lazily imports ``google-adk``; when it is absent (base venv) the guard
must raise a clear ``ImportError``, asserted here. The ``_adk_content_text`` flattener is pure
and unit-tested. A full native ``BaseMemoryService`` test runs when ``google-adk`` is installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.google_adk import ActroneGoogleADKMemory, _adk_content_text
from actrone_memory.manager import MemoryManager


class _Part:
    def __init__(self, text: str) -> None:
        self.text = text


class _Content:
    def __init__(self, *parts: str) -> None:
        self.parts = [_Part(p) for p in parts]


def test_adk_content_text_flattens_parts() -> None:
    assert _adk_content_text(_Content("hello", "world")) == "hello world"
    assert _adk_content_text(None) == ""
    assert _adk_content_text(_Content()) == ""


@pytest.mark.asyncio
async def test_tier1_context_and_remember() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneGoogleADKMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    await mm.inject_memory("support-bot", "Nightly index rebuild runs at 2am.", 0.9)
    assert "rebuild" in await memory.build_context("index rebuild schedule")
    await memory.remember("q", "a")
    await mm.close()


def test_as_memory_service_requires_google_adk() -> None:
    try:
        import google.adk.memory  # noqa: F401
    except ImportError:
        memory: Any = ActroneGoogleADKMemory(agent_id="a", session_id="s")
        with pytest.raises(ImportError, match="google-adk"):
            memory.as_memory_service()
    else:  # pragma: no cover - only when google-adk is installed
        pytest.skip("google-adk installed; native BaseMemoryService path covered elsewhere")
