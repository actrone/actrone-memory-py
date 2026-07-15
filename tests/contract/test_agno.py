"""Contract test for the Agno memory adapter (Tier 1).

Framework-free — runs against a real local MemoryManager (no services, no API key).
"""

from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.agno import ActroneAgnoMemory
from actrone_memory.manager import MemoryManager


@pytest.mark.asyncio
async def test_additional_context_returns_governed_memory_and_empty_when_none() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneAgnoMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    assert await memory.additional_context("nothing here yet") == ""

    await mm.inject_memory("support-bot", "Escalations page the on-call engineer.", 0.9)
    context = await memory.additional_context("how do escalations work")
    assert "Escalations" in context

    await memory.remember("q", "a")
    assert (await mm.get_session_metadata("support-bot", "s1")).turn_count == 1  # type: ignore[union-attr]
    await mm.close()
