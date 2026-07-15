"""Contract test for the smolagents memory adapter (Tier 1).

Framework-free — runs against a real local MemoryManager (no services, no API key).
"""

from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.smolagents import ActroneSmolagentsMemory
from actrone_memory.manager import MemoryManager


@pytest.mark.asyncio
async def test_task_context_prepends_memory_block_and_empty_when_none() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneSmolagentsMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    assert await memory.task_context("nothing here yet") == ""

    await mm.inject_memory("support-bot", "The build server is named atlas.", 0.9)
    block = await memory.task_context("build server name")
    assert "atlas" in block
    # Non-empty context carries a trailing separator so it composes ahead of the task.
    assert block.endswith("---\n\n")

    await memory.remember("q", "a")
    await mm.close()
