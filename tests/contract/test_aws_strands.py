"""Contract test for the AWS Strands memory adapter (Tier 1).

Framework-free, runs against a real local MemoryManager (no services, no API key).
"""

from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.aws_strands import ActroneStrandsMemory
from actrone_memory.manager import MemoryManager


@pytest.mark.asyncio
async def test_system_prompt_appends_memory_and_passes_through_when_empty() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneStrandsMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    assert await memory.system_prompt("You are support.", "nothing") == "You are support."

    await mm.inject_memory("support-bot", "Prod deploys are frozen on Fridays.", 0.9)
    merged = await memory.system_prompt("You are support.", "prod deploys frozen fridays")
    assert "frozen" in merged
    assert "You are support." in merged
    # Memory is appended after the base system prompt.
    assert merged.index("You are support.") < merged.index("frozen")

    await memory.remember("q", "a")
    await mm.close()
