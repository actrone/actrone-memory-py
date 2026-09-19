"""Contract test for the OpenAI Agents SDK memory adapter (Tier 1).

Framework-free, the adapter imports nothing from ``openai-agents``, so this runs against a
real local (hashing + in-memory) MemoryManager with no services and no API key.
"""

from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.openai_agents import ActroneOpenAIAgentsMemory
from actrone_memory.manager import MemoryManager


@pytest.mark.asyncio
async def test_instructions_for_prepends_memory_and_passes_through_when_empty() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneOpenAIAgentsMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    # No memory yet ⇒ base instructions returned unchanged.
    assert await memory.instructions_for("You are support.", "nothing") == "You are support."

    await mm.inject_memory("support-bot", "The customer is on the Enterprise plan.", 0.9)
    merged = await memory.instructions_for("You are support.", "enterprise plan tier")
    assert "Enterprise" in merged
    assert "You are support." in merged
    # Memory is prepended before the base instructions.
    assert merged.index("Enterprise") < merged.index("You are support.")

    turn_id = await memory.remember("q", "a")
    assert turn_id
    await mm.close()
