"""Contract test for the Claude Agent SDK memory adapter (Tier 1).

Framework-free, runs against a real local MemoryManager (no services, no API key).
"""

from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.claude_agent_sdk import ActroneClaudeAgentMemory
from actrone_memory.manager import MemoryManager


@pytest.mark.asyncio
async def test_append_to_system_prompt_appends_memory_and_passes_through_when_empty() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneClaudeAgentMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    assert await memory.append_to_system_prompt("You are support.", "nothing") == "You are support."

    await mm.inject_memory("support-bot", "Refunds over 500 need manager approval.", 0.9)
    merged = await memory.append_to_system_prompt("You are support.", "refunds manager approval")
    assert "Refunds" in merged
    assert "You are support." in merged
    # Memory is APPENDED after the base system prompt (Claude-SDK convention).
    assert merged.index("You are support.") < merged.index("Refunds")

    await memory.remember("q", "a")
    await mm.close()
