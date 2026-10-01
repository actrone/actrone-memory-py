"""Contract test for the AWS Strands memory adapter (Tier 1).

Framework-free, runs against a real local MemoryManager (no services, no API key).
"""

from __future__ import annotations

from typing import Any

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


@pytest.mark.asyncio
async def test_system_prompt_lands_on_a_real_strands_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The documented wiring: Agent(model=model, system_prompt=system_prompt)."""
    pytest.importorskip("strands", reason="strands-agents not installed")
    # Strands builds a Bedrock client on construction; give it isolated placeholder credentials so
    # the host's AWS configuration (for example an SSO login provider) plays no part. Nothing is
    # sent.
    for name, value in {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_CONFIG_FILE": str(tmp_path / "config"),
        "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "credentials"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    from strands import Agent

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    await mm.inject_memory("support-bot", "Production deploys are frozen on Fridays.", 0.9)
    memory = ActroneStrandsMemory("support-bot", "s1", memory_manager=mm)
    system_prompt = await memory.system_prompt(
        "You are support.", "are production deploys frozen on Fridays"
    )

    agent = Agent(system_prompt=system_prompt)
    assert agent.system_prompt == system_prompt
    assert "Production deploys are frozen on Fridays." in str(agent.system_prompt)
    await mm.close()
