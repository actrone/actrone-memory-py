"""Contract test for the Pydantic AI memory adapter (Tier 1).

Framework-free, runs against a real local MemoryManager (no services, no API key).
"""

from __future__ import annotations

from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.pydantic_ai import ActronePydanticAIMemory
from actrone_memory.manager import MemoryManager


@pytest.mark.asyncio
async def test_system_prompt_returns_governed_memory_and_empty_when_none() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActronePydanticAIMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    assert await memory.system_prompt("nothing here yet") == ""

    await mm.inject_memory("support-bot", "Deployments require two approvals.", 0.9)
    prompt = await memory.system_prompt("deployment approvals")
    assert "approvals" in prompt

    await memory.remember("q", "a")
    assert (await mm.get_session_metadata("support-bot", "s1")).turn_count == 1  # type: ignore[union-attr]
    await mm.close()


@pytest.mark.asyncio
async def test_memory_reaches_the_model_through_a_real_dynamic_system_prompt() -> None:
    """The documented wiring end to end, with Pydantic AI's offline FunctionModel as the LLM."""
    pytest.importorskip("pydantic_ai", reason="pydantic-ai not installed")
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    seen: list[str] = []

    def model(messages: list[Any], _info: Any) -> ModelResponse:
        seen.append(
            " ".join(str(getattr(part, "content", "")) for m in messages for part in m.parts)
        )
        return ModelResponse(parts=[TextPart("Two approvals.")])

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    await mm.inject_memory("support-bot", "Deploys need two approvals.", 0.9)
    memory = ActronePydanticAIMemory("support-bot", "s1", memory_manager=mm)
    agent = Agent(FunctionModel(model), deps_type=str)

    @agent.system_prompt
    async def with_memory(ctx: RunContext[str]) -> str:
        return await memory.system_prompt(ctx.deps)

    user_input = "How many approvals does a deploy need?"
    result = await agent.run(user_input, deps=user_input)
    output = getattr(result, "output", None) or getattr(result, "data", None)
    assert output == "Two approvals."
    assert "Deploys need two approvals." in seen[0] and user_input in seen[0]

    await memory.remember(user_input, str(output))
    assert (await mm.get_recent_turns("support-bot", "s1"))[0].assistant_message == "Two approvals."
    await mm.close()
