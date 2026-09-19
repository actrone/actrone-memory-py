"""Pydantic AI, mypy-checked example source for the docs."""
# region pydantic_ai
from actrone_memory import MemoryManager
from actrone_memory.integrations.pydantic_ai import ActronePydanticAIMemory


async def pydantic_ai_example() -> None:
    mm = await MemoryManager.create()
    memory = ActronePydanticAIMemory("support-bot", "s1", memory_manager=mm)
    system_prompt = await memory.system_prompt("deployment approvals")
    # register `system_prompt` via @agent.system_prompt (dynamic), then agent.run(user_input)
    _ = system_prompt
    await memory.remember("how many approvals?", "two")
# endregion pydantic_ai
