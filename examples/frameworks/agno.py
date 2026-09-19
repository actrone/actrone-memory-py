"""Agno, mypy-checked example source for the docs."""
# region agno
from actrone_memory import MemoryManager
from actrone_memory.integrations.agno import ActroneAgnoMemory


async def agno_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneAgnoMemory("support-bot", "s1", memory_manager=mm)
    context = await memory.additional_context("how do escalations work")
    # ...Agent(model=model, additional_context=context); await agent.arun(user_input)...
    _ = context
    await memory.remember("how do escalations work?", "they page the on-call engineer")
# endregion agno
