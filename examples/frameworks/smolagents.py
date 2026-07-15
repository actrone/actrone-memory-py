"""smolagents — mypy-checked example source for the docs."""
# region smolagents
from actrone_memory import MemoryManager
from actrone_memory.integrations.smolagents import ActroneSmolagentsMemory


async def smolagents_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneSmolagentsMemory("support-bot", "s1", memory_manager=mm)
    user_input = "what is the build server named?"
    task = await memory.task_context(user_input) + user_input
    # ...result = agent.run(task)...
    _ = task
    await memory.remember(user_input, "atlas")
# endregion smolagents
