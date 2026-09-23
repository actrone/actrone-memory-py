"""AutoGen, mypy-checked example source for the docs."""
# region autogen
from actrone_memory import MemoryManager
from actrone_memory.integrations.autogen import ActroneAutoGenMemory


async def autogen_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneAutoGenMemory("support-bot", "s1", memory_manager=mm)
    # AssistantAgent("support", model_client=client, memory=[memory])
    _ = memory
# endregion autogen
