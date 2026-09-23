"""LlamaIndex, mypy-checked example source for the docs."""
# region llamaindex
from actrone_memory import MemoryManager
from actrone_memory.integrations.llamaindex import ActroneLlamaMemory


async def llamaindex_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneLlamaMemory("support-bot", "s1", memory_manager=mm)
    # SimpleChatEngine.from_defaults(llm=llm, memory=memory)
    _ = memory
# endregion llamaindex
