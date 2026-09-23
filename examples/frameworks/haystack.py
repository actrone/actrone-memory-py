"""Haystack, mypy-checked example source for the docs."""
# region haystack
from actrone_memory import MemoryManager
from actrone_memory.integrations.haystack import ActroneRetriever, ActroneWriter


async def haystack_example() -> None:
    mm = await MemoryManager.create()
    retriever = ActroneRetriever("support-bot", memory_manager=mm)
    writer = ActroneWriter("support-bot", "s1", memory_manager=mm)
    # pipeline.add_component("memory", retriever); pipeline.add_component("writer", writer)
    _ = (retriever, writer)
# endregion haystack
