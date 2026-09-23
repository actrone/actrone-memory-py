"""LangGraph, mypy-checked example source for the docs."""
# region langgraph
from actrone_memory import MemoryManager
from actrone_memory.integrations.langgraph import ActroneCheckpointer


async def langgraph_example() -> None:
    mm = await MemoryManager.create()
    checkpointer = ActroneCheckpointer("support-bot", memory_manager=mm)
    # graph = builder.compile(checkpointer=checkpointer)
    # await graph.ainvoke(state, config={"configurable": {"thread_id": "s1"}})
    _ = checkpointer
# endregion langgraph
