"""LangChain, mypy-checked example source for the docs."""
# region langchain
from actrone_memory import MemoryManager
from actrone_memory.integrations.langchain import ActroneChatMessageHistory


async def langchain_example() -> None:
    mm = await MemoryManager.create()
    # Works on LangChain 0.x and 1.x (1.x removed BaseMemory; chat history survived).
    history = ActroneChatMessageHistory("support-bot", "s1", memory_manager=mm)
    # RunnableWithMessageHistory(runnable, lambda _session_id: history,
    #     input_messages_key="input", history_messages_key="history")
    context = await history.build_context("deployment approvals")
    _ = (context, await history.aget_messages())
# endregion langchain
