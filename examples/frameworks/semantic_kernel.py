"""Semantic Kernel, mypy-checked example source for the docs."""
# region semantic_kernel
from actrone_memory import MemoryManager
from actrone_memory.integrations.semantic_kernel import ActroneSemanticKernelMemory


async def semantic_kernel_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneSemanticKernelMemory("support-bot", "s1", memory_manager=mm)
    # Tier 1: a system-message string; or Tier 2: memory.add_to_chat_history(history, query)
    system = await memory.system_message("ticket sla")
    _ = system
    await memory.remember("what is the sla?", "24 hours")
# endregion semantic_kernel
