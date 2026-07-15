"""Microsoft Agent Framework — mypy-checked example source for the docs."""
# region microsoft_agent_framework
from actrone_memory import MemoryManager
from actrone_memory.integrations.microsoft_agent_framework import ActroneAgentFrameworkMemory


async def microsoft_agent_framework_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneAgentFrameworkMemory("support-bot", "s1", memory_manager=mm)
    # Tier 2: ChatAgent(chat_client=client, context_providers=[memory.as_context_provider()])
    context = await memory.build_context("who is the account owner")  # Tier 1
    _ = context
    await memory.remember("account owner?", "Jane Doe")
# endregion microsoft_agent_framework
