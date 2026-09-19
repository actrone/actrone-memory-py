"""Google ADK, mypy-checked example source for the docs."""
# region google_adk
from actrone_memory import MemoryManager
from actrone_memory.integrations.google_adk import ActroneGoogleADKMemory


async def google_adk_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneGoogleADKMemory("support-bot", "s1", memory_manager=mm)
    # Tier 2: Runner(agent=agent, app_name="support", memory_service=memory.as_memory_service())
    context = await memory.build_context("index rebuild schedule")  # Tier 1
    _ = context
    await memory.remember("when does the index rebuild?", "nightly at 2am")
# endregion google_adk
