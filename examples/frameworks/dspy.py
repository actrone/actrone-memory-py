"""DSPy, mypy-checked example source for the docs."""
# region dspy
from actrone_memory import MemoryManager
from actrone_memory.integrations.dspy import ActroneRM


async def dspy_example() -> None:
    mm = await MemoryManager.create()
    rm = ActroneRM("support-bot", k=5, memory_manager=mm)
    # dspy.settings.configure(rm=rm); then dspy.Retrieve(k=5)(question) inside your module
    await rm.store_turn("support-bot", "s1", "how many approvals?", "two")
# endregion dspy
