"""Framework-agnostic core — mypy-checked example source for the docs."""
# region core
from actrone_memory import MemoryManager


async def core_example() -> None:
    mm = await MemoryManager.create()  # zero services, local by default
    ctx = await mm.retrieve_context("support-bot", "s1", "deployment approvals", 2000)
    _ = ctx.episodic_memories  # prepend these + ctx.recent_turns to your prompt
    await mm.store_turn("support-bot", "s1", "how many approvals?", "two")
# endregion core
