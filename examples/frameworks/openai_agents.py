"""OpenAI Agents SDK — mypy-checked example source for the docs."""
# region openai_agents
from actrone_memory import MemoryManager
from actrone_memory.integrations.openai_agents import ActroneOpenAIAgentsMemory


async def openai_agents_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneOpenAIAgentsMemory("support-bot", "s1", memory_manager=mm)
    instructions = await memory.instructions_for("You are support.", "enterprise plan")
    # ...Runner.run(Agent(name="support", instructions=instructions), user_input)...
    _ = instructions
    await memory.remember("what plan?", "enterprise")
# endregion openai_agents
