"""CrewAI, mypy-checked example source for the docs."""
# region crewai
from actrone_memory import MemoryManager
from actrone_memory.integrations.crewai import ActroneCrewMemory


async def crewai_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneCrewMemory("support-bot", "s1", memory_manager=mm)
    context = await memory.build_context("deployment approvals")
    # Task(description=f"{context}\n\n{user_input}", agent=agent), then crew.kickoff()
    _ = context
    await memory.save("Deploys need two approvals.", {"task_input": "how many approvals?"})
# endregion crewai
