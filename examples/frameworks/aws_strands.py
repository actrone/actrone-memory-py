"""AWS Strands — mypy-checked example source for the docs."""
# region aws_strands
from actrone_memory import MemoryManager
from actrone_memory.integrations.aws_strands import ActroneStrandsMemory


async def aws_strands_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneStrandsMemory("support-bot", "s1", memory_manager=mm)
    system_prompt = await memory.system_prompt("You are support.", "friday deploy freeze")
    # ...agent = Agent(model=model, system_prompt=system_prompt); agent(user_input)...
    _ = system_prompt
    await memory.remember("can we deploy friday?", "no, prod deploys are frozen on fridays")
# endregion aws_strands
