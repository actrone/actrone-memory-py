"""Claude Agent SDK — mypy-checked example source for the docs."""
# region claude_agent_sdk
from actrone_memory import MemoryManager
from actrone_memory.integrations.claude_agent_sdk import ActroneClaudeAgentMemory


async def claude_agent_sdk_example() -> None:
    mm = await MemoryManager.create()
    memory = ActroneClaudeAgentMemory("support-bot", "s1", memory_manager=mm)
    system_prompt = await memory.append_to_system_prompt("You are support.", "refunds policy")
    # ...query(prompt=user_input, options=ClaudeAgentOptions(system_prompt=system_prompt))...
    _ = system_prompt
    await memory.remember("refund policy?", "manager approval over 500")
# endregion claude_agent_sdk
