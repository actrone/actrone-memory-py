"""Framework recipes — the "install → paste → run" onboarding surface, and the single source of
truth for the ``actrone-memory`` CLI and the docs. Selection, not detection: the developer picks a
framework and gets a short, identical-shape recipe. The Python counterpart of the TypeScript
``recipes.ts`` (literal DX symmetry with ``npx @actrone/memory add``).
"""

from __future__ import annotations

from dataclasses import dataclass

# The seed line that ends every recipe: the one-import path to hosted, governed memory.
HOSTED_UPGRADE_HINT = (
    "# ⬆ swap MemoryManager for actrone's hosted ActroneMemoryManager → hosted, governed"
)


@dataclass(frozen=True)
class Recipe:
    """One framework's copy-paste onboarding recipe."""

    framework: str  # slug used by the CLI (`actrone-memory add <framework>`)
    label: str  # human label
    install: str  # the install command (extra included)
    snippet: str  # the paste-in snippet (ends with the hosted-upgrade seed)


_CORE_HEADER = (
    "from actrone_memory import MemoryManager\n\n"
    "mm = await MemoryManager.create()  # zero services, local by default\n"
)


def _recipe(framework: str, label: str, extra: str, body: str) -> Recipe:
    install = f"pip install actrone-memory[{extra}]" if extra else "pip install actrone-memory"
    return Recipe(framework, label, install, f"{body}\n{HOSTED_UPGRADE_HINT}")


RECIPES: dict[str, Recipe] = {
    "core": _recipe(
        "core",
        "Framework-agnostic core",
        "",
        _CORE_HEADER
        + "ctx = await mm.retrieve_context('support-bot', session_id, user_input, 2000)\n"
        + "# ...prepend ctx.episodic_memories / ctx.recent_turns to your prompt...\n"
        + "await mm.store_turn('support-bot', session_id, user_input, answer)",
    ),
    "langchain": _recipe(
        "langchain",
        "LangChain",
        "langchain",
        "from actrone_memory.integrations.langchain import ActroneMemory\n\n"
        "memory = ActroneMemory(agent_id='support-bot', session_id='s1')\n"
        "chain = ConversationChain(llm=llm, memory=memory)  # drop-in BaseMemory",
    ),
    "langgraph": _recipe(
        "langgraph",
        "LangGraph",
        "langgraph",
        "from actrone_memory.integrations.langgraph import ActroneCheckpointer\n\n"
        "graph = builder.compile(checkpointer=ActroneCheckpointer(agent_id='support-bot'))",
    ),
    "crewai": _recipe(
        "crewai",
        "CrewAI",
        "crewai",
        "from actrone_memory.integrations.crewai import ActroneCrewMemory\n\n"
        "crew = Crew(agents=[...], tasks=[...], memory=ActroneCrewMemory(agent_id='support-bot'))",
    ),
    "autogen": _recipe(
        "autogen",
        "AutoGen",
        "autogen",
        "from actrone_memory.integrations.autogen import ActroneAutoGenMemory\n\n"
        "memory = ActroneAutoGenMemory(agent_id='support-bot', session_id='s1')\n"
        "agent = AssistantAgent('support', model_client=client, memory=[memory])",
    ),
    "llamaindex": _recipe(
        "llamaindex",
        "LlamaIndex",
        "llamaindex",
        "from actrone_memory.integrations.llamaindex import ActroneLlamaMemory\n\n"
        "memory = ActroneLlamaMemory(agent_id='support-bot', session_id='s1')\n"
        "agent = FunctionAgent(tools=[...], llm=llm)\n"
        "resp = await agent.run(user_input, memory=memory)",
    ),
    "haystack": _recipe(
        "haystack",
        "Haystack",
        "haystack",
        "from actrone_memory.integrations.haystack import ActroneRetriever, ActroneWriter\n\n"
        "pipe.add_component('memory', ActroneRetriever(agent_id='support-bot'))\n"
        "pipe.add_component('writer', ActroneWriter(agent_id='support-bot'))",
    ),
    "dspy": _recipe(
        "dspy",
        "DSPy",
        "dspy",
        "import dspy\n"
        "from actrone_memory.integrations.dspy import ActroneRM\n\n"
        "dspy.settings.configure(rm=ActroneRM(agent_id='support-bot'))",
    ),
    "openai_agents": _recipe(
        "openai_agents",
        "OpenAI Agents SDK",
        "openai_agents",
        "from actrone_memory.integrations.openai_agents import ActroneOpenAIAgentsMemory\n\n"
        "memory = ActroneOpenAIAgentsMemory(agent_id='support-bot', session_id='s1')\n"
        "instructions = await memory.instructions_for('You are support.', user_input)\n"
        "result = await Runner.run(Agent(name='support', instructions=instructions), user_input)\n"
        "await memory.remember(user_input, result.final_output)",
    ),
    "pydantic_ai": _recipe(
        "pydantic_ai",
        "Pydantic AI",
        "pydantic_ai",
        "from actrone_memory.integrations.pydantic_ai import ActronePydanticAIMemory\n\n"
        "memory = ActronePydanticAIMemory(agent_id='support-bot', session_id='s1')\n\n"
        "@agent.system_prompt\n"
        "async def with_memory(ctx: RunContext[str]) -> str:\n"
        "    return await memory.system_prompt(ctx.deps)",
    ),
    "claude_agent_sdk": _recipe(
        "claude_agent_sdk",
        "Claude Agent SDK",
        "claude_agent_sdk",
        "from actrone_memory.integrations.claude_agent_sdk import ActroneClaudeAgentMemory\n\n"
        "memory = ActroneClaudeAgentMemory(agent_id='support-bot', session_id='s1')\n"
        "sys_prompt = await memory.append_to_system_prompt('You are support.', user_input)\n"
        "# ...query(prompt=user_input, options=ClaudeAgentOptions(system_prompt=sys_prompt))...\n"
        "await memory.remember(user_input, answer)",
    ),
    "semantic_kernel": _recipe(
        "semantic_kernel",
        "Semantic Kernel",
        "semantic_kernel",
        "from actrone_memory.integrations.semantic_kernel import ActroneSemanticKernelMemory\n\n"
        "memory = ActroneSemanticKernelMemory(agent_id='support-bot', session_id='s1')\n"
        "history = ChatHistory()\n"
        "await memory.add_to_chat_history(history, user_input)  # native ChatHistory\n"
        "await memory.remember(user_input, answer)",
    ),
    "google_adk": _recipe(
        "google_adk",
        "Google ADK",
        "google_adk",
        "from actrone_memory.integrations.google_adk import ActroneGoogleADKMemory\n\n"
        "memory = ActroneGoogleADKMemory(agent_id='support-bot', session_id='s1')\n"
        "runner = Runner(agent=agent, app_name='support', "
        "memory_service=memory.as_memory_service())",
    ),
    "agno": _recipe(
        "agno",
        "Agno",
        "agno",
        "from actrone_memory.integrations.agno import ActroneAgnoMemory\n\n"
        "memory = ActroneAgnoMemory(agent_id='support-bot', session_id='s1')\n"
        "context = await memory.additional_context(user_input)\n"
        "agent = Agent(model=model, additional_context=context)\n"
        "resp = await agent.arun(user_input)\n"
        "await memory.remember(user_input, resp.content)",
    ),
    "smolagents": _recipe(
        "smolagents",
        "smolagents",
        "smolagents",
        "from actrone_memory.integrations.smolagents import ActroneSmolagentsMemory\n\n"
        "memory = ActroneSmolagentsMemory(agent_id='support-bot', session_id='s1')\n"
        "task = await memory.task_context(user_input) + user_input\n"
        "result = agent.run(task)\n"
        "await memory.remember(user_input, str(result))",
    ),
    "aws_strands": _recipe(
        "aws_strands",
        "AWS Strands",
        "strands",
        "from actrone_memory.integrations.aws_strands import ActroneStrandsMemory\n\n"
        "memory = ActroneStrandsMemory(agent_id='support-bot', session_id='s1')\n"
        "system_prompt = await memory.system_prompt('You are support.', user_input)\n"
        "agent = Agent(model=model, system_prompt=system_prompt)\n"
        "result = agent(user_input)\n"
        "await memory.remember(user_input, str(result))",
    ),
    "microsoft_agent_framework": _recipe(
        "microsoft_agent_framework",
        "Microsoft Agent Framework",
        "microsoft_agent_framework",
        "from actrone_memory.integrations.microsoft_agent_framework import (\n"
        "    ActroneAgentFrameworkMemory,\n"
        ")\n\n"
        "memory = ActroneAgentFrameworkMemory(agent_id='support-bot', session_id='s1')\n"
        "agent = ChatAgent(chat_client=client, context_providers=[memory.as_context_provider()])",
    ),
}


def list_frameworks() -> list[str]:
    """The framework slugs, sorted, for ``--list`` and validation."""
    return sorted(RECIPES)


def get_recipe(framework: str) -> Recipe | None:
    """Look up a recipe by slug (case-insensitive)."""
    return RECIPES.get(framework.strip().lower())


def render_recipe(recipe: Recipe) -> str:
    """Render a recipe as a readable "install → paste → run" block for stdout."""
    return (
        f"# {recipe.label} — memory in a few lines\n\n"
        f"1) Install\n   {recipe.install}\n\n"
        f"2) Paste into your agent (or a new file)\n\n{recipe.snippet}\n\n"
        "Docs: https://docs.actrone.com/memory"
    )


def render_standalone_file(recipe: Recipe) -> str:
    """Render a recipe as a self-contained new file for ``--write`` (never edits yours)."""
    return (
        f"# actrone-memory — {recipe.label} recipe (generated; safe to edit).\n"
        f"# Install: {recipe.install}\n"
        "# This is a NEW self-contained file. Import what you need from it into your agent.\n\n"
        f"{recipe.snippet}\n"
    )
