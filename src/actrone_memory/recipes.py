"""Framework recipes, the "install, paste, run" onboarding surface of the ``actrone-memory`` CLI.

Selection, not detection: the developer picks a framework and gets a short recipe. Each recipe's
code is the ``# region`` block of a mypy-checked example under ``examples/frameworks/``, generated
into ``_example_snippets.py`` by ``scripts/extract_snippets.py`` and drift-checked in CI, so what
``actrone-memory add`` prints and writes is exactly the code CI type-checks. The Python
counterpart of the TypeScript ``recipes.ts`` (``npx actrone-memory add``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from actrone_memory._example_snippets import EXAMPLE_SNIPPETS

# The line that ends every recipe: where hosted, governed memory plugs in later.
HOSTED_UPGRADE_HINT = (
    "# Hosted, governed memory: when Actrone's hosted platform launches, "
    "swap MemoryManager for its drop-in ActroneMemoryManager."
)


@dataclass(frozen=True)
class Recipe:
    """One framework's copy-paste onboarding recipe."""

    framework: str  # slug used by the CLI (`actrone-memory add <framework>`)
    label: str  # human label
    install: str  # the install command (extra included)
    snippet: str  # the mypy-checked example code, ending with the hosted-upgrade line


# Slug, label and pip extra, in the order the recipes were introduced.
_FRAMEWORKS: tuple[tuple[str, str, str], ...] = (
    ("core", "Framework-agnostic core", ""),
    ("langchain", "LangChain", "langchain"),
    ("langgraph", "LangGraph", "langgraph"),
    ("crewai", "CrewAI", "crewai"),
    ("autogen", "AutoGen", "autogen"),
    ("llamaindex", "LlamaIndex", "llamaindex"),
    ("haystack", "Haystack", "haystack"),
    ("dspy", "DSPy", "dspy"),
    ("openai_agents", "OpenAI Agents SDK", "openai_agents"),
    ("pydantic_ai", "Pydantic AI", "pydantic_ai"),
    ("claude_agent_sdk", "Claude Agent SDK", "claude_agent_sdk"),
    ("semantic_kernel", "Semantic Kernel", "semantic_kernel"),
    ("google_adk", "Google ADK", "google_adk"),
    ("agno", "Agno", "agno"),
    ("smolagents", "smolagents", "smolagents"),
    ("aws_strands", "AWS Strands", "strands"),
    ("microsoft_agent_framework", "Microsoft Agent Framework", "microsoft_agent_framework"),
)

# Every framework slug the recipes cover, including any whose example is missing (for tests).
RECIPE_FRAMEWORKS: tuple[str, ...] = tuple(slug for slug, _, _ in _FRAMEWORKS)


def _build_recipes() -> dict[str, Recipe]:
    # A framework whose example is missing is left out rather than breaking the import; the test
    # suite requires every framework above to have one.
    recipes: dict[str, Recipe] = {}
    for framework, label, extra in _FRAMEWORKS:
        code = EXAMPLE_SNIPPETS.get(framework)
        if code is None:
            continue
        install = (
            f'pip install "actrone-memory[{extra}]"' if extra else "pip install actrone-memory"
        )
        recipes[framework] = Recipe(
            framework, label, install, f"{code.rstrip()}\n{HOSTED_UPGRADE_HINT}"
        )
    return recipes


RECIPES: dict[str, Recipe] = _build_recipes()

_ENTRY_POINT = re.compile(r"^async def (\w+)\(\)", re.M)


def list_frameworks() -> list[str]:
    """The framework slugs, sorted, for ``--list`` and validation."""
    return sorted(RECIPES)


def get_recipe(framework: str) -> Recipe | None:
    """Look up a recipe by slug (case-insensitive)."""
    return RECIPES.get(framework.strip().lower())


def render_recipe(recipe: Recipe) -> str:
    """Render a recipe as a readable "install, paste, run" block for stdout."""
    return (
        f"# {recipe.label}, memory in a few lines\n\n"
        f"1) Install\n   {recipe.install}\n\n"
        "2) Paste into your agent (or a new file), replacing the example ids, query and answer\n\n"
        f"{recipe.snippet}\n\n"
        "This is the code of a mypy-checked example (examples/frameworks/), type-checked in CI "
        "against the current API. Docs: https://actrone.com/docs/memory/overview"
    )


def render_standalone_file(recipe: Recipe) -> str:
    """Render a recipe as a self-contained new file for ``--write`` (never edits yours).

    The file is importable as written, and running it calls the example once.
    """
    entry = _ENTRY_POINT.search(recipe.snippet)
    runner = (
        '\n\nif __name__ == "__main__":\n'
        f"    import asyncio\n\n    asyncio.run({entry.group(1)}())\n"
        if entry
        else "\n"
    )
    return (
        f"# actrone-memory, {recipe.label} recipe (generated; safe to edit).\n"
        f"# Install: {recipe.install}\n"
        "# A NEW self-contained file: importable as written, and `python <file>` runs the example\n"
        "# once. Replace the example ids, query and answer, then import what you need.\n\n"
        f"{recipe.snippet}{runner}"
    )
