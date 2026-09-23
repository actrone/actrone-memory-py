"""``actrone-memory`` CLI, the non-destructive, existing-project onboarding path.

It only **prints** a framework recipe or **creates one new file**; it never reads or edits your
existing code. The Python counterpart of ``npx actrone-memory add`` (literal DX symmetry)::

    actrone-memory add langgraph                    # print install + recipe
    actrone-memory add langgraph --write memory.py  # write ONE new file
    actrone-memory list                             # list supported frameworks
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from actrone_memory.recipes import (
    get_recipe,
    list_frameworks,
    render_recipe,
    render_standalone_file,
)

USAGE = (
    "actrone-memory, add memory to your agent (non-destructive)\n\n"
    "Usage:\n"
    "  actrone-memory add <framework> [--write <file>]   print a recipe, or write ONE new file\n"
    "  actrone-memory list                               list supported frameworks\n\n"
    f"Frameworks: {', '.join(list_frameworks())}"
)


class CliIO(Protocol):
    """Injected IO so the command logic is unit-testable without touching disk."""

    def log(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def file_exists(self, path: str) -> bool: ...
    def write_file(self, path: str, content: str) -> None: ...


def run_cli(args: list[str], io: CliIO) -> int:
    """Run the CLI with parsed args against an injected :class:`CliIO`. Returns an exit code."""
    command = args[0] if args else None
    rest = args[1:]

    if command is None or command in ("help", "--help", "-h"):
        io.log(USAGE)
        return 1 if command is None else 0

    if command in ("list", "--list"):
        io.log("\n".join(list_frameworks()))
        return 0

    if command != "add":
        io.error(f"Unknown command: {command}\n\n{USAGE}")
        return 1

    framework = next((a for a in rest if not a.startswith("--")), None)
    if framework is None:
        io.error(f"Missing framework.\n\n{USAGE}")
        return 1

    recipe = get_recipe(framework)
    if recipe is None:
        io.error(f"Unknown framework: {framework}\nAvailable: {', '.join(list_frameworks())}")
        return 1

    if "--write" in rest:
        idx = rest.index("--write")
        target = rest[idx + 1] if idx + 1 < len(rest) else None
        if target is None or target.startswith("--"):
            io.error("--write requires a file path, e.g. --write memory.py")
            return 1
        if io.file_exists(target):
            # Non-destructive contract: never overwrite existing files.
            io.error(f"Refusing to overwrite existing file: {target}")
            return 1
        io.write_file(target, render_standalone_file(recipe))
        io.log(
            f"Wrote {target} (a new self-contained file).\n"
            f"Install: {recipe.install}\n"
            "Import what you need from it into your agent, nothing in your project was modified."
        )
        return 0

    io.log(render_recipe(recipe))
    return 0


class _StdIO:
    """Real IO wired to stdout/stderr + the filesystem."""

    def log(self, message: str) -> None:
        sys.stdout.write(f"{message}\n")

    def error(self, message: str) -> None:
        sys.stderr.write(f"{message}\n")

    def file_exists(self, path: str) -> bool:
        return Path(path).exists()

    def write_file(self, path: str, content: str) -> None:
        Path(path).write_text(content, encoding="utf-8")


def main() -> None:
    """Console-script entry point (``actrone-memory``)."""
    sys.exit(run_cli(sys.argv[1:], _StdIO()))


if __name__ == "__main__":
    main()
