#!/usr/bin/env python
"""extract-snippets — pull the ``# region``-marked blocks out of the type-checked example files into
``examples/snippets.json`` (Public-Domain Cutover Runbook Phase 6, item 5).

The docs render these by id instead of hand-typing code, so every documented snippet is real,
mypy-checked code from the current library — an example that stops type-checking fails CI before it
can be published stale.

Markers::

    # region <id>
    ...code...
    # endregion <id>

``--check`` verifies ``snippets.json`` is in sync with the sources (CI gate) instead of writing it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
OUT = EXAMPLES_DIR / "snippets.json"

REGION = re.compile(r"#\s*region\s+(\S+)\s*\n(.*?)\n\s*#\s*endregion", re.DOTALL)


def extract() -> dict[str, str]:
    snippets: dict[str, str] = {}
    # Top-level examples + the per-framework recipe sources under examples/frameworks/ (mirrors the
    # TS examples/frameworks/ pattern) — each # region block becomes a doc snippet.
    sources = sorted(EXAMPLES_DIR.glob("*.py")) + sorted((EXAMPLES_DIR / "frameworks").glob("*.py"))
    for path in sources:
        for match in REGION.finditer(path.read_text(encoding="utf-8")):
            snippet_id, body = match.group(1), match.group(2)
            if snippet_id in snippets:
                raise SystemExit(f"duplicate snippet id {snippet_id!r} (in {path.name})")
            snippets[snippet_id] = body.rstrip() + "\n"
    return snippets


def main() -> None:
    snippets = extract()
    if not snippets:
        raise SystemExit("[extract-snippets] no # region snippets found under examples/")
    serialized = json.dumps(snippets, indent=2, ensure_ascii=False) + "\n"
    ids = ", ".join(snippets)

    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != serialized:
            raise SystemExit(
                "[extract-snippets] DRIFT: examples/snippets.json is stale. "
                "Run `python scripts/extract_snippets.py` and commit."
            )
        print(f"[extract-snippets] snippets.json in sync ({len(snippets)}: {ids}) ok")
    else:
        OUT.write_text(serialized, encoding="utf-8")
        print(f"[extract-snippets] wrote {len(snippets)} snippet(s) -> snippets.json ({ids})")


if __name__ == "__main__":
    main()
