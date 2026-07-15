#!/usr/bin/env python
"""compat-matrix (vendored, per-repo) — expand THIS repo's compatibility.yaml into a CI test matrix.

Each framework is tested at THREE range boundaries (floor / current / next), each in its OWN isolated
environment (frameworks are mutually incompatible — e.g. agent-framework vs openai — so they must never
share a venv). Reads the repo's ``compatibility.yaml`` (at the repo root, the parent of ci/compat-matrix/)
and emits the job list as JSON for GitHub Actions ``matrix.include``.

    python ci/compat-matrix/matrix.py emit actrone-memory-py

Each job: {package, framework, primary, spec, canary, which, allow_fail}. ``next`` (pre/next-major) jobs
are ``allow_fail: true`` — early warning, not a gate.

Vendored from the workspace ``ci/compat-matrix/`` into this repo (per-repo split). The two Python repos'
copies are identical — keep them in sync when the shared logic changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Interface-canary test path per package (relative to the repo root).
CANARY_PATHS = {
    "actrone-memory-py": "tests/contract/test_interface_canaries.py",
    "actrone-py": "tests/unit/test_interface_canaries.py",
}


def _spec(primary: str, which: str, version: str, cap: str) -> str:
    """Build the pip requirement for a boundary. floor → ==version; current → >=version,<cap;
    next → >=version (installed with --pre by the runner)."""
    if which == "floor":
        return f"{primary}=={version}"
    if which == "current":
        return f"{primary}>={version},<{cap}"
    return f"{primary}>={version}"  # next (pre-release; runner adds --pre)


def emit(package: str) -> list[dict[str, object]]:
    # The repo IS the package: compatibility.yaml sits at the repo root — the parent of ci/compat-matrix/.
    root = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load((root / "compatibility.yaml").read_text(encoding="utf-8"))
    canary = CANARY_PATHS[package]
    jobs: list[dict[str, object]] = []
    for name, fw in manifest["frameworks"].items():
        primary = fw["packages"][0]
        cap = str(fw["cap"])
        for which in ("floor", "current", "next"):
            version = str(fw["matrix"][which])
            jobs.append(
                {
                    "package": package,
                    "framework": name,
                    "primary": primary,
                    "spec": _spec(primary, which, version, cap),
                    "canary": canary,
                    "which": which,
                    "allow_fail": which == "next",
                }
            )
    return jobs


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "emit":
        print("usage: matrix.py emit <package-name>", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(emit(sys.argv[2])))


if __name__ == "__main__":
    main()
