#!/usr/bin/env python
"""check-compat, the framework-compatibility drift gate.

Makes ``compatibility.yaml`` the single source of truth: fails if the ``[project.optional-
dependencies]`` extras in ``pyproject.toml`` or the framework compatibility matrix in ``README.md``
disagree with it. Run in CI so "declared" (pyproject/README) can never drift from "tested" (the
compat-matrix job, which reads the same file).

Checks, per framework in the manifest:
  1. the pyproject extra exists and each of its packages is pinned exactly ``>={floor},<{cap}``;
  2. the README compatibility matrix has a row naming the extra and the ``>={floor},<{cap}`` range.
Plus the reverse: every pyproject optional-dependency key is either a manifest framework or a known
non-framework extra (so a new extra can't sneak in undocumented).

Exit 0 when in sync; exit 1 with a precise diff otherwise. No framework installs, no network.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "compatibility.yaml"
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"


def _load() -> tuple[dict[str, Any], dict[str, list[str]], str]:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    readme = README.read_text(encoding="utf-8")
    return manifest, extras, readme


def _spec_bounds(spec: str) -> tuple[str | None, str | None]:
    """Extract ``(lower, upper)`` from a requirement spec like ``langchain>=0.2,<2``."""
    lower = re.search(r">=\s*([0-9][0-9.]*)", spec)
    upper = re.search(r"<\s*([0-9][0-9.]*)", spec)
    return (lower.group(1) if lower else None, upper.group(1) if upper else None)


def check() -> list[str]:
    """Validate the real repo files. Returns a list of drift errors ([] when in sync)."""
    return check_against(*_load())


def _pkg_name(spec: str) -> str:
    """The distribution name from a spec (``autogen-ext[openai]>=0.4`` → ``autogen-ext``)."""
    return re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0]


def _version_key(version: str) -> tuple[int, ...]:
    """Numeric sort key for a plain release version (``0.4.2`` → ``(0, 4, 2)``)."""
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _matrix_errors(name: str, fw: dict[str, Any], floor: str, cap: str) -> list[str]:
    """Rules that keep the compat-matrix job meaningful, caught here on every PR, not weekly.

    - ``contract`` must name at least one symbol: with none, the job selects zero tests and fails
      with pytest's opaque "no tests collected" (exit 5).
    - ``matrix.floor`` must be the declared floor (the job pins it with ``==``).
    - ``matrix.next`` must sit at or beyond ``cap``: a ``next`` inside the range only re-tests
      ``current`` and never warns about the next major.
    """
    errors: list[str] = []
    if not fw.get("contract"):
        errors.append(
            f"[{name}] no contract symbols, so its compat-matrix canary would test nothing"
        )
    matrix = fw.get("matrix") or {}
    pinned_floor = str(matrix.get("floor", ""))
    if _version_key(pinned_floor)[: len(_version_key(floor))] != _version_key(floor):
        errors.append(f"[{name}] matrix.floor {pinned_floor!r} is not the declared floor {floor!r}")
    next_version = str(matrix.get("next", ""))
    if next_version and _version_key(next_version) < _version_key(cap):
        errors.append(
            f"[{name}] matrix.next {next_version!r} is inside the supported range (<{cap}); "
            f"set it to {cap!r} or later so the job tests the next major"
        )
    return errors


def check_against(
    manifest: dict[str, Any],
    extras: dict[str, list[str]],
    readme: str,
    *,
    readme_shows_ranges: bool = True,
) -> list[str]:
    """Pure core: validate ``extras`` (pyproject) + ``readme`` vs ``manifest``. Unit-testable.

    An extra may bundle supporting deps (e.g. ``openai``/``litellm``) beyond the framework itself,
    so the package rule is a SUBSET: every manifest ``packages`` entry must be present with range
    ``>={floor},<{cap}``; extra packages are allowed. ``readme_shows_ranges`` gates whether the
    README row must carry the version range (OSS matrices do; the SDK's D1/D2/D3 matrix does not).
    """
    frameworks: dict[str, Any] = manifest["frameworks"]
    non_framework = set(manifest["non_framework_extras"])
    errors: list[str] = []

    for name, fw in frameworks.items():
        extra = fw["extra"]
        floor, cap = str(fw["floor"]), str(fw["cap"])

        # 1. pyproject extra ↔ manifest (subset rule on the framework's own packages)
        if extra not in extras:
            errors.append(f"[{name}] pyproject has no extra '{extra}'")
            continue
        by_pkg = {_pkg_name(s): s for s in extras[extra]}
        for pkg in fw["packages"]:
            if pkg not in by_pkg:
                errors.append(f"[{name}] extra '{extra}' is missing package '{pkg}'")
                continue
            low, up = _spec_bounds(by_pkg[pkg])
            if low != floor or up != cap:
                errors.append(
                    f"[{name}] '{by_pkg[pkg]}' (>={low},<{up}) != manifest (>={floor},<{cap})"
                )

        # 2. README compatibility matrix ↔ manifest. The range is checked in the framework's OWN
        # row: anywhere in the file let one row's range (`>=1.0,<2`) vouch for another's.
        rows = [
            line for line in readme.splitlines() if line.startswith("|") and f"`{extra}`" in line
        ]
        if not rows:
            errors.append(f"[{name}] README has no compat-matrix row for extra `{extra}`")
        elif readme_shows_ranges and not any(f">={floor},<{cap}" in row for row in rows):
            errors.append(
                f"[{name}] README row for `{extra}` does not show the range `>={floor},<{cap}`"
            )

        # 3. the version-matrix job for this framework can actually test something
        errors.extend(_matrix_errors(name, fw, floor, cap))

    # reverse: no undocumented framework extras
    known = {fw["extra"] for fw in frameworks.values()} | non_framework
    for extra in extras:
        if extra not in known:
            errors.append(
                f"[extra '{extra}'] present in pyproject but not in compatibility.yaml "
                f"(add it as a framework, or to non_framework_extras)"
            )
    return errors


def main() -> None:
    errors = check()
    if errors:
        print("[check-compat] DRIFT, pyproject/README disagree with compatibility.yaml:\n")
        for e in errors:
            print(f"  x {e}")
        print("\nUpdate compatibility.yaml (the source of truth) or fix the drift, then re-run.")
        sys.exit(1)
    n = len(_load()[0]["frameworks"])
    print(f"[check-compat] in sync, {n} frameworks; pyproject extras + README matrix agree OK")


if __name__ == "__main__":
    main()
