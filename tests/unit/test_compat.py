"""Unit tests for the framework-compatibility drift gate (``scripts/check_compat.py``).

Proves the gate (a) reports NO drift on the real repo files, so the SSOT, pyproject extras, and the
README matrix genuinely agree today, and (b) actually CATCHES each drift class (wrong range,
missing README row, undocumented extra) against synthetic inputs. Framework-free, no network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_compat", _ROOT / "scripts" / "check_compat.py"
)
assert _spec and _spec.loader
check_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_compat)


def test_real_repo_is_in_sync() -> None:
    # The authoritative guarantee: compatibility.yaml, pyproject extras, and README agree right now.
    assert check_compat.check() == []


def test_spec_bounds_parses_lower_and_upper() -> None:
    assert check_compat._spec_bounds("langchain>=0.2,<2") == ("0.2", "2")
    assert check_compat._spec_bounds("agno>=1.0,<2") == ("1.0", "2")
    assert check_compat._spec_bounds("weird") == (None, None)


_MANIFEST: dict[str, Any] = {
    "frameworks": {
        "demo": {
            "extra": "demo",
            "packages": ["demo-pkg"],
            "floor": "1.0",
            "cap": "2",
            "matrix": {"floor": "1.0.0", "current": "1.5", "next": "2.0"},
            "readme": "Demo",
            "contract": ["demo:Agent"],
        }
    },
    "non_framework_extras": ["dev"],
}
_README_OK = "| Demo | `demo` | `>=1.0,<2` | 1 |"


def test_in_sync_synthetic_passes() -> None:
    extras = {"demo": ["demo-pkg>=1.0,<2"], "dev": ["pytest"]}
    assert check_compat.check_against(_MANIFEST, extras, _README_OK) == []


def test_catches_wrong_range() -> None:
    extras = {"demo": ["demo-pkg>=1.0,<3"]}  # cap 3 ≠ manifest 2
    errors = check_compat.check_against(_MANIFEST, extras, _README_OK)
    assert any("!= manifest (>=1.0,<2)" in e for e in errors)


def test_catches_missing_readme_row() -> None:
    extras = {"demo": ["demo-pkg>=1.0,<2"]}
    errors = check_compat.check_against(_MANIFEST, extras, "no matrix here")
    assert any("README" in e for e in errors)


def test_catches_undocumented_extra() -> None:
    extras = {"demo": ["demo-pkg>=1.0,<2"], "sneaky": ["sneaky>=1"]}
    errors = check_compat.check_against(_MANIFEST, extras, _README_OK)
    assert any("sneaky" in e for e in errors)


def test_catches_missing_package() -> None:
    extras = {"demo": ["wrong-pkg>=1.0,<2"]}  # framework's own 'demo-pkg' absent
    errors = check_compat.check_against(_MANIFEST, extras, _README_OK)
    assert any("missing package 'demo-pkg'" in e for e in errors)


def test_readme_ranges_optional_for_sdk_style_matrix() -> None:
    # With readme_shows_ranges=False (SDK D1/D2/D3 matrix), only the extra row must be present.
    extras = {"demo": ["demo-pkg>=1.0,<2"]}
    readme_no_range = "| Demo | `demo` | ✅ | ✅ |"
    assert check_compat.check_against(_MANIFEST, extras, readme_no_range) != []  # ranges required
    assert (
        check_compat.check_against(_MANIFEST, extras, readme_no_range, readme_shows_ranges=False)
        == []
    )


def _with(**changes: Any) -> dict[str, Any]:
    """The synthetic manifest with the demo framework's fields replaced."""
    return {**_MANIFEST, "frameworks": {"demo": {**_MANIFEST["frameworks"]["demo"], **changes}}}


def test_catches_framework_without_contract() -> None:
    # With no contract symbols the compat-matrix job selects zero tests (pytest exit 5).
    extras = {"demo": ["demo-pkg>=1.0,<2"]}
    errors = check_compat.check_against(_with(contract=[]), extras, _README_OK)
    assert any("no contract symbols" in e for e in errors)


def test_catches_matrix_floor_that_is_not_the_declared_floor() -> None:
    extras = {"demo": ["demo-pkg>=1.0,<2"]}
    manifest = _with(matrix={"floor": "0.9.0", "current": "1.5", "next": "2.0"})
    errors = check_compat.check_against(manifest, extras, _README_OK)
    assert any("matrix.floor" in e for e in errors)


def test_catches_next_inside_the_supported_range() -> None:
    # A `next` below the cap only re-tests `current` and never warns about the next major.
    extras = {"demo": ["demo-pkg>=1.0,<2"]}
    manifest = _with(matrix={"floor": "1.0.0", "current": "1.5", "next": "1.8"})
    errors = check_compat.check_against(manifest, extras, _README_OK)
    assert any("matrix.next" in e for e in errors)
