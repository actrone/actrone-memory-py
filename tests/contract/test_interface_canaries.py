"""Interface canaries, assert the STABLE framework symbols our adapters bind to still exist.

Driven by the ``contract:`` field in ``compatibility.yaml`` (``module:Symbol``). Each entry is
``importorskip``-ed, so the base venv skips them; the compat-matrix CI job installs each framework
its floor/current/next version and runs these, so a framework that renames or removes a symbol we
depend on (e.g. ``BaseMemoryService`` → something else) fails *precisely* with "the interface
changed" rather than a confusing downstream error, the machine version of a "verified against vX".
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
from pathlib import Path

import pytest
import yaml

# Canaries assert against a SPECIFIC pinned framework version, so they only make sense inside a
# compat-matrix job (which installs that version). The runner sets COMPAT_MATRIX=1; otherwise the
# whole module skips, so a dev venv that happens to have an out-of-range framework never fails here.
pytestmark = pytest.mark.skipif(
    not os.environ.get("COMPAT_MATRIX"),
    reason="interface canaries run only in the compat-matrix (set COMPAT_MATRIX=1)",
)

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "compatibility.yaml").read_text(encoding="utf-8"))

# Flatten every framework's contract symbols into (framework, "module:Symbol") pairs.
_CONTRACTS = [
    (name, symbol)
    for name, fw in _MANIFEST["frameworks"].items()
    for symbol in fw.get("contract", [])
]


# A contract entry is `module:Symbol`, optionally with the keyword parameters our docs tell users to
# pass: `agno.agent:Agent(additional_context)`. A renamed parameter breaks users as surely as a
# removed class, so the parameters are asserted against the callable's real signature too.
_ENTRY = re.compile(r"^(?P<symbol>[^()]+?)(?:\((?P<params>[^)]*)\))?$")


def _assert_accepts(framework: str, ref: str, obj: object, params: list[str]) -> None:
    """Fail unless ``obj`` (a class or function) accepts every keyword parameter in ``params``."""
    try:
        signature = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        pytest.fail(f"{framework}: cannot read the signature of {ref} to check {params} ({exc})")
    accepted = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return  # takes **kwargs: the names cannot be checked statically, and nothing is rejected
    missing = [name for name in params if name not in accepted]
    assert not missing, (
        f"{framework}: {ref} no longer accepts {missing}; the documented wiring changed "
        f"(update the example, the adapter docs and compatibility.yaml). Accepts: {list(accepted)}"
    )


@pytest.mark.parametrize(("framework", "ref"), _CONTRACTS, ids=[f"{n}:{r}" for n, r in _CONTRACTS])
def test_framework_contract_symbol_exists(framework: str, ref: str) -> None:
    module_name, _, spec = ref.partition(":")
    entry = _ENTRY.match(spec)
    assert entry, (
        f"{framework}: malformed contract entry {ref!r} "
        "(expected module:Symbol or module:Symbol(a, b))"
    )
    symbol = entry["symbol"].strip()
    params = [p.strip() for p in (entry["params"] or "").split(",") if p.strip()]
    # Skip ONLY when the framework itself isn't installed (top-level package absent). If it IS
    # installed but our specific sub-module/symbol is gone, that's a genuine FAILURE, the version
    # is in our declared range yet the interface we bind to no longer exists. `google` is a shared
    # PEP-420 namespace (google-genai, google-adk, …), so its framework root is the first TWO
    # components (e.g. `google.adk`), not bare `google`.
    parts = module_name.split(".")
    root = ".".join(parts[:2]) if parts[0] in {"google"} else parts[0]
    pytest.importorskip(root, reason=f"{framework} not installed")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # installed, but our target module was removed/moved
        pytest.fail(
            f"{framework}: '{module_name}' is gone in the installed version, the adapter's "
            f"interface changed. Narrow the range in compatibility.yaml or update it. ({exc})"
        )
    obj: object = module
    for attr in symbol.split("."):
        assert hasattr(obj, attr), (
            f"{framework}: {module_name}.{symbol}, '{attr}' missing; the adapter's assumed "
            f"interface changed (update the adapter + compatibility.yaml)."
        )
        obj = getattr(obj, attr)
    if params:
        _assert_accepts(framework, ref, obj, params)


def test_at_least_one_contract_declared() -> None:
    # Guards against an empty/renamed manifest silently disabling every canary.
    assert _CONTRACTS, "no contract symbols in compatibility.yaml"


def test_every_framework_declares_a_contract() -> None:
    # A framework with no contract has no canary, so its compat-matrix job selects zero tests and
    # fails with pytest's "no tests collected" (exit 5) instead of saying what is wrong. Name the
    # gap here.
    empty = [name for name, fw in _MANIFEST["frameworks"].items() if not fw.get("contract")]
    assert not empty, f"compatibility.yaml frameworks with no contract symbols: {empty}"
