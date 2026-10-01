#!/usr/bin/env bash
# compat-matrix runner (vendored, per-repo). Install THIS package (the repo root = the CWD) + ONE
# framework at a pinned version boundary in a clean env, then run that framework's interface canary
# against the REAL SDK.
#
#   run.sh <canary-path> <pip-spec> <framework> <which>
#
# If ACTRONE_MEMORY_SRC is set (a path to a checked-out actrone-memory-py), it is installed FIRST, the
# actrone-py workflow sets this because actrone-py depends on actrone-memory (plain pip can't resolve
# the pyproject's uv source). It is unset for actrone-memory-py, which has no such dependency.
#
# Runs from a clean env so frameworks never share a dependency graph. Exit code = the test result;
# `next` jobs are allow_fail in the matrix so a pre-release break only warns, and a `next` with no
# release beyond the cap yet exits 0 with a notice.
#
# Vendored from the workspace ci/compat-matrix/ (per-repo split). Keep the two Python repos' copies in sync.
set -euo pipefail

CANARY="${1:?canary path}"
SPEC="${2:?pip spec}"
FRAMEWORK="${3:?framework}"
WHICH="${4:-current}"

PRE=""
if [ "$WHICH" = "next" ]; then PRE="--pre"; fi

echo "== compat-matrix: $FRAMEWORK @ $WHICH ($SPEC) =="
python -m pip install --quiet --upgrade pip

# Cross-repo dependency: actrone-py needs actrone-memory. Its workflow checks the memory repo out and
# points ACTRONE_MEMORY_SRC at it; install that first so the package's requirement is satisfied locally.
if [ -n "${ACTRONE_MEMORY_SRC:-}" ]; then
  echo "== pre-dep: actrone-memory from ${ACTRONE_MEMORY_SRC} =="
  python -m pip install --quiet "${ACTRONE_MEMORY_SRC}"
fi

# THIS package (`.` = repo root = CWD), the framework at the pinned spec and the test tooling, in ONE
# resolver pass, as a user's `pip install actrone-memory[<extra>]` would be. Installing the framework in a
# second step let it silently downgrade a shared dependency (dspy-ai 2.4 pulled pydantic below what
# actrone-memory needs), so the job then failed at import with a misleading error; now an impossible
# combination fails here, at install, with the resolver's explanation.
#
# A `next` job targets the first release beyond the supported cap, which often does not exist yet. That
# is not a failure: report it and stop. Any other install error (a real resolver conflict) still fails.
install_log="$(mktemp)"
# shellcheck disable=SC2086
if ! python -m pip install --quiet $PRE . "$SPEC" pytest pytest-asyncio pyyaml >"$install_log" 2>&1; then
  cat "$install_log"
  if [ "$WHICH" = "next" ] && grep -qE "No matching distribution found|Could not find a version that satisfies" "$install_log"; then
    echo "::notice title=compat-matrix ${FRAMEWORK}@next::no release matches ${SPEC} yet; nothing to test"
    exit 0
  fi
  exit 1
fi

echo "== installed =="
python -m pip show "${SPEC%%[><=!~ ]*}" 2>/dev/null | grep -E '^(Name|Version):' || true

# Run this framework's interface canary AND its own contract tests, so the adapter's native (Tier 2)
# path executes against the REAL SDK at this version, not just a symbol check. COMPAT_MATRIX=1 un-gates
# the version-pinned canary module (skipped everywhere else). The -k keeps only this framework's canary
# ids while selecting every test in the contract file. -o addopts="" drops the package's default --cov
# flags (pytest-cov is not in this minimal env); -p no:cacheprovider so a read-only mount doesn't fail on
# .pytest_cache.
TARGETS=("$CANARY")
CONTRACT_TESTS="tests/contract/test_${FRAMEWORK}.py"
if [ -f "$CONTRACT_TESTS" ]; then TARGETS+=("$CONTRACT_TESTS"); fi
# COMPAT_MATRIX_REQUIRE_FRAMEWORK=1 turns "<framework> not installed" skips into failures (tests/conftest.py):
# the framework IS installed here, so such a skip means it failed to import.
COMPAT_MATRIX=1 COMPAT_MATRIX_REQUIRE_FRAMEWORK=1 python -m pytest "${TARGETS[@]}" -q -rs -o addopts="" -p no:cacheprovider \
  -k "$FRAMEWORK or not test_framework_contract_symbol_exists"

# Reached only when the tests passed (set -e). On a `next` job that means a release above the cap works
# but the extra's range still excludes it, so say so: a silent pass lets the cap fall behind users.
if [ "$WHICH" = "next" ]; then
  PRIMARY="${SPEC%%[><=!~ ]*}"
  VERSION="$(python -m pip show "$PRIMARY" 2>/dev/null | sed -n 's/^Version: //p')"
  echo "::warning title=compat-matrix ${FRAMEWORK}@next::${PRIMARY} ${VERSION} passes but is outside the supported range. Widen the cap in compatibility.yaml."
fi
