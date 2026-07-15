#!/usr/bin/env bash
# compat-matrix runner (vendored, per-repo). Install THIS package (the repo root = the CWD) + ONE
# framework at a pinned version boundary in a clean env, then run that framework's interface canary
# against the REAL SDK.
#
#   run.sh <canary-path> <pip-spec> <framework> <which>
#
# If ACTRONE_MEMORY_SRC is set (a path to a checked-out actrone-memory-py), it is installed FIRST — the
# actrone-py workflow sets this because actrone-py depends on actrone-memory (plain pip can't resolve
# the pyproject's uv source). It is unset for actrone-memory-py, which has no such dependency.
#
# Runs from a clean env so frameworks never share a dependency graph. Exit code = the canary result;
# `next` jobs are allow_fail in the matrix so a pre-release break only warns.
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

# THIS package, installed by path (`.` = repo root = CWD), plus the test tooling.
python -m pip install --quiet . pytest pytest-asyncio pyyaml
# shellcheck disable=SC2086
python -m pip install --quiet $PRE "$SPEC"

echo "== installed =="
python -m pip show "${SPEC%%[><=!~ ]*}" 2>/dev/null | grep -E '^(Name|Version):' || true

# Run only this framework's canary. COMPAT_MATRIX=1 un-gates the version-pinned canary module (skipped
# everywhere else). -o addopts="" drops the package's default --cov flags (pytest-cov not in this minimal
# env); -p no:cacheprovider so a read-only mount doesn't fail on .pytest_cache.
COMPAT_MATRIX=1 python -m pytest "$CANARY" -k "$FRAMEWORK" -q -rs -o addopts="" -p no:cacheprovider
