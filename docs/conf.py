"""Sphinx configuration for the Actrone Memory (Python) API reference.

Generated from the ``actrone_memory`` package source (docstrings + type hints) — the OSS memory
library's single source of truth (Public-Domain Cutover Runbook Phase 6, item 4). Generation is a
build/CI step; the output is never committed.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "Actrone Memory (Python)"
author = "Actrone"
copyright = f"{datetime.now().year}, Actrone"  # noqa: A001

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Benign ambiguous cross-references from the library's top-level re-exports (see actrone-py/docs/conf.py).
suppress_warnings = ["ref.python"]

html_theme = "furo"
html_title = "Actrone Memory (Python)"
# The reference is served from actrone.com/reference/memory-py/ (vendored by the marketing site's
# build-sdk-reference script). Setting the base URL makes Sphinx emit a canonical link on every page,
# so search engines index the reference under its public URL rather than guessing.
html_baseurl = "https://actrone.com/reference/memory-py/"
