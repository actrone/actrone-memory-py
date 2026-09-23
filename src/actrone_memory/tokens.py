"""Token counting for the context budget, kept in step with the TypeScript library.

The default is a dependency-free heuristic, about 4 characters per token (the well-known rough GPT
ratio), identical to the TypeScript ``heuristicTokenCounter``, so both libraries allocate a budget
the same way and neither touches the network. Exact ``cl100k_base`` counts are opt-in through
``MemoryConfig(token_counter="tiktoken")`` and the ``[tiktoken]`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

from actrone_memory.exceptions import ConfigurationError

TokenCounter = Callable[[str], int]
"""Returns how many tokens a text consumes."""


def heuristic_token_counter(text: str) -> int:
    """Estimate tokens as ``ceil(len(text) / 4)``.

    Never returns less than 1 for non-empty text, so a turn or memory always consumes some budget.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def tiktoken_token_counter() -> TokenCounter:
    """Return an exact ``cl100k_base`` counter.

    The encoding is loaded here, once, rather than on every call. tiktoken downloads the encoding
    file on first use unless ``TIKTOKEN_CACHE_DIR`` already holds it, so this is the one opt-in path
    in the library that may reach the network.

    Raises:
        ConfigurationError: If tiktoken is not installed or the encoding cannot be loaded.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise ConfigurationError(
            'token_counter="tiktoken" needs the tiktoken extra: '
            'pip install "actrone-memory[tiktoken]"',
            details={"token_counter": "tiktoken"},
        ) from exc
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # network or cache failure while fetching the encoding
        raise ConfigurationError(
            "Could not load the cl100k_base encoding for tiktoken. It is downloaded once on first "
            "use; check network access, or set TIKTOKEN_CACHE_DIR to a directory that holds it.",
            details={"token_counter": "tiktoken", "error": str(exc)},
        ) from exc

    def count(text: str) -> int:
        return len(encoding.encode(text)) if text else 0

    return count


def build_token_counter(kind: Literal["heuristic", "tiktoken"]) -> TokenCounter:
    """Return the counter ``MemoryConfig.token_counter`` names.

    Raises:
        ConfigurationError: If ``kind`` is "tiktoken" and tiktoken cannot be loaded.
    """
    if kind == "tiktoken":
        return tiktoken_token_counter()
    return heuristic_token_counter
