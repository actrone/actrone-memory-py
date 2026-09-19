"""Contract test for the Semantic Kernel memory adapter (Tier 1 + Tier 2).

Framework-free: Tier 1 runs against a real local MemoryManager; the Tier-2
``add_to_chat_history`` path is duck-typed, so it is exercised with a fake ``ChatHistory`` (a
stand-in exposing ``add_system_message``), no ``semantic-kernel`` install required.
"""

from __future__ import annotations

from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.semantic_kernel import ActroneSemanticKernelMemory
from actrone_memory.manager import MemoryManager


class _FakeChatHistory:
    """Minimal stand-in for ``semantic_kernel.contents.ChatHistory`` (system messages only)."""

    def __init__(self) -> None:
        self.system_messages: list[str] = []

    def add_system_message(self, content: str) -> Any:
        self.system_messages.append(content)


@pytest.mark.asyncio
async def test_system_message_and_add_to_chat_history() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneSemanticKernelMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    # Tier 1, empty until something relevant exists.
    assert await memory.system_message("nothing") == ""

    await mm.inject_memory("support-bot", "The SLA for tickets is 24 hours.", 0.9)
    assert "SLA" in await memory.system_message("ticket sla")

    # Tier 2, populate a real (faked) ChatHistory in place.
    history = _FakeChatHistory()
    added = await memory.add_to_chat_history(history, "ticket sla")
    assert added is True
    assert len(history.system_messages) == 1
    assert "SLA" in history.system_messages[0]

    # Nothing relevant ⇒ no message added, returns False.
    empty = _FakeChatHistory()
    assert await memory.add_to_chat_history(empty, "totally unrelated xyzzy") is False
    assert empty.system_messages == []

    await memory.remember("q", "a")
    await mm.close()
