"""Contract test for the Microsoft Agent Framework memory adapter (Tier 1 + Tier-2 import guard).

Tier 1 (``build_context`` / ``remember``) runs against a real local MemoryManager. The pure
message helpers (``_message_text`` / ``_role_value`` / ``_last_role_text``) are unit-tested with
duck-typed messages. The Tier-2 ``as_context_provider`` path lazily imports ``agent-framework``;
when absent (base venv) the guard must raise a clear ``ImportError``. When it IS installed (the
compat-matrix job), the provider is driven through the framework's real ``before_run`` /
``after_run`` hooks with real ``SessionContext`` / ``Message`` / ``AgentResponse`` objects.
"""

from __future__ import annotations

from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.microsoft_agent_framework import (
    ActroneAgentFrameworkMemory,
    _last_role_text,
    _message_text,
    _role_value,
)
from actrone_memory.manager import MemoryManager


class _Msg:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


def test_pure_message_helpers() -> None:
    assert _message_text(_Msg("user", "hi")) == "hi"
    assert _role_value(_Msg("USER", "hi")) == "user"
    msgs = [_Msg("user", "first"), _Msg("assistant", "reply"), _Msg("user", "second")]
    assert _last_role_text(msgs, "user") == "second"
    assert _last_role_text(msgs, "assistant") == "reply"
    assert _last_role_text(None, "user") == ""
    # Accepts a single message, not just a list.
    assert _last_role_text(_Msg("assistant", "solo"), "assistant") == "solo"


@pytest.mark.asyncio
async def test_tier1_context_and_remember() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneAgentFrameworkMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)

    await mm.inject_memory("support-bot", "The account owner is Jane Doe.", 0.9)
    assert "Jane" in await memory.build_context("who is the account owner")
    await memory.remember("q", "a")
    await mm.close()


def test_as_context_provider_requires_agent_framework() -> None:
    try:
        import agent_framework  # noqa: F401
    except ImportError:
        memory: Any = ActroneAgentFrameworkMemory(agent_id="a", session_id="s")
        with pytest.raises(ImportError, match="agent-framework"):
            memory.as_context_provider()
    else:  # pragma: no cover - only when agent-framework is installed
        pytest.skip("agent-framework installed; the native provider is tested below")


@pytest.mark.asyncio
async def test_native_context_provider_in_a_real_agent_run() -> None:
    """A real ``Agent.run`` with our provider: memory reaches the model, the exchange is stored."""
    af = pytest.importorskip("agent_framework", reason="agent-framework not installed")

    seen: dict[str, Any] = {}

    class _StubClient(af.BaseChatClient):  # type: ignore[misc]
        """Stands in for the LLM: records what the model is sent and replies with fixed text."""

        def _inner_get_response(
            self, *, messages: Any, stream: bool, options: Any, **_: Any
        ) -> Any:
            seen["messages"] = [str(getattr(m, "text", "") or "") for m in messages]
            seen["options"] = dict(options or {})

            async def _reply() -> Any:
                return af.ChatResponse(messages=[af.Message("assistant", ["Jane Doe owns it."])])

            return _reply()

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneAgentFrameworkMemory(agent_id="support-bot", session_id="s1", memory_manager=mm)
    await mm.inject_memory("support-bot", "The account owner is Jane Doe.", 0.9)

    provider = memory.as_context_provider()
    assert isinstance(provider, af.ContextProvider)
    agent = af.Agent(_StubClient(), context_providers=[provider])

    response = await agent.run("who is the account owner")
    assert "Jane Doe owns it" in response.text

    # before_run: the governed memory reached the model, as instructions or a system message.
    sent = " ".join(seen["messages"]) + " " + str(seen["options"].get("instructions", ""))
    assert "The account owner is Jane Doe" in sent

    # after_run: the exchange was stored and is recalled on a later turn.
    assert "Jane Doe owns it" in await memory.build_context("who owns the account")
    await mm.close()
