"""LangGraph checkpointer adapter for actrone-memory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations._context import governed_context
from actrone_memory.manager import MemoryManager


class ActroneCheckpointer:
    """
    Drop-in LangGraph checkpointer. Requires: pip install actrone-memory[langgraph]

    Usage:
        from actrone_memory.integrations.langgraph import ActroneCheckpointer
        checkpointer = ActroneCheckpointer(agent_id="my-agent")
        graph = graph_builder.compile(checkpointer=checkpointer)
    """

    def __init__(
        self,
        agent_id: str,
        token_budget: int = 4096,
        config: MemoryConfig | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        try:
            from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install langgraph extras: pip install actrone-memory[langgraph]"
            ) from exc

        self.agent_id = agent_id
        self.token_budget = token_budget
        self._config = config
        self._mm: MemoryManager | None = memory_manager

    async def _get_manager(self) -> MemoryManager:
        if self._mm is None:
            self._mm = await MemoryManager.create(self._config)
        return self._mm

    async def build_context(
        self, query: str, *, thread_id: str = "default", token_budget: int | None = None
    ) -> str:
        """Governed system-context string for ``query`` (Tier 1, framework-free).

        The universally-correct path: prepend the returned block to a graph node's prompt
        regardless of framework. ``thread_id`` scopes recent-turn recency (LangGraph's session
        equivalent). Returns ``""`` when nothing is relevant. Complements the native
        checkpointer methods below.
        """
        mm = await self._get_manager()
        return await governed_context(
            mm, self.agent_id, thread_id, query, token_budget or self.token_budget
        )

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a checkpoint. Maps to store_turn in actrone-memory."""
        thread_id: str = config.get("configurable", {}).get("thread_id", "default")
        channel_values: dict[str, Any] = checkpoint.get("channel_values", {})

        user_msg = (
            str(channel_values.get("messages", [""])[-2])
            if len(channel_values.get("messages", [])) >= 2
            else ""
        )
        assistant_msg = (
            str(channel_values.get("messages", [""])[-1]) if channel_values.get("messages") else ""
        )

        if user_msg or assistant_msg:
            mm = await self._get_manager()
            await mm.store_turn(self.agent_id, thread_id, user_msg, assistant_msg)

        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": checkpoint.get("id", ""),
            },
        }

    async def aget(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Retrieve the latest checkpoint for a thread."""
        thread_id: str = config.get("configurable", {}).get("thread_id", "default")
        query = config.get("configurable", {}).get("query", "")

        mm = await self._get_manager()
        ctx = await mm.retrieve_context(
            self.agent_id, thread_id, query or "conversation context", self.token_budget
        )

        messages = []
        for turn in ctx.recent_turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})

        if not messages:
            return None

        return {
            "channel_values": {"messages": messages},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

    async def alist(self, config: dict[str, Any], **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """List checkpoints — yields the single current checkpoint."""
        checkpoint = await self.aget(config)
        if checkpoint:
            yield checkpoint
