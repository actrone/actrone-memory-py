"""Contract tests for the LangGraph checkpointer adapter, run against the REAL LangGraph runtime.

Skipped automatically when langgraph is not installed (failed instead inside a compat-matrix job,
where it must import). The graph tests compile a real ``StateGraph`` with ``ActroneCheckpointer``
exactly as the docs show and run it through ``ainvoke``, so a checkpointer LangGraph rejects or
cannot resume from fails here, not for a user.
Run with: pip install actrone-memory[langgraph] && pytest tests/contract/test_langgraph.py
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from actrone_memory.config import MemoryConfig
from actrone_memory.integrations.langgraph import ActroneCheckpointer, _latest_exchange
from actrone_memory.manager import MemoryManager


class _ChatState(TypedDict):
    messages: Annotated[list[Any], add_messages]


def _reply(state: _ChatState) -> dict[str, Any]:
    """A deterministic stand-in for an LLM node: answers with how many user turns it has seen."""
    user_turns = sum(1 for m in state["messages"] if getattr(m, "type", "") == "human")
    return {"messages": [("ai", f"reply number {user_turns}")]}


def _chat_graph(checkpointer: Any) -> Any:
    builder = StateGraph(_ChatState)
    builder.add_node("reply", _reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_is_a_real_checkpoint_saver_accepted_by_compile() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    checkpointer = ActroneCheckpointer(agent_id="agent-1", memory_manager=mm)
    assert isinstance(checkpointer, BaseCheckpointSaver)
    assert isinstance(checkpointer, ActroneCheckpointer)
    _chat_graph(checkpointer)  # LangGraph rejects anything that is not a BaseCheckpointSaver
    await mm.close()


@pytest.mark.asyncio
async def test_graph_state_resumes_and_turns_are_recorded() -> None:
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    graph = _chat_graph(ActroneCheckpointer(agent_id="agent-1", memory_manager=mm))
    thread = {"configurable": {"thread_id": "t1"}}

    first = await graph.ainvoke(
        {"messages": [("user", "When is the nightly index rebuild?")]}, thread
    )
    assert first["messages"][-1].content == "reply number 1"

    # Same thread: the checkpoint is resumed, so the graph sees both user turns.
    second = await graph.ainvoke({"messages": [("user", "And the weekly backup?")]}, thread)
    assert second["messages"][-1].content == "reply number 2"
    assert len(second["messages"]) == 4

    # A different thread starts fresh.
    other = await graph.ainvoke(
        {"messages": [("user", "hello")]}, {"configurable": {"thread_id": "t2"}}
    )
    assert other["messages"][-1].content == "reply number 1"

    # Each completed exchange was recorded once in actrone-memory, per thread.
    context = await mm.retrieve_context("agent-1", "t1", "index rebuild backup", 2000)
    recorded = [(t.user_message, t.assistant_message) for t in context.recent_turns]
    assert ("When is the nightly index rebuild?", "reply number 1") in recorded
    assert ("And the weekly backup?", "reply number 2") in recorded
    assert len(recorded) == 2
    await mm.close()


@pytest.mark.asyncio
async def test_wraps_a_caller_supplied_saver() -> None:
    try:
        from langgraph.checkpoint.memory import InMemorySaver as Saver
    except ImportError:  # older LangGraph only has the MemorySaver name
        from langgraph.checkpoint.memory import MemorySaver as Saver
    inner = Saver()
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    graph = _chat_graph(ActroneCheckpointer(agent_id="agent-1", memory_manager=mm, saver=inner))
    await graph.ainvoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "t1"}})
    # Graph state went to the supplied saver.
    assert inner.get_tuple({"configurable": {"thread_id": "t1"}}) is not None
    await mm.close()


def test_rejects_a_saver_that_is_not_a_checkpoint_saver() -> None:
    with pytest.raises(TypeError, match="BaseCheckpointSaver"):
        ActroneCheckpointer(agent_id="agent-1", memory_manager=AsyncMock(), saver=object())


@pytest.mark.asyncio
async def test_build_context_threads_thread_id_and_renders() -> None:
    # Tier-1 framework-free path: thread_id is the session scope for recency.
    mock_mm = AsyncMock()
    mock_mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[MagicMock(user_message="hi", assistant_message="hello")],
            episodic_memories=[],
        )
    )
    cp = ActroneCheckpointer(agent_id="agent-1", memory_manager=mock_mm)
    context = await cp.build_context("recent conversation", thread_id="t1")
    assert "hi" in context and "hello" in context
    assert mock_mm.retrieve_context.call_args[0][1] == "t1"


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([("user", "q"), ("ai", "a")], ("q", "a")),
        ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], ("q", "a")),
        ([("user", "q")], None),  # no reply yet
        ([("ai", "a")], None),  # a reply with no question before it
        ([], None),
    ],
)
def test_latest_exchange(messages: list[Any], expected: tuple[str, str] | None) -> None:
    assert _latest_exchange(messages) == expected


def test_import_error_without_langgraph() -> None:
    modules = {
        "langgraph": None,
        "langgraph.checkpoint.base": None,
        "langgraph.checkpoint.memory": None,
    }
    with (
        patch.dict("sys.modules", modules),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[langgraph\]"),
    ):
        ActroneCheckpointer(agent_id="agent-1")
