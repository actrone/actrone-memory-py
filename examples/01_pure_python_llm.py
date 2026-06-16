"""
Example 01 — Pure Python with OpenAI

Demonstrates using actrone-memory directly with the OpenAI client,
without any framework (LangChain, LangGraph, CrewAI, etc.).

Prerequisites:
    pip install actrone-memory openai

Environment variables:
    ACTRONE_REDIS_URL    (default: redis://localhost:6379)
    ACTRONE_QDRANT_URL   (default: http://localhost:6333)
    ACTRONE_OPENAI_API_KEY
    OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from actrone_memory import MemoryConfig, MemoryManager

AGENT_ID = "assistant-v1"
SESSION_ID = "demo-session-001"
TOKEN_BUDGET = 4096


async def chat(mm: MemoryManager, client: AsyncOpenAI, user_message: str) -> str:
    # Retrieve context: recent turns + semantically relevant episodic memories
    ctx = await mm.retrieve_context(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        query=user_message,
        token_budget=TOKEN_BUDGET,
    )

    # Build the message list for the LLM
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant with persistent memory."},
    ]

    # Inject relevant episodic memories as system context
    if ctx.episodic_memories:
        memory_block = "\n".join(f"- {m.content}" for m in ctx.episodic_memories)
        messages.append({
            "role": "system",
            "content": f"Relevant memories from previous conversations:\n{memory_block}",
        })

    # Inject recent session turns
    for turn in ctx.recent_turns:
        messages.append({"role": "user", "content": turn.user_message})
        messages.append({"role": "assistant", "content": turn.assistant_message})

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    # Call the LLM
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,  # type: ignore[arg-type]
        max_tokens=512,
    )
    assistant_message = response.choices[0].message.content or ""

    # Persist the turn for future sessions
    await mm.store_turn(
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        user_message=user_message,
        assistant_message=assistant_message,
    )

    print(f"\nBudget used: {ctx.total_tokens_used}/{ctx.token_budget} tokens "
          f"({ctx.budget_utilisation:.0%}) | "
          f"Turns: {len(ctx.recent_turns)} | Memories: {len(ctx.episodic_memories)}")

    return assistant_message


async def main() -> None:
    config = MemoryConfig(
        openai_api_key=os.environ["ACTRONE_OPENAI_API_KEY"],  # type: ignore[arg-type]
        session_ttl_hours=24,
        max_session_turns=20,
        auto_summarise=True,
        summarise_after_turns=10,
    )

    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async with MemoryManager.create(config) as mm:
        # Simulate a multi-turn conversation
        turns = [
            "My name is Alex and I'm building a trading bot.",
            "What are the main risks I should consider?",
            "How should I handle market hours and weekends?",
            # Come back later — memory persists across sessions
            "Remind me, what project was I working on?",
        ]

        for user_msg in turns:
            print(f"\nUser: {user_msg}")
            response = await chat(mm, openai_client, user_msg)
            print(f"Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())
