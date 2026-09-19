"""
Example 02, LangChain Adapter

Gives any LangChain chain persistent two-tier memory. Uses
``ActroneChatMessageHistory``, which targets ``BaseChatMessageHistory``: the one memory
interface LangChain kept across both majors, so this example runs on 0.x and 1.x alike.

If you are pinned to LangChain 0.x and want the classic ``ConversationBufferMemory``
drop-in, use ``ActroneMemory`` from the same module instead. It implements ``BaseMemory``,
which LangChain removed in 1.x.

Prerequisites:
    pip install "actrone-memory[langchain]" langchain-openai

Environment variables (only needed for the durable backend / real LLM):
    ACTRONE_BACKEND=redis_qdrant
    ACTRONE_REDIS_URL
    ACTRONE_QDRANT_URL
    ACTRONE_OPENAI_API_KEY
    OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import os

from actrone_memory.integrations.langchain import ActroneChatMessageHistory


async def main() -> None:
    # 1. Governed history, this is the only change vs. an in-memory chat history
    history = ActroneChatMessageHistory(
        agent_id="langchain-agent",
        session_id="user-session-42",
        token_budget=4096,
    )

    # 2. Use it with any LangChain runnable
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful architecture advisor."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        # The history object is async-only, so drive the chain with ainvoke.
        chain = RunnableWithMessageHistory(
            prompt | llm,
            lambda _session_id: history,  # type: ignore[arg-type]  # structural, not a subclass
            input_messages_key="input",
            history_messages_key="history",
        )

        questions = [
            "I'm designing a microservices architecture for a fintech startup.",
            "What database would you recommend for transaction records?",
            "How should I handle inter-service communication?",
        ]

        for question in questions:
            print(f"\nUser: {question}")
            # LangChain reads history before the LLM call and writes the turn after;
            # actrone-memory handles both, with budget-aware recall on the read.
            response = await chain.ainvoke(
                {"input": question},
                config={"configurable": {"session_id": "user-session-42"}},
            )
            print(f"Assistant: {response.content}")

    except ImportError:
        # Demonstrate the memory API directly if LangChain isn't installed
        print("LangChain not installed, demonstrating the history API directly.\n")

        from langchain_core.messages import AIMessage, HumanMessage

        await history.aadd_messages(
            [
                HumanMessage(content="I'm building a fintech app."),
                AIMessage(content="That sounds interesting! What stack are you using?"),
            ]
        )

        for message in await history.aget_messages():
            print(f"{message.type}: {message.content}")

        # Tier 1: the same memory as a framework-free system-context block.
        print("\nGoverned context for the next prompt:")
        print(await history.build_context("What was I building?"))


if __name__ == "__main__":
    asyncio.run(main())
