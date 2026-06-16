"""
Example 02 — LangChain Adapter

Drop-in replacement for LangChain's ConversationBufferMemory.
Gives any LangChain chain or agent persistent two-tier memory
with zero changes to the chain itself.

Prerequisites:
    pip install "actrone-memory[langchain]" langchain-openai

Environment variables:
    ACTRONE_REDIS_URL
    ACTRONE_QDRANT_URL
    ACTRONE_OPENAI_API_KEY
    OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import os

from actrone_memory.integrations.langchain import ActroneMemory


async def main() -> None:
    # 1. Create the memory object — this is the only change vs. ConversationBufferMemory
    memory = ActroneMemory(
        agent_id="langchain-agent",
        session_id="user-session-42",
        token_budget=4096,
    )

    # 2. Use it with any LangChain chain
    try:
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )

        # Pass ActroneMemory exactly where you'd pass ConversationBufferMemory
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)

        questions = [
            "I'm designing a microservices architecture for a fintech startup.",
            "What database would you recommend for transaction records?",
            "How should I handle inter-service communication?",
        ]

        for question in questions:
            print(f"\nUser: {question}")
            # LangChain calls memory.load_memory_variables() before the LLM
            # and memory.save_context() after — actrone-memory handles both
            response = await chain.ainvoke({"input": question})
            print(f"Assistant: {response['response']}")

    except ImportError:
        # Demonstrate the memory API directly if LangChain isn't installed
        print("LangChain not installed — demonstrating memory API directly.\n")

        await memory.save_context(
            inputs={"input": "I'm building a fintech app."},
            outputs={"response": "That sounds interesting! What stack are you using?"},
        )

        variables = await memory.load_memory_variables({"input": "What was I building?"})
        print("Memory context injected into prompt:")
        print(variables["history"])


if __name__ == "__main__":
    asyncio.run(main())
