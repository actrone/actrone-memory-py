"""
Example 03, CrewAI Multi-Agent with Shared Memory

Each agent in the crew gets its own persistent memory namespace.
Research findings written by the Researcher are retrievable by the
Analyst in future sessions, enabling true cross-turn, cross-agent recall.

Prerequisites:
    pip install "actrone-memory[crewai]" crewai

Environment variables:
    ACTRONE_REDIS_URL
    ACTRONE_QDRANT_URL
    ACTRONE_OPENAI_API_KEY
    OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import os

from actrone_memory.integrations.crewai import ActroneCrewMemory


async def demo_memory_api() -> None:
    """
    Demonstrates the memory contract the CrewAI backend uses,
    runnable without CrewAI installed.
    """
    # Researcher agent memory, saves findings
    researcher_memory = ActroneCrewMemory(
        agent_id="research-crew",
        session_id="analyst-session",
    )

    # Simulate the researcher saving a finding
    await researcher_memory.save(
        value="AAPL Q4 2025 revenue was $124.3B, up 4% YoY. "
              "Services segment grew 14% to $26.3B.",
        metadata={"task_input": "Analyse AAPL Q4 2025 earnings"},
    )

    await researcher_memory.save(
        value="Key risk: iPhone unit sales declined 2% in China due to Huawei competition.",
        metadata={"task_input": "Analyse AAPL Q4 2025 earnings"},
    )

    print("Researcher saved 2 findings to memory.\n")

    # Analyst agent memory, searches the same namespace
    analyst_memory = ActroneCrewMemory(
        agent_id="research-crew",   # same crew namespace
        session_id="analyst-session",
    )

    results = await analyst_memory.search("AAPL revenue growth", limit=5)
    print(f"Analyst retrieved {len(results)} relevant memories:\n")
    for r in results:
        print(f"  [{r['metadata']['content_type']}] {r['content']}")
        print(f"  Score: {r['score']:.2f}\n")


async def demo_with_crewai() -> None:
    """Full CrewAI integration, only runs if crewai is installed."""
    try:
        from crewai import Agent, Crew, Task
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("crewai or langchain-openai not installed.")
        print("Running memory API demo instead.\n")
        await demo_memory_api()
        return

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

    # Each agent gets its own ActroneCrewMemory with a shared crew namespace
    researcher_memory = ActroneCrewMemory(agent_id="equity-research-crew", session_id="q4-2025")
    analyst_memory = ActroneCrewMemory(agent_id="equity-research-crew", session_id="q4-2025")

    researcher = Agent(
        role="Equity Researcher",
        goal="Find and summarise key financial metrics from earnings reports",
        backstory="You are a precise financial researcher who extracts facts from reports.",
        llm=llm,
        memory=True,
        memory_backend=researcher_memory,
    )

    analyst = Agent(
        role="Investment Analyst",
        goal="Synthesise research findings into a clear investment thesis",
        backstory="You turn raw financial data into actionable investment recommendations.",
        llm=llm,
        memory=True,
        memory_backend=analyst_memory,
    )

    research_task = Task(
        description="Analyse Apple's Q4 2025 earnings report and extract key metrics.",
        agent=researcher,
        expected_output="Bullet-point summary of revenue, growth rates, and risks.",
    )

    analysis_task = Task(
        description=(
            "Based on the research findings, produce a buy/hold/sell recommendation for AAPL."
        ),
        agent=analyst,
        expected_output="Investment thesis with recommendation and supporting rationale.",
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        verbose=True,
    )

    result = crew.kickoff()
    print("\nCrew result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(demo_with_crewai())
