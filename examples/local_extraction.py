"""Fact extraction on a local model (actrone.com/docs/memory/overview, "Fact extraction").

CI type-checks this file with ``mypy --strict``, and ``tests/unit/test_local_extraction_example.py``
runs it against a stand-in OpenAI-compatible server and checks what it stores and sends. The
``# region`` block is what the page shows. The live demo suite runs the same wiring on Ollama.
"""

# region memory-py-local-extraction
from actrone_memory import MemoryManager, OpenAIFactExtractor


async def memory_with_local_extraction(
    base_url: str = "http://localhost:11434/v1",
) -> MemoryManager:
    """Extract facts with a model on any OpenAI-compatible server."""
    extractor = OpenAIFactExtractor(
        base_url=base_url,
        api_key="ollama",  # Ollama ignores the key; the client needs one
        model="qwen2.5:3b",
    )
    return await MemoryManager.create(extractor=extractor)


async def learn_from_session(
    memory: MemoryManager, agent_id: str, session_id: str
) -> list[str]:
    """Turn the session's recent turns into stored facts."""
    return await memory.extract_memories(agent_id, session_id)


# endregion memory-py-local-extraction
