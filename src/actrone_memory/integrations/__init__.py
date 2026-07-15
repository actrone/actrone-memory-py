"""Framework integration adapters for actrone-memory.

Each adapter is optional — the corresponding framework must be installed first.
Use the package extras to install the required dependencies::

    pip install actrone-memory[langchain]    # LangChain
    pip install actrone-memory[langgraph]    # LangGraph
    pip install actrone-memory[crewai]       # CrewAI
    pip install actrone-memory[autogen]      # AutoGen 0.4
    pip install actrone-memory[llamaindex]   # LlamaIndex
    pip install actrone-memory[haystack]     # Haystack v2
    pip install actrone-memory[dspy]         # DSPy
    pip install actrone-memory[openai_agents]              # OpenAI Agents SDK
    pip install actrone-memory[pydantic_ai]               # Pydantic AI
    pip install actrone-memory[claude_agent_sdk]          # Claude Agent SDK
    pip install actrone-memory[semantic_kernel]           # Semantic Kernel
    pip install actrone-memory[google_adk]                # Google ADK
    pip install actrone-memory[microsoft_agent_framework] # Microsoft Agent Framework
    pip install actrone-memory[all]          # All frameworks

Two tiers (see ``_context.py``):

- **Tier 1 — governed system-context string.** The universally-correct, framework-free path.
  *Every* adapter exposes a ``build_context(query) -> str`` coroutine (conversational adapters
  render recent turns + episodic memory; the retrieval-shaped DSPy/Haystack adapters render the
  ranked memories) — prepend the returned block to any prompt regardless of framework. The
  Tier-1-only adapters (openai_agents, pydantic_ai, claude_agent_sdk) import no framework at all.
- **Tier 2 — native conformance.** Where a framework has a formal memory contract, the adapter
  also implements it (LangChain ``BaseMemory``, AutoGen ``Memory``, Semantic Kernel
  ``ChatHistory``, Google ADK ``BaseMemoryService``, Microsoft Agent Framework
  ``ContextProvider``, …). Those imports are lazy — pulled only when the native method runs.
"""
