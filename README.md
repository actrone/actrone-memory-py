# 🧠 actrone-memory

> **Persistent memory for AI agents — so they never forget who you are.**

[![PyPI version](https://img.shields.io/pypi/v/actrone-memory?color=brightgreen&label=pypi)](https://pypi.org/project/actrone-memory/)
[![Python](https://img.shields.io/pypi/pyversions/actrone-memory)](https://pypi.org/project/actrone-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/actrone/actrone-memory)
[![CI](https://github.com/actrone/actrone-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/actrone/actrone-memory/actions)

---

## The Problem This Solves

By default, AI agents are **goldfish** 🐟 — they forget everything the moment a conversation ends (and even *during* a long conversation when the context window fills up).

This library gives your agent a **proper memory system**: a fast short-term memory for recent messages, and a long-term memory that stores and searches through everything the agent has ever learned.

```text
Without actrone-memory          With actrone-memory
─────────────────────────         ──────────────────────────────
User: "My name is Alex"           User: "My name is Alex"
AI:   "Hello Alex!"               AI:   "Hello Alex!"

[new session]                     [new session]

User: "What's my name?"           User: "What's my name?"
AI:   "I don't know your name."   AI:   "Your name is Alex!"  ✓
```

---

## Install

```bash
pip install actrone-memory
```

With framework adapters:

```bash
pip install "actrone-memory[langchain]"   # for LangChain users
pip install "actrone-memory[langgraph]"   # for LangGraph users
pip install "actrone-memory[crewai]"      # for CrewAI users
pip install "actrone-memory[local]"       # use local embeddings, no OpenAI needed
pip install "actrone-memory[all]"         # everything
```

**You'll need:** Python 3.11+, a running Redis instance, and a running Qdrant instance.
Start both instantly with Docker:

```bash
docker run -d -p 6379:6379 redis:7.2-alpine
docker run -d -p 6333:6333 qdrant/qdrant:v1.9.2
```

---

## Quickstart — 10 Lines

```python
import asyncio
from actrone_memory import MemoryManager

async def main():
    async with MemoryManager.create() as memory:
        # Save what the user said
        await memory.store_turn(
            agent_id   = "my-agent",
            session_id = "session-1",
            user_message      = "My name is Alex and I'm building a trading bot.",
            assistant_message = "Nice to meet you, Alex! What asset class are you targeting?",
        )

        # Before the next reply, fetch everything relevant
        context = await memory.retrieve_context(
            agent_id     = "my-agent",
            session_id   = "session-1",
            query        = "What should I watch out for?",
            token_budget = 4096,   # how many tokens you can spare for memory
        )

        print(f"Loaded {len(context.recent_turns)} recent turns")
        print(f"Loaded {len(context.episodic_memories)} long-term memories")

asyncio.run(main())
```

Set these environment variables first:

```bash
export ACTRONE_REDIS_URL=redis://localhost:6379
export ACTRONE_QDRANT_URL=http://localhost:6333
export ACTRONE_OPENAI_API_KEY=sk-...
```

---

## How the Memory System Works

Think of it like a human brain — there's a **working memory** for what just happened, and a **long-term memory** for everything else.

```text
┌─────────────────────────────────────────────────────────────────┐
│                        MemoryManager                            │
│                                                                 │
│  ┌──────────────────────────┐   ┌───────────────────────────┐  │
│  │     SHORT-TERM (Redis)   │   │    LONG-TERM (Qdrant)     │  │
│  │                          │   │                           │  │
│  │  The last 50 messages    │   │  Compressed summaries     │  │
│  │  of this conversation.   │   │  of older conversations.  │  │
│  │                          │   │                           │  │
│  │  Fast — under 1ms        │   │  Searched by meaning,     │  │
│  │  Expires after 24h       │   │  not by keyword.          │  │
│  │                          │   │  ~10ms. Never expires.    │  │
│  └──────────────────────────┘   └───────────────────────────┘  │
│                                                                 │
│  When you call retrieve_context(), both are searched in         │
│  parallel, then the most relevant pieces are selected to fit    │
│  inside your token budget — automatically.                      │
└─────────────────────────────────────────────────────────────────┘
```

### What happens when context gets too big?

The library **prioritises intelligently** — it never silently drops important things. It always keeps the most recent messages and the most relevant memories, pruning the least important stuff first.

```text
Your token budget: 4,096 tokens
├── System prompt          ████████░░░░░░░░░░░░  30%  (never pruned)
├── Long-term memories     ██████░░░░░░░░░░░░░░  25%  (lowest relevance dropped first)
├── Recent messages        █████████░░░░░░░░░░░  35%  (oldest dropped first)
└── Current user message   ██░░░░░░░░░░░░░░░░░░  10%  (always kept)
```

### Auto-summarisation

After every 20 messages (configurable), the library quietly compresses the conversation into a summary and saves it to long-term memory. This runs in the background — your users never wait for it.

```text
Message 1  ──┐
Message 2    │
...          │  After 20 messages →  [Summary written to Qdrant]
Message 20 ──┘                           ↑
                                    Available forever, searchable by meaning
```

---

## Framework Adapters

### LangChain — swap in 1 line

```python
# Before (built-in, forgets everything):
# from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory()

# After (persistent, searchable):
from actrone_memory.integrations.langchain import ActroneMemory
memory = ActroneMemory(agent_id="my-agent", session_id="user-123")

# The rest of your code stays exactly the same
chain = ConversationChain(llm=llm, memory=memory)
```

### LangGraph — plug-in checkpointer

```python
from actrone_memory.integrations.langgraph import ActroneCheckpointer

checkpointer = ActroneCheckpointer(agent_id="my-agent")
graph = graph_builder.compile(checkpointer=checkpointer)
# Your graph now remembers state across restarts and sessions
```

### CrewAI — shared crew memory

```python
from actrone_memory.integrations.crewai import ActroneCrewMemory
from crewai import Agent

# Each agent in the crew can remember and share findings
memory = ActroneCrewMemory(agent_id="research-crew", session_id="project-alpha")
researcher = Agent(role="Researcher", memory=True, memory_backend=memory)
```

Full working scripts in the [`examples/`](examples/) folder.

---

## Configuration

Everything is controlled with environment variables — no config files needed.

| Variable | Default | What it does |
| --- | --- | --- |
| `ACTRONE_REDIS_URL` | `redis://localhost:6379` | Where your Redis is running |
| `ACTRONE_QDRANT_URL` | `http://localhost:6333` | Where your Qdrant is running |
| `ACTRONE_OPENAI_API_KEY` | *(required)* | Your OpenAI key for generating memory embeddings |
| `ACTRONE_EMBEDDING_PROVIDER` | `openai` | Set to `local` to run without OpenAI |
| `ACTRONE_SESSION_TTL_HOURS` | `24` | How long short-term memory lasts |
| `ACTRONE_MAX_SESSION_TURNS` | `50` | Max messages kept in short-term memory |
| `ACTRONE_RELEVANCE_THRESHOLD` | `0.72` | How similar a memory must be before it's included (0–1) |
| `ACTRONE_AUTO_SUMMARISE` | `true` | Automatically compress old conversations to long-term memory |
| `ACTRONE_SUMMARISE_AFTER_TURNS` | `20` | How many messages before auto-summarisation kicks in |

---

## Documentation

| | |
| --- | --- |
| 📐 [Architecture Deep Dive](docs/architecture-deep-dive.md) | How the two-tier system works under the hood |
| 📖 [API Reference](docs/api-reference.md) | Every method, every parameter, every error |
| 🍳 [Examples / Cookbook](examples/) | Copy-paste scripts for OpenAI, LangChain, and CrewAI |
| 🤝 [Contributing](CONTRIBUTING.md) | How to set up the dev environment and submit a PR |
| 🔒 [Security Policy](SECURITY.md) | How to report a vulnerability privately |

---

## Part of Actrone

`actrone-memory` is the open-source memory layer powering [Actrone](https://actrone.com) — a full infrastructure platform for production AI agents.

When you're ready for **durable task execution**, **multi-model routing**, **tool supervision**, and **AI governance** on top of your memory layer, the hosted platform is one API key away.

---

## License

[MIT](LICENSE) — free to use in any project, commercial or otherwise.
