# Contributing to actrone-memory

First off — thank you for wanting to help. Whether it's fixing a typo, improving a code example, or building a new integration, every contribution matters.

This document walks you through everything you need to get set up and submit a pull request.

---

## What You'll Need Before You Start

- **Python 3.11 or newer** — check with `python --version`
- **[uv](https://docs.astral.sh/uv/)** — the package manager we use
- **[Docker](https://docs.docker.com/get-docker/)** — needed to run Redis and Qdrant locally for integration tests

Install uv with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setting Up Your Development Environment

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/actrone-memory
cd actrone-memory

# 2. Create an isolated Python environment and install everything
uv venv .venv
uv pip install -e ".[dev,langchain,langgraph,crewai]"

# 3. Start Redis and Qdrant locally
docker run -d --name actrone-redis  -p 6379:6379 redis:7.2-alpine
docker run -d --name actrone-qdrant -p 6333:6333 qdrant/qdrant:v1.9.2

# 4. Set your environment variables
cp .env.example .env
# Then open .env and fill in your values — at minimum ACTRONE_OPENAI_API_KEY
```

You're ready.

---

## Running the Tests

### Unit tests — no infrastructure needed, run these constantly

```bash
pytest tests/unit/ -v
```

These test all the logic using mock objects, so you don't need Redis or Qdrant running. Run them every time you change something.

### Integration tests — requires running Redis + Qdrant

```bash
pytest tests/integration/ -v
```

These test the real connections to Redis and Qdrant. Run these before opening a pull request.

### Full suite with coverage report

```bash
pytest tests/ -v --cov=actrone_memory --cov-report=term-missing
```

**All pull requests must pass the unit suite and maintain ≥ 80% coverage.** CI will check this automatically and block the PR if it drops.

### Checking code style

```bash
ruff check src/ tests/      # find style issues
ruff format src/ tests/     # auto-fix formatting
mypy src/actrone_memory   # type check
```

All three must pass with zero errors before your PR will be reviewed.

---

## How the Codebase is Structured

If you're new to the project, here's where things live:

```text
src/actrone_memory/
│
├── manager.py          ← The main public interface. Start here.
├── config.py           ← All configuration settings.
├── models.py           ← Data types (MemoryEntry, Turn, etc.)
├── exceptions.py       ← Custom error classes.
│
├── l1/
│   └── redis_store.py  ← Short-term memory (Redis operations).
│
├── l2/
│   ├── qdrant_store.py ← Long-term memory (Qdrant operations).
│   └── embedder.py     ← Converts text to embedding vectors.
│
└── integrations/
    ├── langchain.py    ← LangChain adapter.
    ├── langgraph.py    ← LangGraph adapter.
    └── crewai.py       ← CrewAI adapter.

tests/
├── unit/               ← Logic tests using mocks (fast, no infrastructure).
└── integration/        ← Real Redis + Qdrant tests (slower).
```

---

## Code Style Rules

We keep things consistent so the codebase stays readable for everyone:

- **Type annotate everything.** Every function parameter and return value needs a type hint. `mypy` will catch missing ones.
- **Every async function that touches a network must be `async`.** No blocking calls inside async functions.
- **No mutable default arguments.** Use `None` and set the default inside the function instead.
- **No bare `except: pass`.** If you catch an exception, either handle it properly or re-raise it.
- **Line length is 100 characters.** `ruff format` handles this automatically.

---

## Submitting a Pull Request

```text
1. Create a branch from main
   git checkout -b feat/your-feature-name

2. Make your changes and write tests for them

3. Run the test suite and confirm it passes
   pytest tests/unit/ -v

4. Commit using Conventional Commits format:
   feat: add Haystack memory adapter
   fix: handle Redis timeout in append_turn
   docs: add example for multi-agent memory sharing
   test: add coverage for qdrant_store error paths

5. Push and open a pull request against main
```

**In your PR description, explain *why* the change is needed** — not just what you changed. "Fixes a bug" tells us nothing; "Redis was not releasing the pipeline object on timeout, causing connection leaks" tells us everything.

### What we look for in review

- Does this solve a real problem without adding unnecessary complexity?
- Are all new code paths covered by tests, including error cases?
- Does it follow the existing async patterns in `manager.py`?
- Are any new dependencies justified, stable, and pinned in `pyproject.toml`?

---

## Reporting a Bug

Open a [GitHub Issue](https://github.com/actrone/actrone-memory/issues) with:

- Your Python version (`python --version`) and OS
- A minimal code snippet that reproduces the problem
- What you expected to happen vs. what actually happened
- Any relevant error messages or logs (redact API keys and personal data before pasting)

For security vulnerabilities, **do not open a public issue** — see [SECURITY.md](SECURITY.md) for the private reporting process.

---

## Questions?

Open a [Discussion](https://github.com/actrone/actrone-memory/discussions) on GitHub — no question is too basic.
