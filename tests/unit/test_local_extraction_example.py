"""Runs ``examples/local_extraction.py`` (what the docs page shows) against a stand-in
OpenAI-compatible server, the API Ollama, vLLM and LM Studio serve, and asserts what the page
claims: a session's turns become stored facts, extracted with the shared spec's request.

The keyword embedder is pinned through the environment, so the run is deterministic and needs
no model download, while the example itself stays exactly as a user would write it.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from actrone_memory.extraction import EXTRACTION_RESPONSE_SCHEMA, format_extraction_input

pytest.importorskip("openai")

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "local_extraction.py"
_FACT = "The escalation codeword for the payments team is PELICAN-42."


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("example_local_extraction", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load()


def _completion(content: str) -> bytes:
    message = {"role": "assistant", "content": content}
    choice = {"index": 0, "finish_reason": "stop", "message": message}
    completion = {"id": "c1", "object": "chat.completion", "created": 0, "model": "m"}
    return json.dumps({**completion, "choices": [choice]}).encode()


class _ChatCompletions(BaseHTTPRequestHandler):
    """Answers every POST like an OpenAI-compatible chat-completions endpoint."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        _ChatCompletions.requests.append({"path": self.path, "body": body})
        fact = {"content": _FACT, "sensitivity": "none", "topic_tags": [], "importance": 0.8}
        reply = _completion(json.dumps({"facts": [fact]}))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *_args: Any) -> None:  # keep the test output clean
        return


@pytest.fixture
def server_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("ACTRONE_EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("ACTRONE_AUTO_SUMMARISE", "false")
    _ChatCompletions.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletions)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_a_session_becomes_stored_facts_through_a_local_server(server_url: str) -> None:
    user = "Please remember: the escalation codeword for the payments team is PELICAN-42."
    memory = await example.memory_with_local_extraction(base_url=server_url)
    try:
        await memory.store_turn("support-bot", "s1", user, "Noted.")

        stored = await example.learn_from_session(memory, "support-bot", "s1")

        assert len(stored) == 1
        found = await memory.search_memories("support-bot", "payments escalation codeword")
        assert any(m.content == _FACT and m.source == "extracted" for m in found)
    finally:
        await memory.close()

    # One request, to the configured server, carrying the shared extraction spec.
    [request] = _ChatCompletions.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["model"] == "qwen2.5:3b"
    framed = format_extraction_input(f"User: {user}\nAssistant: Noted.")
    assert request["body"]["messages"][1]["content"] == framed
    assert request["body"]["response_format"]["json_schema"]["schema"] == EXTRACTION_RESPONSE_SCHEMA
