"""An Actrone memory export loads into this library.

Actrone exports a person's memory in the open actrone-memory format: a document whose
``memories`` are this library's ``MemoryEntry`` (same fields, same four sensitivity
tags). This sample is what the hosted export writes; if a field or a tag changes on
either side, this test fails.
"""

from __future__ import annotations

import json

from actrone_memory.models import MemoryEntry

EXPORT = """
{
  "format": "actrone-memory",
  "version": 1,
  "exported_at": "2026-11-16T10:49:00Z",
  "person": "user_1",
  "memories": [
    {
      "id": "8d9c3d7e-58a8-4f43-a7a7-1f0f0b6d2a11",
      "agent_id": "0a0a0a0a-0000-0000-0000-000000000002",
      "session_id": "",
      "content": "owner diet vegetarian",
      "content_type": "fact",
      "importance_score": 1,
      "topic_tags": ["health"],
      "timestamp": "2026-11-01T09:00:00Z",
      "source_turn_ids": ["task-1"],
      "source": "user",
      "sensitivity": "low"
    },
    {
      "id": "1f2e3d4c-5b6a-4978-8a9b-0c1d2e3f4a5b",
      "agent_id": "0a0a0a0a-0000-0000-0000-000000000002",
      "session_id": "",
      "content": "owner seat aisle",
      "content_type": "fact",
      "importance_score": 0.7,
      "topic_tags": [],
      "timestamp": "2026-11-01T09:00:00Z",
      "source_turn_ids": [],
      "source": "extracted",
      "sensitivity": "none"
    }
  ]
}
"""


def test_an_actrone_export_loads_as_memory_entries() -> None:
    doc = json.loads(EXPORT)
    assert doc["format"] == "actrone-memory"
    assert doc["version"] == 1
    entries = [MemoryEntry.model_validate(m) for m in doc["memories"]]
    assert [e.content for e in entries] == ["owner diet vegetarian", "owner seat aisle"]
    assert entries[0].source == "user"
    assert entries[0].sensitivity == "low"
    assert entries[0].topic_tags == ["health"]
    assert entries[1].source == "extracted"
    assert all(e.content_type == "fact" for e in entries)
