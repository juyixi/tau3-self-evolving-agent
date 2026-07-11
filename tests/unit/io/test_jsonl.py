from __future__ import annotations

from pathlib import Path

from tau3_retail_evolver.io.jsonl import JsonlWriter


def test_appends_one_canonical_json_object_per_durable_line(tmp_path: Path) -> None:
    path = tmp_path / "rollouts" / "events.jsonl"
    writer = JsonlWriter(path)

    writer.append({"z": 1, "a": {"b": True, "a": None}})
    JsonlWriter(path).append({"event_type": "EpisodeFinished", "reward": 1.0})

    assert path.read_text(encoding="utf-8") == (
        '{"a":{"a":null,"b":true},"z":1}\n'
        '{"event_type":"EpisodeFinished","reward":1.0}\n'
    )


def test_refuses_non_json_safe_event_values(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path / "events.jsonl")

    try:
        writer.append({"not_json": object()})
    except TypeError as error:
        assert "JSON serializable" in str(error)
    else:
        raise AssertionError("expected JSON-safe validation failure")
