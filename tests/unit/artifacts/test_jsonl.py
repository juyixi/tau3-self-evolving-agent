from __future__ import annotations

from pathlib import Path

import pytest

from tau3_evolver.persistence.jsonl import JsonlWriter, iter_jsonl_objects
import tau3_evolver.persistence.jsonl as jsonl


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


def test_best_effort_syncs_the_parent_directory_after_an_append(
    monkeypatch, tmp_path: Path
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(jsonl, "fsync_directory", synced.append)

    JsonlWriter(tmp_path / "events.jsonl").append({"event_type": "EpisodeStarted"})

    assert synced == [tmp_path]


def test_iter_jsonl_objects_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok":true}\nnot-json\n', encoding="utf-8")

    iterator = iter_jsonl_objects(path)

    assert next(iterator) == {"ok": True}
    with pytest.raises(ValueError, match=r"events\.jsonl:2"):
        next(iterator)


def test_iter_jsonl_objects_skips_blank_lines_and_rejects_non_objects(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('\n[]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"JSON object.*events\.jsonl:2"):
        list(iter_jsonl_objects(path))
