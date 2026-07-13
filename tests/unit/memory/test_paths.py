from __future__ import annotations

from pathlib import Path

import pytest

from tau3_retail_evolver.memory.paths import (
    evaluation_quarantine_root,
    project_root,
    training_memory_root,
)


def test_training_memory_is_stable_across_run_ids(tmp_path: Path) -> None:
    first_run_id = "iteration-0001"
    second_run_id = "iteration-0002"

    first = training_memory_root("retail", root=tmp_path)
    second = training_memory_root("retail", root=tmp_path)

    assert first_run_id != second_run_id
    assert first == second == tmp_path.resolve() / "history" / "agents" / "retail" / "memory"


def test_agent_namespaces_are_isolated(tmp_path: Path) -> None:
    retail = training_memory_root("retail", root=tmp_path)
    airline = training_memory_root("airline", root=tmp_path)

    assert retail != airline
    assert retail.parent.name == "retail"
    assert airline.parent.name == "airline"


def test_training_path_does_not_depend_on_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert training_memory_root("retail", root=tmp_path) == (
        tmp_path.resolve() / "history" / "agents" / "retail" / "memory"
    )


def test_default_project_root_does_not_depend_on_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(tmp_path)

    assert project_root() == expected


@pytest.mark.parametrize(
    "agent_id", ("", ".", "..", "RETAIL", "retail/other", r"retail\\other")
)
def test_path_resolver_rejects_unsafe_agent_id(tmp_path: Path, agent_id: str) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        training_memory_root(agent_id, root=tmp_path)


def test_uppercase_namespaces_fail_before_history_directory_creation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        training_memory_root("Retail", root=tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        evaluation_quarantine_root("EVAL-0001", "retail", root=tmp_path)

    assert not (tmp_path / "history").exists()


def test_streaming_evaluation_uses_quarantine(tmp_path: Path) -> None:
    assert evaluation_quarantine_root(
        "eval-0001", "retail", root=tmp_path
    ) == (
        tmp_path.resolve()
        / "history"
        / "evaluations"
        / "eval-0001"
        / "retail"
        / "quarantine"
    )
