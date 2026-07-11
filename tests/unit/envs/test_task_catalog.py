from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import tau3_retail_evolver.envs.runtime as runtime_module
from tau3_retail_evolver.envs.runtime import RuntimeFingerprint, Tau2Runtime
from tau3_retail_evolver.envs.split_guard import require_learning_split
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog


SPLIT_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "tau2_retail" / "split_tasks.json"
)


def _write_tasks(path: Path, task_ids: set[str]) -> None:
    path.write_text(
        json.dumps([{"id": task_id} for task_id in sorted(task_ids, key=int)]),
        encoding="utf-8",
    )


def test_official_retail_splits_are_complete_disjoint_and_stably_fingerprinted(
    tmp_path: Path,
) -> None:
    split_data = json.loads(SPLIT_FIXTURE.read_text(encoding="utf-8"))
    tasks_path = tmp_path / "tasks.json"
    _write_tasks(tasks_path, set(split_data["base"]))

    catalog = RetailTaskCatalog.from_files(tasks_path, SPLIT_FIXTURE)

    train_ids = catalog.task_ids("train")
    test_ids = catalog.task_ids("test")
    base_ids = catalog.task_ids("base")
    expected_fingerprint = hashlib.sha256(
        json.dumps(split_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert len(train_ids) == 74
    assert len(test_ids) == 40
    assert len(base_ids) == 114
    assert set(train_ids).isdisjoint(test_ids)
    assert set(base_ids) == set(train_ids) | set(test_ids)
    assert catalog.split_sha256 == expected_fingerprint


@pytest.mark.parametrize(
    ("pin_contents", "message"),
    ((None, "missing"), ("not-a-commit", "malformed"), ("b" * 40, "mismatch")),
)
def test_runtime_rejects_missing_malformed_or_mismatched_checkout_pin(
    tmp_path: Path, pin_contents: str | None, message: str
) -> None:
    checkout = tmp_path / "external" / "tau2-bench"
    checkout.mkdir(parents=True)
    pin_path = checkout.with_suffix(".commit")
    if pin_contents is not None:
        pin_path.write_text(pin_contents + "\n", encoding="utf-8")
    fingerprint = RuntimeFingerprint(
        repo_path=checkout,
        git_commit="a" * 40,
        package_version="1.0.0",
        retail_tasks_path=Path("tasks.json"),
        retail_split_path=Path("split_tasks.json"),
        gym_available=True,
    )

    with pytest.raises(RuntimeError, match=message):
        Tau2Runtime.require_pinned_commit(fingerprint)


def test_official_compatibility_rejects_split_count_mismatch(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    split_path = tmp_path / "split_tasks.json"
    _write_tasks(tasks_path, {"0"})
    split_path.write_text(
        json.dumps({"train": ["0"], "test": [], "base": ["0"]}),
        encoding="utf-8",
    )
    catalog = RetailTaskCatalog.from_files(tasks_path, split_path)

    with pytest.raises(RuntimeError, match="train=74.*test=40.*base=114"):
        catalog.require_official_compatibility()


def test_official_compatibility_rejects_same_size_split_hash_edit(tmp_path: Path) -> None:
    split_data = json.loads(SPLIT_FIXTURE.read_text(encoding="utf-8"))
    split_data["train"] = list(reversed(split_data["train"]))
    split_data["base"] = split_data["train"] + split_data["test"]
    tasks_path = tmp_path / "tasks.json"
    split_path = tmp_path / "split_tasks.json"
    _write_tasks(tasks_path, set(split_data["base"]))
    split_path.write_text(json.dumps(split_data), encoding="utf-8")
    catalog = RetailTaskCatalog.from_files(tasks_path, split_path)

    with pytest.raises(RuntimeError, match="235237983dd826c6"):
        catalog.require_official_compatibility()


@pytest.mark.parametrize("split", ("test", "base"))
def test_require_learning_split_rejects_non_learning_splits(split: str) -> None:
    with pytest.raises(ValueError, match="learning split"):
        require_learning_split(split)


def test_require_learning_split_allows_train() -> None:
    require_learning_split("train")


def test_from_files_rejects_split_task_ids_missing_from_catalog(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    split_path = tmp_path / "split_tasks.json"
    _write_tasks(tasks_path, {"0"})
    split_path.write_text(
        json.dumps({"train": ["0"], "test": ["1"], "base": ["0", "1"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="1"):
        RetailTaskCatalog.from_files(tasks_path, split_path)


def test_from_files_rejects_invalid_split_structure(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    split_path = tmp_path / "split_tasks.json"
    _write_tasks(tasks_path, {"0"})
    split_path.write_text(json.dumps({"train": ["0"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="test"):
        RetailTaskCatalog.from_files(tasks_path, split_path)


def test_runtime_probe_reports_checkout_metadata_without_leaking_sys_path(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "tau2-bench"
    (checkout / "data" / "tau2" / "domains" / "retail").mkdir(parents=True)
    (checkout / "src" / "tau2" / "gym").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        "[project]\nname = 'tau2-bench'\nversion = '2.0.0'\n", encoding="utf-8"
    )
    (checkout / "src" / "tau2" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "src" / "tau2" / "gym" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "src" / "tau2" / "gym" / "gym_agent.py").write_text(
        "class AgentGymEnv:\n    pass\n", encoding="utf-8"
    )
    _write_tasks(
        checkout / "data" / "tau2" / "domains" / "retail" / "tasks.json", {"0"}
    )
    (checkout / "data" / "tau2" / "domains" / "retail" / "split_tasks.json").write_text(
        json.dumps({"train": ["0"], "test": [], "base": ["0"]}), encoding="utf-8"
    )
    subprocess.run(["git", "init"], cwd=checkout, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "fixture"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )

    before_sys_path = list(sys.path)
    fingerprint = Tau2Runtime.inspect(checkout)

    assert fingerprint.repo_path == checkout.resolve()
    assert len(fingerprint.git_commit) == 40
    assert fingerprint.package_version == "2.0.0"
    assert fingerprint.retail_tasks_path == (
        checkout / "data" / "tau2" / "domains" / "retail" / "tasks.json"
    )
    assert fingerprint.retail_split_path == (
        checkout / "data" / "tau2" / "domains" / "retail" / "split_tasks.json"
    )
    assert fingerprint.gym_available is True
    assert sys.path == before_sys_path


def test_verified_gym_factory_rejects_agent_gym_from_another_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_checkout = tmp_path / "expected-tau2"
    actual_checkout = tmp_path / "actual-tau2"
    (expected_checkout / "src" / "tau2").mkdir(parents=True)
    actual_module = actual_checkout / "src" / "tau2" / "gym" / "gym_agent.py"
    actual_module.parent.mkdir(parents=True)
    actual_module.write_text("", encoding="utf-8")
    fake_factory = lambda **kwargs: kwargs
    monkeypatch.setattr(
        "tau3_retail_evolver.envs.runtime.importlib.import_module",
        lambda name: SimpleNamespace(
            __file__=str(actual_module), AgentGymEnv=fake_factory
        ),
    )

    with pytest.raises(RuntimeError) as error:
        Tau2Runtime.load_verified_gym_factory(expected_checkout)

    message = str(error.value)
    assert str((expected_checkout / "src").resolve()) in message
    assert str(actual_module.resolve()) in message


def test_runtime_probe_imports_exact_agent_module_and_requires_callable_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "tau2-bench"
    retail_root = checkout / "data" / "tau2" / "domains" / "retail"
    source_root = checkout / "src"
    (source_root / "tau2").mkdir(parents=True)
    retail_root.mkdir(parents=True)
    _write_tasks(retail_root / "tasks.json", {"0"})
    (retail_root / "split_tasks.json").write_text(
        json.dumps({"train": ["0"], "test": [], "base": ["0"]}),
        encoding="utf-8",
    )
    imported_names: list[str] = []

    def import_noncallable_agent(name: str) -> SimpleNamespace:
        imported_names.append(name)
        return SimpleNamespace(
            __file__=str(source_root / "tau2" / "gym" / "gym_agent.py"),
            AgentGymEnv=object(),
        )

    monkeypatch.setattr(runtime_module, "_git_commit", lambda path: "a" * 40)
    monkeypatch.setattr(runtime_module, "_package_version", lambda path: "1.0.0")
    monkeypatch.setattr(runtime_module.importlib, "import_module", import_noncallable_agent)

    with pytest.raises(RuntimeError, match="callable"):
        Tau2Runtime.inspect(checkout)

    assert imported_names == ["tau2.gym.gym_agent"]


def test_runtime_probe_missing_checkout_names_path_and_recovery_action(
    tmp_path: Path,
) -> None:
    missing_checkout = tmp_path / "missing-tau2-bench"

    with pytest.raises(RuntimeError) as error:
        Tau2Runtime.inspect(missing_checkout)

    assert str(missing_checkout.resolve()) in str(error.value)
    assert "Clone" in str(error.value)


def test_runtime_probe_wraps_missing_git_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "tau2-bench"
    checkout.mkdir()

    def raise_missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git executable not found")

    monkeypatch.setattr(subprocess, "run", raise_missing_git)

    with pytest.raises(RuntimeError) as error:
        Tau2Runtime.inspect(checkout)

    message = str(error.value)
    assert str(checkout.resolve()) in message
    assert "Install Git" in message
    assert "PATH" in message
