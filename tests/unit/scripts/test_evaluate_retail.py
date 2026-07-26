from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import evaluate_retail
from tau3_retail_evolver.eval.guard import EvaluationProtocol
from tau3_retail_evolver.eval.runner import EvaluationRunResult, TrialEpisode
from tau3_retail_evolver.fast_loop.events import RunMode
from tau3_retail_evolver.fast_loop.runner import EpisodeResult


def _episode(task_id: str) -> EpisodeResult:
    return EpisodeResult(
        task_id=task_id,
        final_reward=1.0,
        steps=1,
        terminal_evaluation={"reward": 1.0},
        simulation_result={"termination_reason": "agent_stop"},
        selected_memory_ids=(),
        written_memory_ids=(),
        truncated=False,
    )


def test_parser_defaults_to_full_single_trial_test_evaluation() -> None:
    args = evaluate_retail.parse_args(
        [
            "--protocol",
            "no_memory",
            "--run-id",
            "eval-001",
            "--model-revision",
            "qwen-sha",
        ]
    )

    assert args.split == "test"
    assert args.task_ids is None
    assert args.num_trials == 1
    assert args.seeds is None
    assert args.official_base_reproduction is False


def test_seed_resolution_uses_config_seed_or_exact_explicit_set() -> None:
    assert evaluate_retail._resolve_seeds(None, num_trials=4, base_seed=42) == (
        42,
        43,
        44,
        45,
    )
    assert evaluate_retail._resolve_seeds(
        [5, 9],
        num_trials=2,
        base_seed=42,
    ) == (5, 9)

    with pytest.raises(ValueError, match="num-trials"):
        evaluate_retail._resolve_seeds([5], num_trials=2, base_seed=42)
    with pytest.raises(ValueError, match="unique"):
        evaluate_retail._resolve_seeds([5, 5], num_trials=2, base_seed=42)


def test_task_resolution_defaults_to_official_order_and_validates_subset() -> None:
    official = ("75", "76", "77")

    assert evaluate_retail._resolve_task_ids(None, official) == official
    assert evaluate_retail._resolve_task_ids(["77", "75"], official) == (
        "77",
        "75",
    )
    with pytest.raises(ValueError, match="official"):
        evaluate_retail._resolve_task_ids(["0"], official)
    with pytest.raises(ValueError, match="unique"):
        evaluate_retail._resolve_task_ids(["75", "75"], official)


def test_base_reproduction_is_flag_only_and_no_memory() -> None:
    args = evaluate_retail.parse_args(
        [
            "--protocol",
            "test_static",
            "--run-id",
            "eval-base",
            "--model-revision",
            "qwen-sha",
            "--official-base-reproduction",
            "--memory-snapshot",
            "snapshot",
        ]
    )

    with pytest.raises(ValueError, match="no_memory"):
        evaluate_retail._validate_early_arguments(args)


def test_static_requires_snapshot_and_nonstatic_rejects_it() -> None:
    missing = evaluate_retail.parse_args(
        [
            "--protocol",
            "test_static",
            "--run-id",
            "eval-static",
            "--model-revision",
            "qwen-sha",
        ]
    )
    unexpected = evaluate_retail.parse_args(
        [
            "--protocol",
            "no_memory",
            "--run-id",
            "eval-none",
            "--model-revision",
            "qwen-sha",
            "--memory-snapshot",
            "snapshot",
        ]
    )

    with pytest.raises(ValueError, match="memory-snapshot"):
        evaluate_retail._validate_early_arguments(missing)
    with pytest.raises(ValueError, match="does not accept"):
        evaluate_retail._validate_early_arguments(unexpected)


def test_main_writes_manifest_events_contract_and_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = SimpleNamespace(
        tau2=SimpleNamespace(repo_path=tmp_path / "tau2", domain="retail"),
        model=SimpleNamespace(base_model="Qwen/Qwen3.5-9B"),
        rollout=SimpleNamespace(
            temperature=1.0,
            top_p=0.95,
            max_episode_steps=40,
        ),
        training=SimpleNamespace(seed=42),
        memory=SimpleNamespace(
            agent_id="retail",
            retrieve_top_k=50,
            maintenance_period=30,
        ),
        evaluation=SimpleNamespace(nl_assertions=object()),
    )
    runtime = SimpleNamespace(
        repo_path=tmp_path / "tau2",
        retail_tasks_path=tmp_path / "tasks.json",
        retail_split_path=tmp_path / "split.json",
        git_commit="tau2-sha",
    )

    class FakeCatalog:
        split_sha256 = "split-sha"

        def require_official_compatibility(self) -> None:
            pass

        def task_ids(self, split: str) -> tuple[str, ...]:
            assert split == "test"
            return ("75", "76")

    class FakeGroups:
        def signature_for(self, task_id: str) -> str:
            return f"retail-actions-v1:{task_id * 64}"[:82]

    class FakeEnvironment:
        user_simulator_config = {
            "model": "deepseek/deepseek-v4-pro",
            "api_key": "simulator-secret",
        }

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> EvaluationRunResult:
        captured.update(kwargs)
        assert kwargs["context"].mode is RunMode.EVALUATE
        return EvaluationRunResult(
            episodes=(
                TrialEpisode(0, 42, _episode("75")),
                TrialEpisode(0, 42, _episode("76")),
            ),
            maintenance_rounds_by_trial=((),),
            output_memory_snapshot_ids=(None,),
        )

    monkeypatch.setattr(evaluate_retail, "load_config", lambda _path: config)
    monkeypatch.setattr(
        evaluate_retail.Tau2Runtime,
        "inspect_metadata",
        lambda _path: runtime,
    )
    monkeypatch.setattr(
        evaluate_retail.Tau2Runtime,
        "require_pinned_commit",
        lambda _runtime: None,
    )
    monkeypatch.setattr(
        evaluate_retail.Tau2Runtime,
        "load_verified_gym_factory",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        evaluate_retail.RetailTaskCatalog,
        "from_files",
        lambda *_args: FakeCatalog(),
    )
    monkeypatch.setattr(
        evaluate_retail.RetailTaskGroups,
        "from_file",
        lambda *_args, **_kwargs: FakeGroups(),
    )
    monkeypatch.setattr(
        evaluate_retail,
        "bind_tau2_nl_assertions",
        lambda _config: {
            "model": "openrouter/openai/gpt-4.1",
            "temperature": 0.0,
        },
    )
    monkeypatch.setattr(evaluate_retail, "Tau2RetailEnv", FakeEnvironment)
    monkeypatch.setattr(
        evaluate_retail,
        "OpenAICompatibleHttpClient",
        lambda **kwargs: captured.setdefault("client", kwargs) or object(),
    )
    monkeypatch.setattr(
        evaluate_retail,
        "OpenAICompatibleFastLoopPolicy",
        lambda **kwargs: captured.setdefault("policy", kwargs) or object(),
    )
    monkeypatch.setattr(evaluate_retail, "run_evaluation_trials", fake_run)
    monkeypatch.setattr(
        evaluate_retail,
        "build_embedding_provider",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("no-memory evaluation must not load embeddings")
        ),
    )
    monkeypatch.setenv("QWEN_API_KEY", "qwen-secret")

    result = evaluate_retail.main(
        [
            "--protocol",
            "no_memory",
            "--run-id",
            "eval-001",
            "--output-root",
            str(tmp_path / "runs"),
            "--project-root",
            str(tmp_path / "project"),
            "--num-trials",
            "1",
            "--qwen-base-url",
            "http://qwen.invalid/v1",
            "--model-revision",
            "qwen-sha",
        ]
    )

    run_path = tmp_path / "runs" / "eval-001"
    manifest = json.loads((run_path / "manifest.json").read_text("utf-8"))
    report = json.loads(
        (run_path / "evaluation_report.json").read_text("utf-8")
    )
    stdout = json.loads(capsys.readouterr().out)

    assert result == 0
    assert manifest["split"] == "test"
    assert manifest["task_ids"] == ["75", "76"]
    assert manifest["rollout_options"]["protocol"] == "no_memory"
    assert manifest["rollout_options"]["num_trials"] == 1
    assert manifest["rollout_options"]["seeds"] == [42]
    assert report["summary"]["episode_count"] == 2
    assert report["summary"]["success_rate"] == 1.0
    assert report["provenance"]["protocol"] == "no_memory"
    assert captured["task_ids"] == ("75", "76")
    assert captured["seeds"] == (42,)
    assert captured["retriever_factory"] is None
    assert stdout["run_id"] == "eval-001"
    assert stdout["report_path"].endswith("evaluation_report.json")

    artifacts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in run_path.rglob("*")
        if path.is_file()
    )
    assert "qwen-secret" not in artifacts
    assert "simulator-secret" not in artifacts
