from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.eval.guard import EvaluationGuard, EvaluationProtocol
from tau3_retail_evolver.eval.metrics import (
    EvaluationProvenance,
    build_evaluation_report,
    compare_evaluation_reports,
)
from tau3_retail_evolver.eval.runner import run_evaluation_trials
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.runner import FastLoopConfig, LifecycleResponse
from tau3_retail_evolver.memory.paths import training_memory_root
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENV = (
    "QWEN_BASE_URL",
    "QWEN_MODEL_REVISION",
)
MISSING_ENV = tuple(name for name in REQUIRED_ENV if not os.environ.get(name))
USER_PROVIDER_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class EventCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class DeterministicEmbeddings:
    model_revision = "fake-embedding@1"
    dimension = 2

    def embed(self, _text: str) -> tuple[float, ...]:
        return (1.0, 0.0)

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]


class OneStepEnvironment:
    def reset(self, *, seed: int) -> ResetResult:
        return ResetResult(
            observation=f"Customer request for seed {seed}",
            info={
                "policy": {"text": "Follow policy"},
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup_order"},
                    }
                ],
            },
        )

    def step(self, _action: str) -> StepResult:
        return StepResult(
            observation="Done",
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            info={
                "parse_error": None,
                "reward_info": {
                    "reward": 1.0,
                    "reward_breakdown": {"DB": 1.0},
                },
                "simulation_run": {"termination_reason": "agent_stop"},
            },
        )

    def close(self) -> None:
        pass


class ScriptedPolicy:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

    def generate(self, _prompt: Any) -> LifecycleResponse:
        return LifecycleResponse(
            raw_output=self.outputs.pop(0),
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            latency_s=0.0,
        )

    def repair(
        self,
        _prompt: Any,
        _raw_output: str,
        _error: str,
    ) -> LifecycleResponse:
        raise AssertionError("repair was not expected")


def _context(run_id: str, events: EventCollector) -> RunContext:
    return RunContext(
        run_id=run_id,
        iteration=1,
        split="test",
        model_revision="qwen-sha",
        adapter_revision="adapter-sha",
        memory_snapshot_id=None,
        seed=42,
        event_writer=events,
        mode=RunMode.EVALUATE,
        task_groups={"75": f"retail-actions-v1:{'7' * 64}"},
    )


def _provenance(
    guard: EvaluationGuard,
    *,
    memory_snapshot_id: str | None,
) -> EvaluationProvenance:
    return EvaluationProvenance(
        run_id=guard.run_id,
        protocol=guard.protocol,
        official_base_reproduction=False,
        split="test",
        checkpoint="checkpoint-1",
        base_model="Qwen/Qwen3.5-9B",
        model_revision="qwen-sha",
        adapter_revision="adapter-sha",
        tau2_commit="tau2-sha",
        split_hash="split-sha",
        task_ids=("75",),
        seeds=(42,),
        user_simulator_config={"model": "deepseek/deepseek-v4-pro"},
        nl_evaluator={"model": "openrouter/openai/gpt-4.1"},
        memory_snapshot_id=memory_snapshot_id,
        max_episode_steps=40,
        model_serving_contract={"max_tokens": 8192},
        capabilities=guard.capabilities.as_dict(),
    )


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _real_evaluation_config_path() -> Path:
    configured = Path(
        os.environ.get(
            "RETAIL_EVALUATION_CONFIG",
            str(WORKTREE_ROOT / "configs" / "default.yaml"),
        )
    )
    if not configured.is_absolute():
        configured = WORKTREE_ROOT / configured
    return configured


def test_static_and_streaming_reports_preserve_train_memory_isolation(
    tmp_path: Path,
) -> None:
    train_root = training_memory_root("retail", root=tmp_path)
    training_memory = MemoryRepository(train_root)
    item = training_memory.add(
        tier="tip",
        content="Verify the order before changing it.",
        source_task_ids=("0",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )
    snapshot = training_memory.snapshot()
    train_hashes = _hash_tree(train_root)

    static_guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STATIC,
        run_id="eval-static",
        agent_id="retail",
        project_root=tmp_path,
        memory_snapshot_path=snapshot.path,
    )
    static_events = EventCollector()
    static_run = run_evaluation_trials(
        task_ids=("75",),
        seeds=(42,),
        env_factory=lambda _task_id: OneStepEnvironment(),
        policy=ScriptedPolicy(
            [
                json.dumps({"memory_ids": [item.id]}),
                '{"action":"finish"}',
            ]
        ),
        guard=static_guard,
        retriever_factory=lambda _memory: Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(memory_enabled=True),
        context=_context("eval-static", static_events),
        maintenance_period=30,
    )
    static_report = build_evaluation_report(
        _provenance(
            static_guard,
            memory_snapshot_id=snapshot.memory_snapshot_id,
        ),
        static_run,
    )

    streaming_guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STREAMING,
        run_id="eval-streaming",
        agent_id="retail",
        project_root=tmp_path,
    )
    streaming_events = EventCollector()
    streaming_run = run_evaluation_trials(
        task_ids=("75",),
        seeds=(42,),
        env_factory=lambda _task_id: OneStepEnvironment(),
        policy=ScriptedPolicy(
            [
                '{"memory_ids":[]}',
                '{"action":"finish"}',
                json.dumps(
                    {
                        "memories": [
                            {
                                "tier": "tip",
                                "content": "Check the order first.",
                                "metadata": {},
                            }
                        ]
                    }
                ),
            ]
        ),
        guard=streaming_guard,
        retriever_factory=lambda _memory: Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(memory_enabled=True),
        context=_context("eval-streaming", streaming_events),
        maintenance_period=30,
    )
    streaming_report = build_evaluation_report(
        _provenance(streaming_guard, memory_snapshot_id=None),
        streaming_run,
    )
    comparison = compare_evaluation_reports(
        {
            "trained_static": static_report,
            "trained_streaming": streaming_report,
        },
        baseline_label="trained_static",
    )

    assert _hash_tree(train_root) == train_hashes
    assert static_report["summary"]["success_rate"] == 1.0
    assert streaming_report["summary"]["success_rate"] == 1.0
    assert comparison["rows"][1]["protocol"] == "test_streaming"
    assert all(event["mode"] == "evaluate" for event in static_events.events)
    assert all(event["trial_index"] == 0 for event in streaming_events.events)
    quarantine = streaming_guard.quarantine_root / "trial-000"
    assert MemoryRepository(quarantine).list()


real_retail_evaluation = pytest.mark.skipif(
    os.environ.get("RUN_RETAIL_EVALUATION_INTEGRATION") != "1"
    or bool(MISSING_ENV),
    reason=(
        "set RUN_RETAIL_EVALUATION_INTEGRATION=1 and provide the Qwen "
        "runtime variables"
    ),
)


@pytest.mark.tau2_integration
@real_retail_evaluation
def test_real_retail_test_smoke_produces_official_report(
    tmp_path: Path,
) -> None:
    config_path = _real_evaluation_config_path()
    config = load_config(config_path)
    credential_envs = {config.evaluation.nl_assertions.api_key_env}
    user_provider = config.tau2.user_llm.partition("/")[0].lower()
    user_key_env = USER_PROVIDER_KEY_ENV.get(user_provider)
    if user_key_env is not None:
        credential_envs.add(user_key_env)
    missing_credentials = sorted(
        name for name in credential_envs if not os.environ.get(name)
    )
    if missing_credentials:
        pytest.skip(
            "configured evaluation credentials are missing: "
            + ", ".join(missing_credentials)
        )
    runtime = Tau2Runtime.inspect_metadata(config.tau2.repo_path)
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(
        runtime.retail_tasks_path,
        runtime.retail_split_path,
    )
    catalog.require_official_compatibility()
    task_id = catalog.task_ids("test")[0]
    run_id = "pytest-stage8-test-smoke"
    output_root = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_retail",
            "--config",
            str(config_path),
            "--protocol",
            "no_memory",
            "--task-id",
            task_id,
            "--num-trials",
            "1",
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--project-root",
            str(tmp_path / "project"),
            "--qwen-base-url",
            os.environ["QWEN_BASE_URL"],
            "--model-revision",
            os.environ["QWEN_MODEL_REVISION"],
        ],
        cwd=WORKTREE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
        timeout=1800,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(
        (
            output_root
            / run_id
            / "evaluation_report.json"
        ).read_text("utf-8")
    )
    assert report["provenance"]["split"] == "test"
    assert report["provenance"]["task_ids"] == [task_id]
    assert report["provenance"]["tau2_commit"] == runtime.git_commit
    assert report["provenance"]["split_hash"] == catalog.split_sha256
    assert report["summary"]["episode_count"] == 1
    assert isinstance(report["episodes"][0]["reward_info"], dict)
