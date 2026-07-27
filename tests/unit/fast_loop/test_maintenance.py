from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.decisions import MaintenanceDecision
from tau3_retail_evolver.fast_loop.maintenance import (
    _bind_command_round,
    MaintenanceState,
    bounded_diagnostics,
    run_due_maintenance,
)
from tau3_retail_evolver.fast_loop.prompts import MAX_DIAGNOSTIC_CONTENT_CHARS
from tau3_retail_evolver.fast_loop.runner import LifecycleResponse
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.operations import DeleteCommand
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.types import MemoryStatus, MemoryTier


@dataclass
class EventCollector:
    events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def test_bind_command_round_overrides_model_supplied_round() -> None:
    decision = MaintenanceDecision(
        commands=(
            DeleteCommand(
                memory_ids=("memory-1",),
                updated_round=0,
                reason="stale",
            ),
        )
    )

    bound = _bind_command_round(decision, maintenance_round=5)

    assert isinstance(bound.commands[0], DeleteCommand)
    assert bound.commands[0].updated_round == 5


class ScriptedPolicy:
    def __init__(
        self,
        outputs: list[str | BaseException],
        repairs: list[str | BaseException] | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.repairs = list(repairs or [])
        self.prompts: list[Any] = []
        self.repair_calls: list[tuple[Any, str, str]] = []

    def generate(self, prompt: Any) -> LifecycleResponse:
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return _response(output)

    def repair(self, prompt: Any, raw_output: str, error: str) -> LifecycleResponse:
        self.repair_calls.append((prompt, raw_output, error))
        output = self.repairs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return _response(output)


class BlockingPolicy(ScriptedPolicy):
    def __init__(self) -> None:
        super().__init__(["{\"commands\":[]}"])
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt: Any) -> LifecycleResponse:
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().generate(prompt)


def _response(raw_output: str) -> LifecycleResponse:
    return LifecycleResponse(
        raw_output=raw_output,
        sampling_params={"temperature": 0.7, "top_p": 0.9},
        latency_s=0.01,
    )


def _context(events: EventCollector, **overrides: Any) -> RunContext:
    values = {
        "run_id": "learn-run-sensitive",
        "iteration": 4,
        "split": "train",
        "model_revision": "Qwen@on-policy",
        "adapter_revision": "adapter-4",
        "memory_snapshot_id": "snapshot-3",
        "seed": 23,
        "event_writer": events,
        "mode": RunMode.LEARN,
    }
    values.update(overrides)
    return RunContext(**values)


def _run(
    repository: MemoryRepository,
    policy: ScriptedPolicy,
    events: EventCollector | None = None,
    *,
    completed_train_tasks: int = 30,
    period: int = 30,
    context: RunContext | None = None,
    per_tier_limit: int = 100,
):
    collector = events or EventCollector()
    return run_due_maintenance(
        completed_train_tasks=completed_train_tasks,
        period=period,
        repository=repository,
        policy=policy,
        context=context or _context(collector),
        per_tier_limit=per_tier_limit,
    )


def _seed(repository: MemoryRepository):
    first = repository.add(
        tier="tip",
        content="Verify identity before refund.",
        source_task_ids=("task-secret-1",),
        created_round=0,
        metadata={"private": "do-not-log"},
        embedding=(1.0, 0.0),
        embedding_model_revision="embedding-secret",
    )
    second = repository.add(
        tier="tip",
        content="Ask for confirmation before refund.",
        source_task_ids=("task-secret-2",),
        created_round=0,
    )
    tool = repository.add(
        tier="tool",
        content="Use the refund helper.",
        source_task_ids=("task-secret-3",),
        created_round=0,
    )
    return first, second, tool


def _state_payload(repository: MemoryRepository) -> dict[str, Any]:
    return json.loads((repository.root / "maintenance_state.json").read_text("utf-8"))


@pytest.mark.parametrize("completed_train_tasks", range(30))
def test_round_zero_is_not_due_and_emits_no_events(
    tmp_path: Path, completed_train_tasks: int
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    policy = ScriptedPolicy([])
    events = EventCollector()

    result = _run(
        repository,
        policy,
        events,
        completed_train_tasks=completed_train_tasks,
    )

    assert result.due is False
    assert result.executed is False
    assert result.maintenance_round == 0
    assert result.commands == ()
    assert policy.prompts == []
    assert events.events == []
    assert not (repository.root / "maintenance_state.json").exists()


def test_rounds_execute_once_resume_and_survive_reopen(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    repository = MemoryRepository(root)
    first_policy = ScriptedPolicy(["{\"commands\":[]}"])

    first = _run(repository, first_policy, completed_train_tasks=30)
    repeated = _run(repository, ScriptedPolicy([]), completed_train_tasks=30)
    reopened = MemoryRepository(root)
    resumed = _run(reopened, ScriptedPolicy([]), completed_train_tasks=30)
    second = _run(
        reopened,
        ScriptedPolicy(["{\"commands\":[]}"]),
        completed_train_tasks=60,
    )

    assert (first.due, first.executed, first.maintenance_round) == (True, True, 1)
    assert (repeated.due, repeated.executed) == (True, False)
    assert (resumed.due, resumed.executed) == (True, False)
    assert (second.due, second.executed, second.maintenance_round) == (True, True, 2)
    assert _state_payload(reopened) == {
        "schema_version": 2,
        "completed_rounds": [1, 2],
        "review_cursor_by_tier": {
            "trajectory": None,
            "tip": None,
            "skill": None,
            "tool": None,
        },
    }
    assert (root / "maintenance_state.json").read_bytes() == (
        b'{"completed_rounds":[1,2],"review_cursor_by_tier":{"skill":null,"tip":null,"tool":null,"trajectory":null},"schema_version":2}\n'
    )


def test_current_missed_round_executes_once_after_downtime(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    policy = ScriptedPolicy(["{\"commands\":[]}"])

    result = _run(repository, policy, completed_train_tasks=61)

    assert (result.due, result.executed, result.maintenance_round) == (True, True, 2)
    assert _state_payload(repository)["completed_rounds"] == [2]


@pytest.mark.parametrize(
    ("completed_train_tasks", "period", "message"),
    [(-1, 30, "non-negative"), (30, 0, "positive"), (30, -1, "positive")],
)
def test_schedule_inputs_are_validated(
    tmp_path: Path, completed_train_tasks: int, period: int, message: str
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    with pytest.raises(ValueError, match=message):
        _run(
            repository,
            ScriptedPolicy([]),
            completed_train_tasks=completed_train_tasks,
            period=period,
        )


def test_maintenance_state_is_frozen_and_canonical() -> None:
    state = MaintenanceState(completed_rounds=(1, 3))
    assert state.schema_version == 2
    with pytest.raises((AttributeError, TypeError)):
        state.completed_rounds = (1,)  # type: ignore[misc]
    for invalid in ((0,), (2, 1), (1, 1)):
        with pytest.raises(ValueError):
            MaintenanceState(completed_rounds=invalid)


def test_bounded_diagnostics_are_public_active_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)
    repository.add(
        tier="trajectory",
        content="A public trajectory.",
        source_task_ids=("hidden-trajectory-task",),
        created_round=0,
    )
    repository.add(
        tier="skill",
        content="A public skill.",
        source_task_ids=("hidden-skill-task",),
        created_round=0,
    )
    repository.update_status(second.id, MemoryStatus.RETIRED, updated_round=1)

    diagnostics = bounded_diagnostics(repository, per_tier_limit=1)

    assert tuple(diagnostics) == ("trajectory", "tip", "skill", "tool")
    assert all(set(tier_data) == {"items"} for tier_data in diagnostics.values())
    assert [item["id"] for item in diagnostics["tip"]["items"]] == [first.id]
    assert all(len(tier_data["items"]) == 1 for tier_data in diagnostics.values())
    allowed = {
        "id",
        "tier",
        "content",
        "version",
        "status",
    }
    for tier_data in diagnostics.values():
        assert all(set(item) == allowed for item in tier_data["items"])
    assert diagnostics == bounded_diagnostics(repository, per_tier_limit=1)
    assert "task-secret" not in json.dumps(diagnostics)
    assert "embedding" not in json.dumps(diagnostics)
    assert "do-not-log" not in json.dumps(diagnostics)


def test_bounded_diagnostics_truncate_content_without_mutating_memory(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    original_content = "x" * (MAX_DIAGNOSTIC_CONTENT_CHARS + 37)
    item = repository.add(
        tier="tip",
        content=original_content,
        source_task_ids=("hidden-source-task",),
        created_round=0,
        metadata={"api_token": "must-not-appear"},
    )

    diagnostics = bounded_diagnostics(repository, per_tier_limit=1)

    diagnostic = diagnostics["tip"]["items"][0]
    assert diagnostic["content"] == original_content[:MAX_DIAGNOSTIC_CONTENT_CHARS]
    assert repository.get(item.id).content == original_content
    assert "hidden-source-task" not in json.dumps(diagnostics)
    assert "must-not-appear" not in json.dumps(diagnostics)


@pytest.mark.parametrize("limit", [0, 101])
def test_bounded_diagnostics_reject_invalid_limits(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="per_tier_limit"):
        bounded_diagnostics(MemoryRepository(tmp_path / "memory"), per_tier_limit=limit)


def test_lookup_command_returns_ids_and_emits_canonical_events(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, _, _ = _seed(repository)
    policy = ScriptedPolicy(
        [json.dumps({"commands": [{"operation": "lookup", "memory_ids": [first.id]}]})]
    )
    events = EventCollector()

    result = _run(repository, policy, events)

    assert result.looked_up_ids == (first.id,)
    assert result.created_ids == ()
    assert result.updated_ids == ()
    assert [event["event_type"] for event in events.events] == [
        "MaintenanceStarted",
        "MaintenanceProposed",
        "MaintenanceCommitted",
    ]
    started, proposed, committed = events.events
    assert started["maintenance_round"] == 1
    assert started["completed_train_tasks"] == 30
    assert started["period"] == 30
    assert started["per_tier_counts"] == {
        "trajectory": 0,
        "tip": 2,
        "skill": 0,
        "tool": 1,
    }
    assert started["diagnostics"] == policy.prompts[0].payload["diagnostics"]
    assert proposed["commands"] == [
        {"operation": "lookup", "memory_ids": [first.id]}
    ]
    assert proposed["repair_used"] is False
    assert committed["looked_up_ids"] == [first.id]
    assert committed["created_ids"] == []
    assert committed["updated_ids"] == []
    assert committed["completed_rounds"] == [1]
    assert all(event["task_id"] == "maintenance-round-1" for event in events.events)
    prompt_text = policy.prompts[0].model_dump_json()
    assert "learn-run-sensitive" not in prompt_text
    assert "maintenance-round-1" not in prompt_text
    assert "task-secret" not in prompt_text


def test_same_tier_merge_and_soft_delete_use_current_round(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, tool = _seed(repository)
    merge_policy = ScriptedPolicy(
        [
            json.dumps(
                {
                    "commands": [
                        {
                            "operation": "merge",
                            "source_ids": [first.id, second.id],
                            "content": "Verify identity and confirmation before refund.",
                            "updated_round": 1,
                            "metadata": {"reason": "deduplicate"},
                        }
                    ]
                }
            )
        ]
    )

    merged = _run(repository, merge_policy, completed_train_tasks=30)
    deleted = _run(
        repository,
        ScriptedPolicy(
            [
                json.dumps(
                    {
                        "commands": [
                            {
                                "operation": "delete",
                                "memory_ids": [tool.id],
                                "updated_round": 2,
                                "reason": "obsolete",
                            }
                        ]
                    }
                )
            ]
        ),
        completed_train_tasks=60,
    )

    assert len(merged.created_ids) == 1
    assert set(merged.updated_ids) == {first.id, second.id}
    assert repository.get(merged.created_ids[0]).tier == MemoryTier.TIP
    assert deleted.updated_ids == (tool.id,)
    assert repository.get(tool.id).status == MemoryStatus.RETIRED


def test_nested_camelcase_attribution_triggers_clean_repair(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)
    events = EventCollector()
    sentinel = "maintenance-attribution-sentinel"
    base_command = {
        "operation": "merge",
        "source_ids": [first.id, second.id],
        "content": "Verify identity and confirmation before refund.",
        "updated_round": 1,
    }
    policy = ScriptedPolicy(
        [
            json.dumps(
                {
                    "commands": [
                        {
                            **base_command,
                            "metadata": {
                                "audit": {"attributionScore": sentinel}
                            },
                        }
                    ]
                }
            )
        ],
        repairs=[
            json.dumps(
                {
                    "commands": [
                        {
                            **base_command,
                            "metadata": {"audit": {"reason": "deduplicate"}},
                        }
                    ]
                }
            )
        ],
    )

    result = _run(repository, policy, events)

    assert len(policy.repair_calls) == 1
    assert events.events[1]["event_type"] == "MaintenanceProposed"
    assert events.events[1]["repair_used"] is True
    assert sentinel not in repr(events.events)
    merged = repository.get(result.created_ids[0])
    assert merged is not None
    assert merged.metadata == {
        "audit": {"reason": "deduplicate"},
        "merged_from": sorted([first.id, second.id]),
    }
    assert sentinel not in repr(merged.metadata)


def test_nested_credential_metadata_triggers_clean_repair(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)
    events = EventCollector()
    sentinel = "maintenance-credential-sentinel"
    base_command = {
        "operation": "merge",
        "source_ids": [first.id, second.id],
        "content": "Verify identity and confirmation before refund.",
        "updated_round": 1,
    }
    policy = ScriptedPolicy(
        [
            json.dumps(
                {
                    "commands": [
                        {
                            **base_command,
                            "metadata": {"audit": {"apiToken": sentinel}},
                        }
                    ]
                }
            )
        ],
        repairs=[
            json.dumps(
                {
                    "commands": [
                        {
                            **base_command,
                            "metadata": {"audit": {"reason": "deduplicate"}},
                        }
                    ]
                }
            )
        ],
    )

    result = _run(repository, policy, events)

    assert len(policy.repair_calls) == 1
    assert events.events[1]["repair_used"] is True
    merged = repository.get(result.created_ids[0])
    assert merged is not None
    assert sentinel not in json.dumps(merged.model_dump(mode="json"))
    assert sentinel not in json.dumps(events.events)


def test_attribution_separator_variant_after_repair_fails_without_mutation(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)
    events = EventCollector()
    sentinel = "maintenance-attribution-sentinel"
    base_command = {
        "operation": "merge",
        "source_ids": [first.id, second.id],
        "content": "Verify identity and confirmation before refund.",
        "updated_round": 1,
    }
    policy = ScriptedPolicy(
        [
            json.dumps(
                {
                    "commands": [
                        {
                            **base_command,
                            "metadata": {
                                "audit": {"attributionScore": sentinel}
                            },
                        }
                    ]
                }
            )
        ],
        repairs=[
            json.dumps(
                {
                    "commands": [
                        {
                            **base_command,
                            "metadata": {
                                "audit": {"attribution.score": sentinel}
                            },
                        }
                    ]
                }
            )
        ],
    )

    with pytest.raises(ValueError, match="attribution score"):
        _run(repository, policy, events)

    assert len(policy.repair_calls) == 1
    assert [event["event_type"] for event in events.events] == [
        "MaintenanceStarted",
        "MaintenanceFailed",
    ]
    assert sentinel not in repr(events.events)
    assert repository.get(first.id).status == MemoryStatus.ACTIVE
    assert repository.get(second.id).status == MemoryStatus.ACTIVE
    assert len(repository.list(tier=MemoryTier.TIP)) == 2
    assert not (repository.root / "maintenance_state.json").exists()


def test_empty_decision_completes_noop_round(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    result = _run(repository, ScriptedPolicy(["{\"commands\":[]}"]))
    assert result.commands == ()
    assert result.executed is True
    assert _state_payload(repository)["completed_rounds"] == [1]


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (
            {
                "commands": [
                    {"operation": "lookup", "memory_ids": ["mem-tip-a"]},
                    {
                        "operation": "delete",
                        "memory_ids": ["mem-tip-b"],
                        "updated_round": 1,
                        "reason": "mixed",
                    },
                ]
            },
            "lookup commands cannot be mixed",
        ),
        (
            {
                "commands": [
                    {
                        "operation": "delete",
                        "memory_ids": ["mem-tip-a"],
                        "updated_round": 2,
                        "reason": "wrong round",
                    }
                ]
            },
            "updated_round",
        ),
    ],
)
def test_invalid_command_shape_fails_after_one_repair_without_state(
    tmp_path: Path, decision: dict[str, Any], message: str
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    encoded = json.dumps(decision)
    policy = ScriptedPolicy([encoded], repairs=[encoded])

    with pytest.raises(ValueError, match=message):
        _run(repository, policy)

    assert len(policy.repair_calls) == 1
    assert not (repository.root / "maintenance_state.json").exists()


def test_cross_tier_merge_is_rejected_by_real_operations(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, _, tool = _seed(repository)
    decision = json.dumps(
        {
            "commands": [
                {
                    "operation": "merge",
                    "source_ids": [first.id, tool.id],
                    "content": "Invalid cross-tier merge.",
                    "updated_round": 1,
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="same tier"):
        _run(repository, ScriptedPolicy([decision]))

    assert repository.get(first.id).status == MemoryStatus.ACTIVE
    assert repository.get(tool.id).status == MemoryStatus.ACTIVE
    assert not (repository.root / "maintenance_state.json").exists()


def test_invalid_output_failed_repair_emits_sanitized_failure(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    policy = ScriptedPolicy(["secret-invalid"], repairs=["still-secret-invalid"])

    with pytest.raises(ValueError, match="invalid maintenance decision after repair"):
        _run(repository, policy, events)

    assert len(policy.repair_calls) == 1
    assert [event["event_type"] for event in events.events] == [
        "MaintenanceStarted",
        "MaintenanceFailed",
    ]
    failed = events.events[-1]
    assert failed["maintenance_round"] == 1
    assert failed["error"] == {"type": "ValueError", "message": "operation failed"}
    assert "secret-invalid" not in repr(events.events)
    assert not (repository.root / "maintenance_state.json").exists()


def test_operation_failure_does_not_complete_round(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    decision = '{"commands":[{"operation":"lookup","memory_ids":["missing"]}]}'

    with pytest.raises(KeyError, match="unknown memory"):
        _run(repository, ScriptedPolicy([decision]), events)

    assert [event["event_type"] for event in events.events] == [
        "MaintenanceStarted",
        "MaintenanceProposed",
        "MaintenanceFailed",
    ]
    assert not (repository.root / "maintenance_state.json").exists()


def test_state_write_failure_leaves_round_retryable_after_command_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tau3_retail_evolver.fast_loop import maintenance

    repository = MemoryRepository(tmp_path / "memory")
    first, _, _ = _seed(repository)
    decision = json.dumps(
        {
            "commands": [
                {
                    "operation": "delete",
                    "memory_ids": [first.id],
                    "updated_round": 1,
                    "reason": "obsolete",
                }
            ]
        }
    )
    original_write = maintenance.write_bytes_atomic
    failed = False

    def fail_once(path: Path, content: bytes) -> None:
        nonlocal failed
        if path.name == "maintenance_state.json" and not failed:
            failed = True
            raise OSError("sensitive state-write detail")
        original_write(path, content)

    monkeypatch.setattr(maintenance, "write_bytes_atomic", fail_once)
    with pytest.raises(OSError, match="sensitive state-write detail"):
        _run(repository, ScriptedPolicy([decision]))

    assert repository.get(first.id).status == MemoryStatus.RETIRED
    assert not (repository.root / "maintenance_state.json").exists()

    retried = _run(repository, ScriptedPolicy([decision]))
    assert retried.updated_ids == (first.id,)
    assert repository.get(first.id).version == 2
    assert _state_payload(repository)["completed_rounds"] == [1]


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"schema_version":2,"completed_rounds":[]}',
        '{"schema_version":1,"completed_rounds":[1,1]}',
        '{"schema_version":1,"completed_rounds":[2,1]}',
        '{"schema_version":1,"completed_rounds":[3]}',
        '{"schema_version":1,"completed_rounds":[0]}',
    ],
)
def test_malformed_state_is_rejected_before_policy(tmp_path: Path, payload: str) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    (repository.root / "maintenance_state.json").write_text(payload, encoding="utf-8")
    policy = ScriptedPolicy([])

    with pytest.raises(ValueError, match="maintenance state"):
        _run(repository, policy, completed_train_tasks=60)

    assert policy.prompts == []


@pytest.mark.parametrize(
    ("split", "mode"),
    [
        ("test", RunMode.LEARN),
        ("train", RunMode.BASELINE),
        ("train", RunMode.EVALUATE),
    ],
)
def test_nonlearning_context_is_rejected_before_state_or_policy_access(
    tmp_path: Path, split: str, mode: RunMode
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    (repository.root / "maintenance_state.json").write_text("not-json", encoding="utf-8")
    events = EventCollector()
    policy = ScriptedPolicy([])

    with pytest.raises(ValueError, match="train.*learn"):
        _run(repository, policy, events, context=_context(events, split=split, mode=mode))

    assert policy.prompts == []
    assert events.events == []


def test_read_only_repository_is_rejected_before_state_or_policy_access(
    tmp_path: Path,
) -> None:
    source = MemoryRepository(tmp_path / "memory")
    snapshot = source.snapshot()
    repository = ReadOnlyMemoryRepository(snapshot.path)
    events = EventCollector()
    policy = ScriptedPolicy([])

    with pytest.raises(TypeError, match="mutable MemoryRepository"):
        run_due_maintenance(
            completed_train_tasks=30,
            period=30,
            repository=repository,  # type: ignore[arg-type]
            policy=policy,
            context=_context(events),
        )

    assert policy.prompts == []
    assert events.events == []


def test_event_failure_does_not_replace_original_error(tmp_path: Path) -> None:
    class FailingWriter:
        def __init__(self) -> None:
            self.calls = 0

        def append(self, event: dict[str, Any]) -> None:
            self.calls += 1
            if self.calls >= 2:
                raise OSError("event sink failed")

    repository = MemoryRepository(tmp_path / "memory")
    writer = FailingWriter()
    policy = ScriptedPolicy(["invalid"], repairs=["also-invalid"])

    with pytest.raises(ValueError, match="invalid maintenance decision") as error:
        _run(
            repository,
            policy,
            context=_context(EventCollector(), event_writer=writer),
        )

    assert any("event sink failed" in note for note in error.value.__notes__)


def test_two_threads_execute_same_round_only_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tau3_retail_evolver.fast_loop import maintenance

    root = tmp_path / "memory"
    MemoryRepository(root)
    policy = BlockingPolicy()
    first_events = EventCollector()
    second_events = EventCollector()
    second_contended = threading.Event()
    tracking_guard = threading.Lock()
    lock_acquire_calls = 0
    application_calls = 0
    actual_apply_batch = maintenance.MemoryOperations.apply_batch
    actual_scheduler_lock = maintenance.reentrant_process_lock(
        root,
        namespace="fast-loop-maintenance",
    )
    actual_thread_lock = actual_scheduler_lock._thread_lock
    first_thread_id: int | None = None
    second_thread_checked = False

    class TrackingRLock:
        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            nonlocal first_thread_id, lock_acquire_calls, second_thread_checked
            current_thread_id = threading.get_ident()
            with tracking_guard:
                lock_acquire_calls += 1
                if first_thread_id is None:
                    first_thread_id = current_thread_id
                probe_contention = (
                    current_thread_id != first_thread_id
                    and not second_thread_checked
                )
                if probe_contention:
                    second_thread_checked = True

            if probe_contention:
                if actual_thread_lock.acquire(blocking=False):
                    return True
                second_contended.set()
                return actual_thread_lock.acquire()
            if timeout == -1:
                return actual_thread_lock.acquire(blocking)
            return actual_thread_lock.acquire(blocking, timeout)

        def release(self) -> None:
            actual_thread_lock.release()

    monkeypatch.setattr(
        actual_scheduler_lock,
        "_thread_lock",
        TrackingRLock(),
    )

    def tracking_apply_batch(self: Any, commands: Any):
        nonlocal application_calls
        with tracking_guard:
            application_calls += 1
        return actual_apply_batch(self, commands)

    monkeypatch.setattr(
        maintenance.MemoryOperations,
        "apply_batch",
        tracking_apply_batch,
    )

    def invoke(events: EventCollector):
        return _run(MemoryRepository(root), policy, events)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(invoke, first_events)
        assert policy.entered.wait(timeout=5)
        second_future = executor.submit(invoke, second_events)
        assert second_contended.wait(timeout=5)
        policy.release.set()
        results = (first_future.result(timeout=5), second_future.result(timeout=5))

    assert sorted(result.executed for result in results) == [False, True]
    assert lock_acquire_calls == 2
    assert second_contended.is_set()
    assert len(policy.prompts) == 1
    assert application_calls == 1
    assert sum(len(events.events) for events in (first_events, second_events)) == 3
    assert _state_payload(MemoryRepository(root))["completed_rounds"] == [1]
