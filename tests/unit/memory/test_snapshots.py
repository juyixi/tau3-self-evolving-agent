from pathlib import Path

from tau3_evolver.benchmarks.types import PreparedBenchmark, RuntimeOrigin
from tau3_evolver.execution import ExecutionRequest
from tau3_evolver.memory.paths import training_memory_root
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.snapshots import resolve_memory


def _prepared(name: str = "retail") -> PreparedBenchmark:
    task = object()
    return PreparedBenchmark(
        name=name,
        task_type=object,
        task_catalog=(task,),
        task_ids=("1",),
        split_name="train",
        split_hash="hash",
        environment_factory=lambda: None,
        runtime=None,
        run_domain=lambda config: config,
        text_run_config_type=dict,
        registry=object(),
        runtime_origin=RuntimeOrigin(Path("tau2"), None, None),
        default_memory_namespace=name,
        task_group=name,
    )


def test_no_memory_does_not_create_memory_directories(tmp_path: Path) -> None:
    request = ExecutionRequest(
        benchmark="retail",
        mode="test",
        memory_enabled=False,
        run_id="run-1",
    )

    resolved = resolve_memory(request, _prepared(), root=tmp_path)

    assert resolved.source is None
    assert resolved.destination is None
    assert not (tmp_path / "history").exists()


def test_same_domain_training_freezes_s0_and_opens_local_destination(
    tmp_path: Path,
) -> None:
    request = ExecutionRequest(
        benchmark="retail",
        mode="train",
        memory_enabled=True,
        run_id="run-1",
    )

    resolved = resolve_memory(request, _prepared(), root=tmp_path)

    assert resolved.source is not None
    assert resolved.destination is not None
    assert resolved.source.is_read_only
    assert resolved.destination.root == training_memory_root("retail", root=tmp_path)
    assert resolved.input_snapshot_id == resolved.source.memory_snapshot_id
    assert resolved.generation == 1


def test_cross_domain_test_uses_foreign_snapshot_without_destination(
    tmp_path: Path,
) -> None:
    retail = MemoryRepository(training_memory_root("retail", root=tmp_path))
    snapshot = retail.snapshot()
    request = ExecutionRequest(
        benchmark="airline",
        mode="test",
        memory_enabled=True,
        memory_source="retail",
        memory_snapshot=Path(snapshot.memory_snapshot_id),
        run_id="run-1",
    )

    resolved = resolve_memory(request, _prepared("airline"), root=tmp_path)

    assert resolved.source_namespace == "retail"
    assert resolved.source is not None
    assert resolved.source.root == snapshot.path
    assert resolved.destination is None
    assert not training_memory_root("airline", root=tmp_path).exists()
