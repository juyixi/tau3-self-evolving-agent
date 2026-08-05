from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from tau3_evolver.benchmarks.tau2.runtime import Tau2Runtime
from tau3_evolver.benchmarks.types import PreparedBenchmark, RuntimeOrigin
from tau3_evolver.config import ProjectConfig
from tau3_evolver.evaluation.tau2_nl_assertions import bind_tau2_nl_assertions
from tau3_evolver.execution.request import ExecutionMode


@dataclass(frozen=True, slots=True)
class Tau2BenchmarkDefinition:
    """Shared static definition for Tau2 protocol-compatible benchmarks."""

    name: str
    default_memory_namespace: str
    task_group: str
    train_split: str = "train"
    test_split: str = "test"

    def prepare(
        self, config: ProjectConfig, mode: ExecutionMode
    ) -> PreparedBenchmark:
        runtime = Tau2Runtime.bind(config.tau2.repo_path)
        split_name = self.train_split if mode is ExecutionMode.TRAIN else self.test_split
        task_loader = runtime.registry.get_tasks_loader(self.name)
        split_loader = runtime.registry.get_task_splits_loader(self.name)
        if split_loader is None:
            raise RuntimeError(f"Tau2 benchmark {self.name!r} has no task split loader")
        split_catalog = split_loader()
        if split_name not in split_catalog:
            raise RuntimeError(
                f"Tau2 benchmark {self.name!r} has no {split_name!r} task split"
            )
        tasks = tuple(task_loader(split_name))
        task_ids = tuple(str(task.id) for task in tasks)
        expected_ids = tuple(str(task_id) for task_id in split_catalog[split_name])
        if task_ids != expected_ids:
            raise RuntimeError(
                f"Tau2 benchmark {self.name!r} task loader order differs from its "
                f"{split_name!r} split catalog"
            )
        split_hash = hashlib.sha256(
            json.dumps(task_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return PreparedBenchmark(
            name=self.name,
            task_type=runtime.task_type,
            task_catalog=tasks,
            task_ids=task_ids,
            split_name=split_name,
            split_hash=split_hash,
            environment_factory=runtime.registry.get_env_constructor(self.name),
            runtime=runtime,
            run_domain=runtime.run_domain,
            text_run_config_type=runtime.text_run_config_type,
            registry=runtime.registry,
            runtime_origin=RuntimeOrigin(
                source_root=runtime.source_root,
                package_version=runtime.package_version,
                git_commit=runtime.git_commit,
            ),
            default_memory_namespace=self.default_memory_namespace,
            task_group=self.task_group,
            evaluator_binding=bind_tau2_nl_assertions,
        )


RETAIL = Tau2BenchmarkDefinition(
    name="retail",
    default_memory_namespace="retail",
    task_group="retail",
)

AIRLINE = Tau2BenchmarkDefinition(
    name="airline",
    default_memory_namespace="airline",
    task_group="airline",
)
