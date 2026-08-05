import os
from pathlib import Path

import pytest

from tau3_evolver.benchmarks import benchmark_registry
from tau3_evolver.config import load_config
from tau3_evolver.execution.request import ExecutionMode


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.tau2_integration
@pytest.mark.skipif(
    os.environ.get("RUN_TAU2_INTEGRATION") != "1",
    reason="set RUN_TAU2_INTEGRATION=1 to run benchmark preflight",
)
@pytest.mark.parametrize("benchmark", ("retail", "airline"))
def test_real_tau2_benchmark_preflight(benchmark: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config = config.model_copy(
        update={
            "tau2": config.tau2.model_copy(
                update={"repo_path": PROJECT_ROOT / config.tau2.repo_path}
            )
        }
    )

    prepared = benchmark_registry.resolve(benchmark).prepare(
        config, ExecutionMode.TRAIN
    )

    assert prepared.name == benchmark
    assert prepared.split_name == "train"
    assert prepared.task_ids
    assert prepared.runtime_origin.source_root.is_dir()
