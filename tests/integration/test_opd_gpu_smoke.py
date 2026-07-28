from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pytest


pytestmark = [
    pytest.mark.opd_gpu_smoke,
    pytest.mark.skipif(
        os.environ.get("RUN_OPD_GPU_SMOKE") != "1",
        reason="set RUN_OPD_GPU_SMOKE=1 to run the cached Qwen3.5 OPD smoke",
    ),
]


def _write_one_example_dataset(root: Path, model_revision: str) -> None:
    from tau3_retail_evolver.slow_loop.examples import (
        OPD_DATASET_SCHEMA_VERSION,
        OPD_SAMPLE_UNIT_CONTRACT,
        OPDExample,
    )

    datasets = root / "datasets"
    datasets.mkdir(parents=True)
    example = OPDExample(
        example_id="gpu-smoke-sel-0",
        kind="sel",
        public_input={"request": "Select no memories for this smoke test."},
        privileged_hindsight={"candidate_scores": []},
        response_schema={"type": "object"},
        sampling_contract={},
        provenance={},
    )
    paths: dict[str, Path] = {}
    for kind in ("sel", "act", "write", "maint"):
        path = datasets / f"{kind}.jsonl"
        path.write_text(
            json.dumps(example.model_dump(mode="json"), sort_keys=True) + "\n"
            if kind == "sel"
            else "",
            encoding="utf-8",
        )
        paths[kind] = path
    manifest = {
        "dataset_schema_version": OPD_DATASET_SCHEMA_VERSION,
        "dataset_build_id": "gpu-smoke-dataset",
        "sample_unit_contract": OPD_SAMPLE_UNIT_CONTRACT,
        "policy_lineage": {
            "model_revision": model_revision,
            "adapter_revision": "gpu-smoke-parent",
        },
        "artifacts": {
            f"datasets/{kind}.jsonl": {
                "line_count": 1 if kind == "sel" else 0,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for kind, path in paths.items()
        },
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_cached_qwen35_trains_one_example_with_adapter_only_update(tmp_path: Path) -> None:
    import peft
    import torch

    from tau3_retail_evolver.config import (
        LoraConfig,
        ModelConfig,
        RolloutConfig,
        TrainingConfig,
    )
    from tau3_retail_evolver.models.qwen35 import (
        load_qwen35_tokenizer,
        load_shared_qwen35_policy,
    )
    from tau3_retail_evolver.slow_loop.trainer import OPDTrainer, TrainingRequest

    assert torch.cuda.is_available(), "RUN_OPD_GPU_SMOKE=1 requires CUDA"
    assert torch.cuda.is_bf16_supported(), "RUN_OPD_GPU_SMOKE=1 requires BF16 CUDA"
    model_revision = os.environ.get("OPD_GPU_SMOKE_MODEL_REVISION")
    assert model_revision, "set OPD_GPU_SMOKE_MODEL_REVISION to a cached immutable revision"

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_one_example_dataset(dataset_dir, model_revision)
    model_config = ModelConfig(base_model="Qwen/Qwen3.5-9B")
    training_config = TrainingConfig(
        max_sequence_length=128,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        generation_max_new_tokens=1,
    )
    tokenizer = load_qwen35_tokenizer(
        model_config.base_model,
        revision=model_revision,
        local_files_only=True,
    )
    model = load_shared_qwen35_policy(
        model_config,
        LoraConfig(),
        training_config,
        revision=model_revision,
        local_files_only=True,
    ).to("cuda")
    evidence: dict[str, Any] = {}

    class RecordingAdamW(torch.optim.AdamW):
        def __init__(self, params: Iterable[torch.nn.Parameter], *, lr: float) -> None:
            materialized = list(params)
            evidence["before"] = [parameter.detach().clone() for parameter in materialized]
            super().__init__(materialized, lr=lr)

        def step(self, closure: Any = None) -> Any:
            gradients = [parameter.grad for group in self.param_groups for parameter in group["params"]]
            evidence["lora_grad_norm"] = math.sqrt(
                sum(float(gradient.float().square().sum()) for gradient in gradients if gradient is not None)
            )
            result = super().step(closure)
            evidence["after"] = [
                parameter.detach().clone()
                for group in self.param_groups
                for parameter in group["params"]
            ]
            return result

    result = OPDTrainer(
        model,
        tokenizer,
        training_config,
        RolloutConfig(),
        optimizer_factory=RecordingAdamW,
    ).train(
        TrainingRequest(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            model_revision=model_revision,
            adapter_revision="gpu-smoke-parent",
            kind="sel",
        )
    )

    metrics = [
        json.loads(line)
        for line in (output_dir / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 1
    assert math.isfinite(metrics[0]["metrics"]["forward_kl"])
    assert evidence["lora_grad_norm"] > 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(evidence["before"], evidence["after"], strict=True)
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
    assert not list(result.latest_checkpoint.rglob("pytorch_model*"))
    assert not list(result.latest_checkpoint.rglob("model*.safetensors"))

    manifest = json.loads(
        (result.latest_checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    adapter_path = (result.latest_checkpoint / manifest["adapter_path"]).resolve()
    trained_adapter_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in peft.get_peft_model_state_dict(
            model, adapter_name="shared_policy"
        ).items()
    }
    assert trained_adapter_state
    del model
    torch.cuda.empty_cache()
    reloaded = load_shared_qwen35_policy(
        model_config,
        LoraConfig(),
        training_config,
        revision=model_revision,
        adapter_path=adapter_path,
        local_files_only=True,
    )
    assert len(reloaded.peft_config) == 1
    reloaded_adapter_state = {
        name: tensor.detach().cpu()
        for name, tensor in peft.get_peft_model_state_dict(
            reloaded, adapter_name="shared_policy"
        ).items()
    }
    assert reloaded_adapter_state.keys() == trained_adapter_state.keys()
    assert all(
        torch.equal(trained_adapter_state[name], reloaded_adapter_state[name])
        for name in trained_adapter_state
    )
