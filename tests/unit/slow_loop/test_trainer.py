from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tau3_retail_evolver.config import RolloutConfig, TrainingConfig
from tau3_retail_evolver.slow_loop.alignment import render_public_prompt
from tau3_retail_evolver.slow_loop.examples import OPDExample
from tau3_retail_evolver.slow_loop.opd_step import OPDStepResult
from tau3_retail_evolver.slow_loop import trainer as trainer_module
from tau3_retail_evolver.slow_loop.trainer import OPDTrainer, TrainingRequest


KINDS = ("sel", "act", "write", "maint")


class ToyTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        assert add_special_tokens is False
        ids = [ord(character) % 17 + 1 for character in text]
        if return_tensors == "pt":
            input_ids = torch.tensor([ids], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        return {"input_ids": ids}


class ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.lora_weight = nn.Parameter(torch.tensor(0.5))
        self.generate_calls: list[dict[str, Any]] = []
        self.fail_on_generate_call: int | None = None
        self.empty_generation = False

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        call_number = len(self.generate_calls) + 1
        self.generate_calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "training": self.training,
                "grad_enabled": torch.is_grad_enabled(),
                **kwargs,
            }
        )
        if self.fail_on_generate_call == call_number:
            raise RuntimeError("injected generation failure")
        if self.empty_generation:
            return input_ids.detach().clone()
        response = torch.tensor(
            [[(call_number % 11) + 1]], dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat((input_ids, response), dim=1)

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        assert input_ids.device == self.lora_weight.device
        assert attention_mask.device == self.lora_weight.device
        vocabulary = torch.arange(19, device=input_ids.device, dtype=torch.float32)
        centers = input_ids.to(torch.float32).unsqueeze(-1).remainder(19)
        logits = -(vocabulary - centers).square() / 20
        logits = logits + self.lora_weight * (vocabulary / 19)
        return SimpleNamespace(logits=logits)


class StochasticToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.rand(())
        self.base_weight = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.lora_weight = nn.Parameter(torch.tensor(0.5))
        self.generate_calls = 0
        self.fail_on_generate_call: int | None = None

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        self.generate_calls += 1
        python_draw = random.randint(1, 18)
        torch_draw = int(torch.randint(1, 19, ()).item())
        if self.fail_on_generate_call == self.generate_calls:
            raise RuntimeError("injected stochastic generation failure")
        response_id = (python_draw + torch_draw) % 18 + 1
        response = torch.tensor(
            [[response_id]], dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat((input_ids, response), dim=1)

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        vocabulary = torch.arange(19, device=input_ids.device, dtype=torch.float32)
        centers = input_ids.to(torch.float32).unsqueeze(-1).remainder(19)
        logits = -(vocabulary - centers).square() / 20
        features = (vocabulary / 19).expand_as(logits)
        features = torch.nn.functional.dropout(
            features,
            p=0.4,
            training=self.training,
        )
        return SimpleNamespace(logits=logits + self.lora_weight * features)


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params: Iterable[nn.Parameter], *, lr: float) -> None:
        materialized = list(params)
        self.received_parameters = materialized
        self.step_count = 0
        super().__init__(materialized, lr=lr)

    def step(self, closure: Any = None) -> Any:
        self.step_count += 1
        return super().step(closure)


class OptimizerFactory:
    def __init__(self) -> None:
        self.optimizers: list[RecordingSGD] = []

    def __call__(self, params: Iterable[nn.Parameter], *, lr: float) -> RecordingSGD:
        optimizer = RecordingSGD(params, lr=lr)
        self.optimizers.append(optimizer)
        return optimizer


class ToyCheckpointSaver:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, model: ToyPolicy, destination: Path) -> Path:
        self.calls.append(destination)
        adapter = destination / "shared_policy"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (adapter / "stage6_adapter_contract.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "requested_target_modules": "all-linear",
                    "resolved_target_modules": ["q_proj"],
                    "options": {
                        "alpha_pattern": {},
                        "bias": "none",
                        "init_lora_weights": True,
                        "lora_alpha": 64,
                        "lora_dropout": 0.05,
                        "modules_to_save": None,
                        "r": 32,
                        "rank_pattern": {},
                        "task_type": "CAUSAL_LM",
                        "use_dora": False,
                        "use_rslora": False,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save({"lora_weight": model.lora_weight.detach()}, adapter / "adapter_model.bin")
        return adapter


def _example(kind: str, index: int) -> OPDExample:
    return OPDExample(
        example_id=f"{kind}-{index}",
        kind=kind,
        public_input={"visible": f"{kind}-{index}"},
        privileged_hindsight={"secret": f"hindsight-{kind}-{index}"},
        response_schema={"type": "object"},
        sampling_contract={},
        provenance={},
    )


def _write_dataset(
    root: Path,
    counts: dict[str, int],
    *,
    model_revision: str = "model-a",
    adapter_revision: str | None = "adapter-a",
) -> dict[str, list[OPDExample]]:
    datasets = root / "datasets"
    datasets.mkdir(parents=True)
    examples: dict[str, list[OPDExample]] = {}
    for kind in KINDS:
        examples[kind] = [_example(kind, index) for index in range(counts.get(kind, 0))]
        (datasets / f"{kind}.jsonl").write_text(
            "".join(
                json.dumps(example.model_dump(mode="json"), sort_keys=True) + "\n"
                for example in examples[kind]
            ),
            encoding="utf-8",
        )
    artifacts = {
        f"datasets/{kind}.jsonl": {
            "line_count": len(examples[kind]),
            "sha256": hashlib.sha256((datasets / f"{kind}.jsonl").read_bytes()).hexdigest(),
        }
        for kind in KINDS
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_schema_version": 1,
                "dataset_build_id": "dataset-a",
                "policy_lineage": {
                    "model_revision": model_revision,
                    "adapter_revision": adapter_revision,
                },
                "artifacts": artifacts,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return examples


def _config(**overrides: Any) -> TrainingConfig:
    values = {
        "max_sequence_length": 128,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1,
        "generation_max_new_tokens": 3,
        "learning_rate": 0.1,
        **overrides,
    }
    return TrainingConfig(**values)


def _request(
    dataset: Path,
    output: Path,
    *,
    resume_from: Path | None = None,
    loaded_adapter_path: Path | None = None,
) -> TrainingRequest:
    return TrainingRequest(
        dataset_dir=dataset,
        output_dir=output,
        model_revision="model-a",
        adapter_revision="adapter-a",
        resume_from=resume_from,
        loaded_adapter_path=loaded_adapter_path,
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _checkpoint_adapter_path(checkpoint: Path) -> Path:
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    return (checkpoint / manifest["adapter_path"]).resolve()


def _load_toy_policy(adapter_path: Path) -> ToyPolicy:
    state = torch.load(adapter_path / "adapter_model.bin", weights_only=True)
    model = ToyPolicy()
    model.lora_weight.data.copy_(state["lora_weight"])
    return model


def _load_stochastic_policy(adapter_path: Path) -> StochasticToyPolicy:
    state = torch.load(adapter_path / "adapter_model.bin", weights_only=True)
    model = StochasticToyPolicy()
    model.lora_weight.data.copy_(state["lora_weight"])
    return model


def test_online_generation_uses_only_public_prompts_and_balances_kinds(tmp_path: Path) -> None:
    examples = _write_dataset(
        tmp_path / "dataset", {"sel": 1, "act": 2, "write": 1, "maint": 1}
    )
    model = ToyPolicy().train()
    saver = ToyCheckpointSaver()

    result = OPDTrainer(
        model,
        ToyTokenizer(),
        _config(),
        RolloutConfig(temperature=0.7, top_p=0.8),
        checkpoint_saver=saver,
        optimizer_factory=OptimizerFactory(),
    ).train(_request(tmp_path / "dataset", tmp_path / "output"))

    generations = _rows(result.output_dir / "training_generations.jsonl")
    assert [row["kind"] for row in generations] == [*KINDS, *KINDS]
    assert [row["example_id"] for row in generations] == [
        "sel-0",
        "act-0",
        "write-0",
        "maint-0",
        "sel-0",
        "act-1",
        "write-0",
        "maint-0",
    ]
    assert len(model.generate_calls) == 8
    tokenizer = ToyTokenizer()
    scheduled = [
        examples[row["kind"]][int(row["example_id"].rsplit("-", 1)[1])]
        for row in generations
    ]
    for call, example in zip(model.generate_calls, scheduled, strict=True):
        expected = tokenizer(
            render_public_prompt(example), add_special_tokens=False, return_tensors="pt"
        )["input_ids"][:, -125:]
        assert call["input_ids"].tolist() == expected.tolist()
        assert call["training"] is False
        assert call["grad_enabled"] is False
        assert call["max_new_tokens"] == 3
        assert call["temperature"] == 0.7
        assert call["top_p"] == 0.8
    assert model.training is True
    assert all(row["response_ids"] for row in generations)


def test_generation_caps_new_tokens_and_left_truncates_to_context(tmp_path: Path) -> None:
    examples = _write_dataset(tmp_path / "dataset", {"sel": 1})
    model = ToyPolicy()

    OPDTrainer(
        model,
        ToyTokenizer(),
        _config(generation_max_new_tokens=200),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(_request(tmp_path / "dataset", tmp_path / "output"))

    call = model.generate_calls[0]
    full_prompt_ids = ToyTokenizer()(
        render_public_prompt(examples["sel"][0]),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    assert call["max_new_tokens"] == 127
    assert call["input_ids"].shape[1] == 1
    assert call["input_ids"].tolist() == full_prompt_ids[:, -1:].tolist()
    assert call["attention_mask"].shape == call["input_ids"].shape
    assert call["input_ids"].shape[1] + call["max_new_tokens"] == 128


def test_effective_batch_and_final_partial_average_each_optimizer_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 5})
    model = ToyPolicy()
    optimizer_factory = OptimizerFactory()

    def constant_gradient_step(model: ToyPolicy, batch: Any) -> OPDStepResult:
        del batch
        loss = model.lora_weight
        return OPDStepResult(loss=loss, metrics={"forward_kl": loss.detach()})

    monkeypatch.setattr(
        "tau3_retail_evolver.slow_loop.trainer.shared_policy_opd_step",
        constant_gradient_step,
    )
    result = OPDTrainer(
        model,
        ToyTokenizer(),
        _config(per_device_batch_size=2, gradient_accumulation_steps=2),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=optimizer_factory,
    ).train(_request(tmp_path / "dataset", tmp_path / "output"))

    optimizer = optimizer_factory.optimizers[0]
    assert optimizer.received_parameters == [model.lora_weight]
    assert optimizer.step_count == result.optimizer_steps == 2
    assert model.lora_weight.item() == pytest.approx(0.3)
    metrics = _rows(result.output_dir / "training_metrics.jsonl")
    assert [row["optimizer_window_size"] for row in metrics] == [4, 4, 4, 4, 1]


@pytest.mark.parametrize(
    ("model_revision", "adapter_revision", "match"),
    (("model-b", "adapter-a", "model_revision"), ("model-a", "adapter-b", "adapter_revision")),
)
def test_rejects_source_lineage_mismatch_before_generation(
    tmp_path: Path,
    model_revision: str,
    adapter_revision: str,
    match: str,
) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 1})
    model = ToyPolicy()

    with pytest.raises(ValueError, match=match):
        OPDTrainer(
            model,
            ToyTokenizer(),
            _config(),
            RolloutConfig(),
            checkpoint_saver=ToyCheckpointSaver(),
            optimizer_factory=OptimizerFactory(),
        ).train(
            TrainingRequest(
                dataset_dir=tmp_path / "dataset",
                output_dir=tmp_path / "output",
                model_revision=model_revision,
                adapter_revision=adapter_revision,
            )
        )

    assert model.generate_calls == []


def test_checkpoint_contains_adapter_optimizer_and_atomic_json_manifests_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 2})
    output = tmp_path / "output"
    stale_temp = output / "checkpoints" / ".step-00000001.tmp-stale"
    stale_temp.mkdir(parents=True)
    (stale_temp / "incomplete").write_text("stale", encoding="utf-8")
    saver = ToyCheckpointSaver()
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("tau3_retail_evolver.slow_loop.trainer.os.replace", record_replace)
    result = OPDTrainer(
        ToyPolicy(),
        ToyTokenizer(),
        _config(),
        RolloutConfig(),
        checkpoint_saver=saver,
        optimizer_factory=OptimizerFactory(),
    ).train(_request(tmp_path / "dataset", output))

    checkpoint = result.latest_checkpoint
    checkpoint_manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    temporary_checkpoint = saver.calls[0].parent
    assert temporary_checkpoint.parent == checkpoint.parent
    assert temporary_checkpoint.name.startswith(".step-00000001.tmp-")
    assert temporary_checkpoint != stale_temp
    assert not temporary_checkpoint.exists()
    assert stale_temp.is_dir()
    assert (checkpoint / "adapter" / "shared_policy" / "adapter_config.json").is_file()
    assert (
        checkpoint
        / "adapter"
        / "shared_policy"
        / "stage6_adapter_contract.json"
    ).is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "rng_state.pt").is_file()
    assert (checkpoint / "checkpoint_manifest.json").is_file()
    assert checkpoint_manifest["schedule_sha256"] == checkpoint_manifest[
        "schedule_fingerprint"
    ]
    assert (result.output_dir / "training_manifest.json").is_file()
    assert not list(result.output_dir.rglob("pytorch_model*"))
    replaced_names = [destination.name for _, destination in replacements]
    assert "checkpoint_manifest.json" in replaced_names
    assert "training_manifest.json" in replaced_names
    assert any(
        source == temporary_checkpoint and destination == checkpoint
        for source, destination in replacements
    )


def test_failed_example_is_not_appended_and_empty_response_is_rejected(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 1})
    model = ToyPolicy()
    model.empty_generation = True

    with pytest.raises(ValueError, match="nonempty response suffix"):
        OPDTrainer(
            model,
            ToyTokenizer(),
            _config(),
            RolloutConfig(),
            checkpoint_saver=ToyCheckpointSaver(),
            optimizer_factory=OptimizerFactory(),
        ).train(_request(tmp_path / "dataset", tmp_path / "output"))

    generations = tmp_path / "output" / "training_generations.jsonl"
    metrics = tmp_path / "output" / "training_metrics.jsonl"
    assert not generations.exists() or generations.read_text(encoding="utf-8") == ""
    assert not metrics.exists() or metrics.read_text(encoding="utf-8") == ""


def test_resume_uses_latest_completed_step_and_removes_uncommitted_rows(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 6})
    output = tmp_path / "output"
    model = ToyPolicy()
    model.fail_on_generate_call = 6
    trainer = OPDTrainer(
        model,
        ToyTokenizer(),
        _config(per_device_batch_size=2, gradient_accumulation_steps=2),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    )

    with pytest.raises(RuntimeError, match="injected generation failure"):
        trainer.train(_request(tmp_path / "dataset", output))

    checkpoint = output / "checkpoints" / "step-00000001"
    assert checkpoint.is_dir()
    assert [row["sequence_index"] for row in _rows(output / "training_generations.jsonl")] == [
        0,
        1,
        2,
        3,
        4,
    ]
    with (output / "training_generations.jsonl").open("ab") as destination:
        destination.write(b'{"torn":')
    with (output / "training_metrics.jsonl").open("ab") as destination:
        destination.write(b'\xff{"torn":')

    adapter_path = _checkpoint_adapter_path(checkpoint)
    resumed_model = _load_toy_policy(adapter_path)
    result = OPDTrainer(
        resumed_model,
        ToyTokenizer(),
        _config(per_device_batch_size=2, gradient_accumulation_steps=2),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(
        _request(
            tmp_path / "dataset",
            output,
            resume_from=checkpoint,
            loaded_adapter_path=adapter_path,
        )
    )

    generations = _rows(result.output_dir / "training_generations.jsonl")
    assert [row["sequence_index"] for row in generations] == list(range(6))
    assert len({row["sequence_index"] for row in generations}) == 6
    assert result.completed_examples == 6
    assert result.optimizer_steps == 2


def test_stochastic_resume_matches_uninterrupted_training_with_fresh_loaded_model(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, {"sel": 4})
    config = _config(
        seed=1729,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    )

    uninterrupted_model = StochasticToyPolicy()
    uninterrupted = OPDTrainer(
        uninterrupted_model,
        ToyTokenizer(),
        config,
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(_request(dataset, tmp_path / "uninterrupted"))

    interrupted_model = StochasticToyPolicy()
    interrupted_model.fail_on_generate_call = 3
    interrupted_output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="stochastic generation failure"):
        OPDTrainer(
            interrupted_model,
            ToyTokenizer(),
            config,
            RolloutConfig(),
            checkpoint_saver=ToyCheckpointSaver(),
            optimizer_factory=OptimizerFactory(),
        ).train(_request(dataset, interrupted_output))

    checkpoint = interrupted_output / "checkpoints" / "step-00000001"
    adapter_path = _checkpoint_adapter_path(checkpoint)
    random.seed(999)
    torch.manual_seed(999)
    resumed_model = _load_stochastic_policy(adapter_path)
    resumed = OPDTrainer(
        resumed_model,
        ToyTokenizer(),
        config,
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(
        _request(
            dataset,
            interrupted_output,
            resume_from=checkpoint,
            loaded_adapter_path=adapter_path,
        )
    )

    assert [row["response_ids"] for row in _rows(
        uninterrupted.output_dir / "training_generations.jsonl"
    )] == [row["response_ids"] for row in _rows(
        resumed.output_dir / "training_generations.jsonl"
    )]
    torch.testing.assert_close(
        resumed_model.lora_weight,
        uninterrupted_model.lora_weight,
        rtol=0,
        atol=0,
    )


def test_restore_rng_state_explicitly_trusts_the_project_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "rng_state.pt"
    path.write_bytes(b"project-owned rng state")
    calls: list[tuple[Path, dict[str, Any]]] = []
    state = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }

    def load(candidate: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((candidate, kwargs))
        return state

    monkeypatch.setattr(trainer_module.torch, "load", load)
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: False)

    trainer_module._restore_rng_state(path)

    assert calls == [(path, {"map_location": "cpu", "weights_only": False})]


def test_restore_rng_state_validates_python_state_before_setting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "rng_state.pt"
    path.write_bytes(b"project-owned rng state")
    state = {
        "python_random_state": (3, "not-an-internal-state", None),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    setstate_calls: list[Any] = []
    monkeypatch.setattr(trainer_module.torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(trainer_module.random, "setstate", setstate_calls.append)
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="Python RNG state"):
        trainer_module._restore_rng_state(path)

    assert setstate_calls == []


@pytest.mark.parametrize(
    "log_name", ("training_generations.jsonl", "training_metrics.jsonl")
)
def test_resume_rejects_committed_log_metadata_mismatch(
    tmp_path: Path, log_name: str
) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 2})
    output = tmp_path / "output"
    first = OPDTrainer(
        ToyPolicy(),
        ToyTokenizer(),
        _config(),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(_request(tmp_path / "dataset", output))
    adapter_path = _checkpoint_adapter_path(first.latest_checkpoint)
    rows = _rows(output / log_name)
    rows[0]["example_id"] = "wrong-example"
    (output / log_name).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="metadata"):
        OPDTrainer(
            _load_toy_policy(adapter_path),
            ToyTokenizer(),
            _config(),
            RolloutConfig(),
            checkpoint_saver=ToyCheckpointSaver(),
            optimizer_factory=OptimizerFactory(),
        ).train(
            _request(
                tmp_path / "dataset",
                output,
                resume_from=first.latest_checkpoint,
                loaded_adapter_path=adapter_path,
            )
        )


def test_resume_rejects_stage5_artifact_schedule_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, {"sel": 2})
    output = tmp_path / "output"
    first = OPDTrainer(
        ToyPolicy(),
        ToyTokenizer(),
        _config(),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(_request(dataset, output))
    adapter_path = _checkpoint_adapter_path(first.latest_checkpoint)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["datasets/sel.jsonl"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    optimizer_factory = OptimizerFactory()

    with pytest.raises(ValueError, match="schedule fingerprint"):
        OPDTrainer(
            _load_toy_policy(adapter_path),
            ToyTokenizer(),
            _config(),
            RolloutConfig(),
            checkpoint_saver=ToyCheckpointSaver(),
            optimizer_factory=optimizer_factory,
        ).train(
            _request(
                dataset,
                output,
                resume_from=first.latest_checkpoint,
                loaded_adapter_path=adapter_path,
            )
        )

    assert optimizer_factory.optimizers == []


def test_resume_rejects_training_config_mismatch(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 2})
    output = tmp_path / "output"
    first = OPDTrainer(
        ToyPolicy(),
        ToyTokenizer(),
        _config(),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(_request(tmp_path / "dataset", output))
    adapter_path = _checkpoint_adapter_path(first.latest_checkpoint)

    with pytest.raises(ValueError, match="training config"):
        OPDTrainer(
            _load_toy_policy(adapter_path),
            ToyTokenizer(),
            _config(learning_rate=0.2),
            RolloutConfig(),
            checkpoint_saver=ToyCheckpointSaver(),
            optimizer_factory=OptimizerFactory(),
        ).train(
            _request(
                tmp_path / "dataset",
                output,
                resume_from=first.latest_checkpoint,
                loaded_adapter_path=adapter_path,
            )
        )


def test_resume_republishes_a_missing_final_training_manifest(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 2})
    output = tmp_path / "output"
    model = ToyPolicy()
    first = OPDTrainer(
        model,
        ToyTokenizer(),
        _config(),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(_request(tmp_path / "dataset", output))
    (output / "training_manifest.json").unlink()
    adapter_path = _checkpoint_adapter_path(first.latest_checkpoint)

    resumed = OPDTrainer(
        _load_toy_policy(adapter_path),
        ToyTokenizer(),
        _config(),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(
        _request(
            tmp_path / "dataset",
            output,
            resume_from=first.latest_checkpoint,
            loaded_adapter_path=adapter_path,
        )
    )

    assert resumed.manifest["status"] == "complete"
    assert resumed.manifest["completed_examples"] == 2
    assert resumed.manifest["latest_checkpoint"] == "checkpoints/step-00000001"


@pytest.mark.parametrize("loaded_path", (None, Path("wrong-adapter")))
def test_resume_rejects_missing_or_wrong_loaded_adapter_before_optimizer_creation(
    tmp_path: Path, loaded_path: Path | None
) -> None:
    _write_dataset(tmp_path / "dataset", {"sel": 2})
    output = tmp_path / "output"
    first = OPDTrainer(
        ToyPolicy(),
        ToyTokenizer(),
        _config(),
        RolloutConfig(),
        checkpoint_saver=ToyCheckpointSaver(),
        optimizer_factory=OptimizerFactory(),
    ).train(_request(tmp_path / "dataset", output))
    expected_adapter = _checkpoint_adapter_path(first.latest_checkpoint)
    optimizer_factory = OptimizerFactory()

    with pytest.raises(ValueError, match="loaded_adapter_path"):
        OPDTrainer(
            _load_toy_policy(expected_adapter),
            ToyTokenizer(),
            _config(),
            RolloutConfig(),
            checkpoint_saver=ToyCheckpointSaver(),
            optimizer_factory=optimizer_factory,
        ).train(
            _request(
                tmp_path / "dataset",
                output,
                resume_from=first.latest_checkpoint,
                loaded_adapter_path=loaded_path,
            )
        )

    assert optimizer_factory.optimizers == []
