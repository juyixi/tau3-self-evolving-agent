from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import torch
from torch import Tensor, nn

from tau3_retail_evolver.config import RolloutConfig, TrainingConfig
from tau3_retail_evolver.io.jsonl import JsonlWriter, iter_jsonl_objects
from tau3_retail_evolver.models.lora import save_adapter_checkpoint
from tau3_retail_evolver.slow_loop.alignment import (
    AlignedOPDBatch,
    build_aligned_batch,
    render_public_prompt,
)
from tau3_retail_evolver.slow_loop.examples import OPDExample
from tau3_retail_evolver.slow_loop.opd_step import shared_policy_opd_step


_KINDS = ("sel", "act", "write", "maint")
_MANIFEST_SCHEMA_VERSION = 1
CheckpointSaver = Callable[[Any, Path], Path]
OptimizerFactory = Callable[..., torch.optim.Optimizer]


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    dataset_dir: Path
    output_dir: Path
    model_revision: str
    adapter_revision: str | None
    resume_from: Path | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_dir: Path
    latest_checkpoint: Path
    completed_examples: int
    optimizer_steps: int
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ScheduledExample:
    sequence_index: int
    epoch: int
    example: OPDExample


class OPDTrainer:
    """Train the current shared LoRA policy from an online Stage 5 OPD dataset."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        training_config: TrainingConfig,
        rollout_config: RolloutConfig,
        *,
        checkpoint_saver: CheckpointSaver = save_adapter_checkpoint,
        optimizer_factory: OptimizerFactory = torch.optim.AdamW,
    ) -> None:
        if not isinstance(training_config, TrainingConfig):
            raise TypeError("training_config must be a TrainingConfig")
        if not isinstance(rollout_config, RolloutConfig):
            raise TypeError("rollout_config must be a RolloutConfig")
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.rollout_config = rollout_config
        self.checkpoint_saver = checkpoint_saver
        self.optimizer_factory = optimizer_factory

    def train(self, request: TrainingRequest) -> TrainingResult:
        if not isinstance(request, TrainingRequest):
            raise TypeError("request must be a TrainingRequest")
        _validate_revision(request.model_revision, "model_revision")
        if request.adapter_revision is not None:
            _validate_revision(request.adapter_revision, "adapter_revision")

        dataset_dir = Path(request.dataset_dir).resolve()
        output_dir = Path(request.output_dir).resolve()
        dataset_manifest = _read_json_object(dataset_dir / "dataset_manifest.json")
        source_lineage = _validate_source_lineage(dataset_manifest, request)
        schedule = _build_schedule(
            _load_examples(dataset_dir), self.training_config.num_train_epochs
        )
        if not schedule:
            raise ValueError("Stage 5 dataset contains no OPD examples")

        output_dir.mkdir(parents=True, exist_ok=True)
        generation_path = output_dir / "training_generations.jsonl"
        metric_path = output_dir / "training_metrics.jsonl"
        trainable_parameters = _trainable_lora_parameters(self.model)
        optimizer = self.optimizer_factory(
            trainable_parameters, lr=self.training_config.learning_rate
        )
        optimizer.zero_grad(set_to_none=True)

        completed_examples = 0
        optimizer_steps = 0
        latest_checkpoint: Path | None = None
        if request.resume_from is None:
            _require_fresh_output(output_dir)
        else:
            latest_checkpoint = Path(request.resume_from).resolve()
            resume_manifest = _read_json_object(
                latest_checkpoint / "checkpoint_manifest.json"
            )
            completed_examples, optimizer_steps = self._validate_resume(
                resume_manifest,
                request=request,
                dataset_manifest=dataset_manifest,
                source_lineage=source_lineage,
                total_examples=len(schedule),
            )
            _load_optimizer_state(optimizer, latest_checkpoint / "optimizer.pt")
            optimizer.zero_grad(set_to_none=True)
            _restore_committed_rows(generation_path, completed_examples)
            _restore_committed_rows(metric_path, completed_examples)
            if completed_examples == len(schedule):
                _atomic_write_json(
                    output_dir / "training_manifest.json",
                    {
                        **resume_manifest,
                        "latest_checkpoint": latest_checkpoint.relative_to(
                            output_dir
                        ).as_posix(),
                        "status": "complete",
                    },
                )

        self.model.train()
        effective_batch_size = (
            self.training_config.per_device_batch_size
            * self.training_config.gradient_accumulation_steps
        )
        device = _model_device(self.model)
        generation_writer = JsonlWriter(generation_path)
        metric_writer = JsonlWriter(metric_path)

        while completed_examples < len(schedule):
            window_end = min(
                completed_examples + effective_batch_size,
                len(schedule),
            )
            window = schedule[completed_examples:window_end]
            window_size = len(window)
            for scheduled in window:
                response_ids = _generate_student_response(
                    self.model,
                    self.tokenizer,
                    scheduled.example,
                    self.training_config,
                    self.rollout_config,
                    device=device,
                )
                batch = build_aligned_batch(
                    scheduled.example,
                    self.tokenizer,
                    response_ids=response_ids,
                    max_length=self.training_config.max_sequence_length,
                )
                result = shared_policy_opd_step(
                    self.model, _batch_to_device(batch, device)
                )
                (result.loss / window_size).backward()
                generation_writer.append(
                    {
                        "epoch": scheduled.epoch,
                        "example_id": scheduled.example.example_id,
                        "kind": scheduled.example.kind,
                        "response_ids": list(response_ids),
                        "sequence_index": scheduled.sequence_index,
                    }
                )
                metric_writer.append(
                    {
                        "epoch": scheduled.epoch,
                        "example_id": scheduled.example.example_id,
                        "kind": scheduled.example.kind,
                        "loss": _scalar(result.loss),
                        "metrics": {
                            name: _scalar(value) for name, value in result.metrics.items()
                        },
                        "optimizer_window_size": window_size,
                        "sequence_index": scheduled.sequence_index,
                    }
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_examples = window_end
            optimizer_steps += 1
            latest_checkpoint, checkpoint_manifest = self._save_checkpoint(
                output_dir,
                optimizer,
                request=request,
                dataset_manifest=dataset_manifest,
                source_lineage=source_lineage,
                completed_examples=completed_examples,
                optimizer_steps=optimizer_steps,
                total_examples=len(schedule),
            )
            training_manifest = {
                **checkpoint_manifest,
                "latest_checkpoint": latest_checkpoint.relative_to(output_dir).as_posix(),
                "status": (
                    "complete" if completed_examples == len(schedule) else "in_progress"
                ),
            }
            _atomic_write_json(output_dir / "training_manifest.json", training_manifest)

        if latest_checkpoint is None:
            raise RuntimeError("training completed without an adapter checkpoint")
        manifest = _read_json_object(output_dir / "training_manifest.json")
        return TrainingResult(
            output_dir=output_dir,
            latest_checkpoint=latest_checkpoint,
            completed_examples=completed_examples,
            optimizer_steps=optimizer_steps,
            manifest=manifest,
        )

    def _validate_resume(
        self,
        manifest: Mapping[str, Any],
        *,
        request: TrainingRequest,
        dataset_manifest: Mapping[str, Any],
        source_lineage: Mapping[str, Any],
        total_examples: int,
    ) -> tuple[int, int]:
        if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise ValueError("resume checkpoint schema mismatch")
        if manifest.get("dataset_build_id") != dataset_manifest.get("dataset_build_id"):
            raise ValueError("resume dataset lineage mismatch")
        if manifest.get("source_lineage") != source_lineage:
            raise ValueError("resume source lineage mismatch")
        if manifest.get("trainer_start") != _trainer_start(request):
            raise ValueError("resume trainer lineage mismatch")
        if manifest.get("training_config") != self.training_config.model_dump(mode="json"):
            raise ValueError("resume training config mismatch")
        if manifest.get("rollout_config") != self.rollout_config.model_dump(mode="json"):
            raise ValueError("resume rollout config mismatch")
        if manifest.get("total_examples") != total_examples:
            raise ValueError("resume sampling schedule mismatch")
        completed = manifest.get("completed_examples")
        steps = manifest.get("optimizer_steps")
        if type(completed) is not int or not 0 <= completed <= total_examples:
            raise ValueError("resume completed example count is invalid")
        if type(steps) is not int or steps < 1:
            raise ValueError("resume optimizer step count is invalid")
        return completed, steps

    def _save_checkpoint(
        self,
        output_dir: Path,
        optimizer: torch.optim.Optimizer,
        *,
        request: TrainingRequest,
        dataset_manifest: Mapping[str, Any],
        source_lineage: Mapping[str, Any],
        completed_examples: int,
        optimizer_steps: int,
        total_examples: int,
    ) -> tuple[Path, dict[str, Any]]:
        checkpoint = output_dir / "checkpoints" / f"step-{optimizer_steps:08d}"
        if checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
        checkpoint.mkdir(parents=True)
        try:
            adapter_path = Path(
                self.checkpoint_saver(self.model, checkpoint / "adapter")
            ).resolve()
            try:
                adapter_relative_path = adapter_path.relative_to(checkpoint)
            except ValueError as error:
                raise ValueError("checkpoint saver returned a path outside the checkpoint") from error
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
            manifest = {
                "adapter_path": adapter_relative_path.as_posix(),
                "adapter_revision": f"opd-step-{optimizer_steps:08d}",
                "completed_examples": completed_examples,
                "dataset_build_id": dataset_manifest.get("dataset_build_id"),
                "optimizer_steps": optimizer_steps,
                "rollout_config": self.rollout_config.model_dump(mode="json"),
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "source_lineage": dict(source_lineage),
                "status": "checkpoint",
                "total_examples": total_examples,
                "trainer_start": _trainer_start(request),
                "training_config": self.training_config.model_dump(mode="json"),
            }
            _atomic_write_json(checkpoint / "checkpoint_manifest.json", manifest)
        except BaseException:
            shutil.rmtree(checkpoint, ignore_errors=True)
            raise
        return checkpoint, manifest


def _load_examples(dataset_dir: Path) -> dict[str, tuple[OPDExample, ...]]:
    examples: dict[str, tuple[OPDExample, ...]] = {}
    for kind in _KINDS:
        path = dataset_dir / "datasets" / f"{kind}.jsonl"
        rows = tuple(OPDExample.model_validate(row) for row in iter_jsonl_objects(path))
        if any(example.kind != kind for example in rows):
            raise ValueError(f"dataset kind mismatch in {path}")
        examples[kind] = rows
    return examples


def _build_schedule(
    examples: Mapping[str, Sequence[OPDExample]], num_epochs: int
) -> tuple[_ScheduledExample, ...]:
    active_kinds = [kind for kind in _KINDS if examples.get(kind)]
    if not active_kinds:
        return ()
    rounds = max(len(examples[kind]) for kind in active_kinds)
    schedule: list[_ScheduledExample] = []
    for epoch in range(num_epochs):
        for round_index in range(rounds):
            for kind in active_kinds:
                kind_examples = examples[kind]
                schedule.append(
                    _ScheduledExample(
                        sequence_index=len(schedule),
                        epoch=epoch,
                        example=kind_examples[round_index % len(kind_examples)],
                    )
                )
    return tuple(schedule)


def _generate_student_response(
    model: nn.Module,
    tokenizer: Any,
    example: OPDExample,
    training_config: TrainingConfig,
    rollout_config: RolloutConfig,
    *,
    device: torch.device,
) -> tuple[int, ...]:
    prompt = render_public_prompt(example)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError) as error:
        raise TypeError("tokenizer output must contain input_ids") from error
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise TypeError("generation input_ids must contain one tensor sequence")
    if input_ids.shape[1] == 0:
        raise ValueError("public prompt must encode to at least one token")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if not isinstance(attention_mask, Tensor) or attention_mask.shape != input_ids.shape:
        raise TypeError("generation attention_mask must match input_ids")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    original_training = model.training
    try:
        with torch.no_grad():
            model.eval()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=training_config.generation_max_new_tokens,
                temperature=rollout_config.temperature,
                top_p=rollout_config.top_p,
            )
    finally:
        model.train(original_training)
    if not isinstance(generated, Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
        raise TypeError("model.generate must return one tensor sequence")
    response = generated[0, input_ids.shape[1] :]
    if response.numel() == 0:
        raise ValueError("student generation must produce a nonempty response suffix")
    return tuple(int(token_id) for token_id in response.detach().cpu().tolist())


def _batch_to_device(batch: AlignedOPDBatch, device: torch.device) -> AlignedOPDBatch:
    return AlignedOPDBatch(
        **{field.name: getattr(batch, field.name).to(device) for field in fields(batch)}
    )


def _trainable_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    if not callable(getattr(model, "named_parameters", None)):
        raise TypeError("model must expose named_parameters()")
    trainable: list[nn.Parameter] = []
    non_lora: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable.append(parameter)
        if "lora_" not in name:
            non_lora.append(name)
    if not trainable:
        raise ValueError("shared policy has no trainable LoRA parameters")
    if non_lora:
        raise ValueError(f"non-LoRA parameters are trainable: {', '.join(non_lora)}")
    return trainable


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters") from error


def _validate_source_lineage(
    manifest: Mapping[str, Any], request: TrainingRequest
) -> dict[str, Any]:
    lineage = manifest.get("policy_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("dataset manifest policy_lineage is missing")
    if lineage.get("model_revision") != request.model_revision:
        raise ValueError("dataset source model_revision does not match trainer start")
    if lineage.get("adapter_revision") != request.adapter_revision:
        raise ValueError("dataset source adapter_revision does not match trainer start")
    return {
        "adapter_revision": request.adapter_revision,
        "model_revision": request.model_revision,
    }


def _trainer_start(request: TrainingRequest) -> dict[str, Any]:
    return {
        "adapter_revision": request.adapter_revision,
        "model_revision": request.model_revision,
    }


def _validate_revision(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_fresh_output(output_dir: Path) -> None:
    managed = (
        output_dir / "training_generations.jsonl",
        output_dir / "training_metrics.jsonl",
        output_dir / "training_manifest.json",
        output_dir / "checkpoints",
    )
    if any(path.exists() for path in managed):
        raise FileExistsError("training output already exists; resume from a checkpoint")


def _restore_committed_rows(path: Path, completed_examples: int) -> None:
    rows = list(iter_jsonl_objects(path)) if path.exists() else []
    if len(rows) < completed_examples:
        raise ValueError(f"resume log has fewer committed rows than checkpoint: {path}")
    committed = rows[:completed_examples]
    if [row.get("sequence_index") for row in committed] != list(
        range(completed_examples)
    ):
        raise ValueError(f"resume log sequence does not match checkpoint: {path}")
    if len(rows) != completed_examples:
        _atomic_write_jsonl(path, committed)


def _load_optimizer_state(
    optimizer: torch.optim.Optimizer, path: Path
) -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state)


def _scalar(value: Tensor) -> float:
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise TypeError("OPD metrics must be scalar tensors")
    return float(value.detach().cpu().item())


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
