from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any
import uuid

import torch
from torch import Tensor, nn

from tau3_evolver.config import RolloutConfig, TrainingConfig
from tau3_evolver.artifacts.jsonl import JsonlWriter, iter_jsonl_objects
from tau3_evolver.models.lora import save_adapter_checkpoint
from tau3_evolver.slow_loop.alignment import (
    AlignedOPDBatch,
    build_aligned_batch,
    render_public_prompt,
)
from tau3_evolver.slow_loop.examples import (
    OPD_DATASET_SCHEMA_VERSION,
    OPD_SAMPLE_UNIT_CONTRACT,
    OPDExample,
)
from tau3_evolver.slow_loop.opd_step import shared_policy_opd_step


OPD_KINDS = ("sel", "act", "write", "maint")
_KINDS = OPD_KINDS
_MANIFEST_SCHEMA_VERSION = 1
_MAX_PUBLISHED_CHECKPOINTS = 2
CheckpointSaver = Callable[[Any, Path], Path]
OptimizerFactory = Callable[..., torch.optim.Optimizer]


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    dataset_dir: Path
    output_dir: Path
    model_revision: str
    adapter_revision: str | None
    kind: str
    resume_from: Path | None = None
    loaded_adapter_path: Path | None = None
    allow_empty_debug: bool = False


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


@dataclass(frozen=True, slots=True)
class KindSample:
    epoch: int
    round_index: int
    kind: str
    index: int


def natural_kind_schedule(
    kind_count: int,
    *,
    kind: str,
    num_epochs: int,
    seed: int,
) -> tuple[KindSample, ...]:
    """Visit each example of one OPD capability exactly once per epoch."""
    if kind not in OPD_KINDS:
        raise ValueError(f"unknown OPD kind: {kind}")
    if type(kind_count) is not int or kind_count < 0:
        raise ValueError("kind count must be a non-negative integer")
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    schedule: list[KindSample] = []
    for epoch in range(num_epochs):
        indexes = list(range(kind_count))
        random.Random(f"{seed}:{epoch}:{kind}:opd-natural-v1").shuffle(indexes)
        schedule.extend(
            KindSample(
                epoch=epoch,
                round_index=round_index,
                kind=kind,
                index=index,
            )
            for round_index, index in enumerate(indexes)
        )
    return tuple(schedule)


class OPDTrainer:
    """Train one capability LoRA with a shared teacher/student policy."""

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
        _validate_kind(request.kind)

        dataset_dir = Path(request.dataset_dir).resolve()
        output_dir = Path(request.output_dir).resolve()
        dataset_manifest = _read_json_object(dataset_dir / "dataset_manifest.json")
        _validate_dataset_contract(dataset_manifest)
        source_lineage = _validate_source_lineage(dataset_manifest, request)
        if request.resume_from is None:
            _seed_training_rng(self.training_config.seed)
        schedule = _build_schedule(
            _load_examples(dataset_dir),
            self.training_config.num_train_epochs,
            kind=request.kind,
            seed=self.training_config.seed,
        )
        if not schedule and not request.allow_empty_debug:
            raise ValueError(f"Stage 5 dataset contains no {request.kind} OPD examples")
        schedule_fingerprint = _schedule_fingerprint(schedule, dataset_manifest)

        output_dir.mkdir(parents=True, exist_ok=True)
        generation_path = output_dir / "training_generations.jsonl"
        metric_path = output_dir / "training_metrics.jsonl"

        completed_examples = 0
        optimizer_steps = 0
        latest_checkpoint: Path | None = None
        resume_manifest: Mapping[str, Any] | None = None
        if request.resume_from is None:
            if request.loaded_adapter_path is not None:
                raise ValueError("loaded_adapter_path requires resume_from")
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
                checkpoint=latest_checkpoint,
                schedule_fingerprint=schedule_fingerprint,
            )
            committed_schedule = schedule[:completed_examples]
            _restore_committed_rows(generation_path, committed_schedule)
            _restore_committed_rows(metric_path, committed_schedule)

        trainable_parameters = _trainable_lora_parameters(self.model)
        optimizer = self.optimizer_factory(
            trainable_parameters, lr=self.training_config.learning_rate
        )
        optimizer.zero_grad(set_to_none=True)
        if latest_checkpoint is not None:
            _load_optimizer_state(optimizer, latest_checkpoint / "optimizer.pt")
            optimizer.zero_grad(set_to_none=True)
            _restore_rng_state(latest_checkpoint / "rng_state.pt")
            assert resume_manifest is not None
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

        if not schedule:
            latest_checkpoint, checkpoint_manifest = self._save_checkpoint(
                output_dir,
                optimizer,
                request=request,
                dataset_manifest=dataset_manifest,
                source_lineage=source_lineage,
                completed_examples=0,
                optimizer_steps=0,
                total_examples=0,
                schedule_fingerprint=schedule_fingerprint,
            )
            training_manifest = {
                **checkpoint_manifest,
                "debug_initialized_without_examples": True,
                "latest_checkpoint": latest_checkpoint.relative_to(output_dir).as_posix(),
                "status": "complete",
            }
            _atomic_write_json(output_dir / "training_manifest.json", training_manifest)
            return TrainingResult(
                output_dir=output_dir,
                latest_checkpoint=latest_checkpoint,
                completed_examples=0,
                optimizer_steps=0,
                manifest=training_manifest,
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
                schedule_fingerprint=schedule_fingerprint,
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
        checkpoint: Path,
        schedule_fingerprint: str,
    ) -> tuple[int, int]:
        _require_latest_published_checkpoint(checkpoint)
        if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise ValueError("resume checkpoint schema mismatch")
        if manifest.get("dataset_build_id") != dataset_manifest.get("dataset_build_id"):
            raise ValueError("resume dataset lineage mismatch")
        if manifest.get("source_lineage") != source_lineage:
            raise ValueError("resume source lineage mismatch")
        if manifest.get("opd_kind") != request.kind:
            raise ValueError("resume OPD kind mismatch")
        if manifest.get("trainer_start") != _trainer_start(request):
            raise ValueError("resume trainer lineage mismatch")
        if manifest.get("training_config") != self.training_config.model_dump(mode="json"):
            raise ValueError("resume training config mismatch")
        if manifest.get("rollout_config") != self.rollout_config.model_dump(mode="json"):
            raise ValueError("resume rollout config mismatch")
        if manifest.get("total_examples") != total_examples:
            raise ValueError("resume sampling schedule mismatch")
        if manifest.get("schedule_fingerprint") != schedule_fingerprint:
            raise ValueError("resume schedule fingerprint mismatch")
        if manifest.get("schedule_sha256") != schedule_fingerprint:
            raise ValueError("resume schedule sha256 mismatch")
        _validate_loaded_adapter_path(manifest, request, checkpoint)
        if not (checkpoint / "rng_state.pt").is_file():
            raise ValueError("resume checkpoint rng_state.pt is missing")
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
        schedule_fingerprint: str,
    ) -> tuple[Path, dict[str, Any]]:
        checkpoints = output_dir / "checkpoints"
        checkpoint = checkpoints / f"step-{optimizer_steps:08d}"
        if checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
        checkpoints.mkdir(parents=True, exist_ok=True)
        temporary = checkpoints / f".{checkpoint.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            adapter_path = Path(
                self.checkpoint_saver(self.model, temporary / "adapter")
            ).resolve()
            try:
                adapter_relative_path = adapter_path.relative_to(temporary)
            except ValueError as error:
                raise ValueError("checkpoint saver returned a path outside the checkpoint") from error
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            _save_rng_state(temporary / "rng_state.pt")
            manifest = {
                "adapter_path": adapter_relative_path.as_posix(),
                "adapter_revision": (
                    f"opd-{request.kind}-step-{optimizer_steps:08d}"
                ),
                "completed_examples": completed_examples,
                "dataset_build_id": dataset_manifest.get("dataset_build_id"),
                "opd_kind": request.kind,
                "optimizer_steps": optimizer_steps,
                "rollout_config": self.rollout_config.model_dump(mode="json"),
                "schedule_fingerprint": schedule_fingerprint,
                "schedule_sha256": schedule_fingerprint,
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "source_lineage": dict(source_lineage),
                "status": "checkpoint",
                "total_examples": total_examples,
                "trainer_start": _trainer_start(request),
                "training_config": self.training_config.model_dump(mode="json"),
            }
            _atomic_write_json(temporary / "checkpoint_manifest.json", manifest)
            _fsync_files(temporary)
            _fsync_directory(temporary)
            os.replace(temporary, checkpoint)
            _fsync_directory(checkpoints)
            _prune_published_checkpoints(
                checkpoints,
                keep=_MAX_PUBLISHED_CHECKPOINTS,
            )
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return checkpoint, manifest


def _prune_published_checkpoints(checkpoints: Path, *, keep: int) -> None:
    published = sorted(
        (
            child
            for child in checkpoints.iterdir()
            if child.is_dir()
            and not child.is_symlink()
            and re.fullmatch(r"step-(\d{8})", child.name)
            and (child / "checkpoint_manifest.json").is_file()
        ),
        key=lambda path: int(path.name.removeprefix("step-")),
        reverse=True,
    )
    for checkpoint in published[keep:]:
        shutil.rmtree(checkpoint)
    if len(published) > keep:
        _fsync_directory(checkpoints)


def _load_examples(dataset_dir: Path) -> dict[str, tuple[OPDExample, ...]]:
    examples: dict[str, tuple[OPDExample, ...]] = {}
    for kind in _KINDS:
        path = dataset_dir / "datasets" / f"{kind}.jsonl"
        rows = tuple(OPDExample.model_validate(row) for row in iter_jsonl_objects(path))
        if any(example.kind != kind for example in rows):
            raise ValueError(f"dataset kind mismatch in {path}")
        examples[kind] = rows
    return examples


def _validate_dataset_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("dataset_schema_version") != OPD_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"OPD dataset schema version must be {OPD_DATASET_SCHEMA_VERSION}"
        )
    if manifest.get("sample_unit_contract") != OPD_SAMPLE_UNIT_CONTRACT:
        raise ValueError(
            "OPD dataset must use task-level sel/act/write and "
            "maintenance-round maint samples"
        )


def _require_latest_published_checkpoint(checkpoint: Path) -> None:
    candidates = [
        child.resolve()
        for child in checkpoint.parent.iterdir()
        if child.is_dir()
        and re.fullmatch(r"step-(\d{8})", child.name)
        and (child / "checkpoint_manifest.json").is_file()
    ]
    if not candidates:
        raise ValueError("resume output has no published checkpoints")
    latest = max(candidates, key=lambda path: int(path.name.removeprefix("step-")))
    if checkpoint != latest:
        raise ValueError(f"resume checkpoint must be the latest published checkpoint: {latest}")


def _build_schedule(
    examples: Mapping[str, Sequence[OPDExample]],
    num_epochs: int,
    *,
    kind: str,
    seed: int,
) -> tuple[_ScheduledExample, ...]:
    _validate_kind(kind)
    selected = examples.get(kind, ())
    natural = natural_kind_schedule(
        len(selected),
        kind=kind,
        num_epochs=num_epochs,
        seed=seed,
    )
    return tuple(
        _ScheduledExample(
            sequence_index=sequence_index,
            epoch=sample.epoch,
            example=selected[sample.index],
        )
        for sequence_index, sample in enumerate(natural)
    )


def _schedule_fingerprint(
    schedule: Sequence[_ScheduledExample], dataset_manifest: Mapping[str, Any]
) -> str:
    artifacts = dataset_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("dataset manifest artifact hash metadata is missing")
    artifact_hashes: dict[str, dict[str, Any]] = {}
    for path in sorted(artifacts):
        metadata = artifacts[path]
        if not isinstance(path, str) or not isinstance(metadata, Mapping):
            raise ValueError("dataset manifest artifact hash metadata is invalid")
        sha256 = metadata.get("sha256")
        if not isinstance(sha256, str) or not sha256:
            raise ValueError("dataset manifest artifact hash metadata is invalid")
        artifact_hashes[path] = {
            "line_count": metadata.get("line_count"),
            "sha256": sha256,
        }
    payload = {
        "artifacts": artifact_hashes,
        "schedule": [_schedule_identity(item) for item in schedule],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schedule_identity(scheduled: _ScheduledExample) -> dict[str, Any]:
    return {
        "epoch": scheduled.epoch,
        "example_id": scheduled.example.example_id,
        "kind": scheduled.example.kind,
        "sequence_index": scheduled.sequence_index,
    }


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
    max_new_tokens = min(
        training_config.generation_max_new_tokens,
        training_config.max_sequence_length - 1,
    )
    prompt_budget = training_config.max_sequence_length - max_new_tokens
    input_ids = input_ids[:, -prompt_budget:]
    attention_mask = attention_mask[:, -prompt_budget:]
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
                max_new_tokens=max_new_tokens,
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
        "opd_kind": request.kind,
    }


def _validate_loaded_adapter_path(
    manifest: Mapping[str, Any], request: TrainingRequest, checkpoint: Path
) -> None:
    relative_path = manifest.get("adapter_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("resume checkpoint adapter_path is invalid")
    expected = (checkpoint / relative_path).resolve()
    try:
        expected.relative_to(checkpoint)
    except ValueError as error:
        raise ValueError("resume checkpoint adapter_path escapes the checkpoint") from error
    if request.loaded_adapter_path is None:
        raise ValueError("resume requires loaded_adapter_path")
    if Path(request.loaded_adapter_path).resolve() != expected:
        raise ValueError("loaded_adapter_path does not match resume checkpoint adapter_path")


def _validate_revision(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"OPD kind must be one of {', '.join(_KINDS)}")


def _require_fresh_output(output_dir: Path) -> None:
    managed = (
        output_dir / "training_generations.jsonl",
        output_dir / "training_metrics.jsonl",
        output_dir / "training_manifest.json",
    )
    if any(path.exists() for path in managed):
        raise FileExistsError("training output already exists; resume from a checkpoint")
    checkpoints = output_dir / "checkpoints"
    if checkpoints.exists() and any(
        not (child.name.startswith(".step-") and ".tmp-" in child.name)
        for child in checkpoints.iterdir()
    ):
        raise FileExistsError("training output already exists; resume from a checkpoint")


def _restore_committed_rows(
    path: Path, committed_schedule: Sequence[_ScheduledExample]
) -> None:
    committed = _read_committed_jsonl_prefix(path, len(committed_schedule))
    for row, scheduled in zip(committed, committed_schedule, strict=True):
        identity = _schedule_identity(scheduled)
        if any(row.get(name) != value for name, value in identity.items()):
            raise ValueError(f"resume log metadata does not match schedule: {path}")
    _atomic_write_jsonl(path, committed)


def _read_committed_jsonl_prefix(path: Path, count: int) -> list[dict[str, Any]]:
    try:
        source = path.open("rb")
    except OSError as error:
        raise ValueError(f"unable to read resume JSONL file: {path}") from error
    rows: list[dict[str, Any]] = []
    with source:
        for line_number in range(1, count + 1):
            raw_line = source.readline()
            if not raw_line or not raw_line.endswith(b"\n"):
                raise ValueError(
                    f"resume log has fewer complete committed rows than checkpoint: {path}"
                )
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid committed JSONL at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"committed JSONL row must be an object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _load_optimizer_state(
    optimizer: torch.optim.Optimizer, path: Path
) -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state)


def _seed_training_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_rng_state(path: Path) -> None:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_states"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def _restore_rng_state(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"resume checkpoint rng_state.pt is missing: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint rng_state.pt must contain a mapping")
    python_state = state.get("python_random_state")
    _validate_python_random_state(python_state)
    torch_state = state.get("torch_cpu_rng_state")
    if not isinstance(torch_state, Tensor):
        raise ValueError("resume checkpoint Torch CPU RNG state is invalid")
    cuda_states: Sequence[Tensor] | None = None
    if torch.cuda.is_available():
        raw_cuda_states = state.get("torch_cuda_rng_states")
        if not isinstance(raw_cuda_states, Sequence) or not all(
            isinstance(cuda_state, Tensor) for cuda_state in raw_cuda_states
        ):
            raise ValueError("resume checkpoint CUDA RNG states are invalid")
        cuda_states = raw_cuda_states
    try:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(list(cuda_states))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("resume checkpoint RNG state is invalid") from error


def _validate_python_random_state(state: Any) -> None:
    if (
        not isinstance(state, tuple)
        or len(state) != 3
        or type(state[0]) is not int
        or not isinstance(state[1], tuple)
        or not state[1]
        or any(type(value) is not int for value in state[1])
        or (
            state[2] is not None
            and type(state[2]) not in {int, float}
        )
    ):
        raise ValueError("resume checkpoint Python RNG state is invalid")


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


def _fsync_files(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        with path.open("r+b") as source:
            os.fsync(source.fileno())
