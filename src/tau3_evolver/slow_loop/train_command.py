from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
from typing import Any

from tau3_evolver.config import ProjectConfig, load_config
from tau3_evolver.models.lora import (
    validate_stage6_adapter_contract,
    validate_stage6_lora_settings,
)
from tau3_evolver.slow_loop.training import OPD_KINDS
from tau3_evolver.slow_loop.audit import AuditReport, audit_dataset
from tau3_evolver.slow_loop.examples import (
    OPD_DATASET_SCHEMA_VERSION,
    OPD_SAMPLE_UNIT_CONTRACT,
)


@dataclass(frozen=True, slots=True)
class _Preflight:
    config: ProjectConfig
    dataset_dir: Path
    output_dir: Path
    model_revision: str
    adapter_revision: str
    kind: str
    examples_per_epoch: int
    dataset_build_id: str
    resume_from: Path | None
    resume_adapter_path: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Qwen3.5 LoRA capability on an audited Stage 5 dataset."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--kind", choices=OPD_KINDS, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = _run_preflight(args)
    if args.dry_run:
        _print_json(_preflight_summary(preflight))
        return 0

    runtime = _load_training_runtime()
    _require_cuda_bf16(runtime.torch, preflight.config)
    _seed_training_rng(runtime.torch, preflight.config.training.seed)
    tokenizer = runtime.load_qwen35_tokenizer(
        preflight.config.model.base_model,
        revision=preflight.model_revision,
    )
    model = runtime.load_shared_qwen35_policy(
        preflight.config.model,
        preflight.config.lora,
        preflight.config.training,
        revision=preflight.model_revision,
        adapter_path=preflight.resume_adapter_path,
    )
    model = model.to("cuda")
    trainer = runtime.OPDTrainer(
        model,
        tokenizer,
        preflight.config.training,
        preflight.config.rollout,
    )
    result = trainer.train(
        runtime.TrainingRequest(
            dataset_dir=preflight.dataset_dir,
            output_dir=preflight.output_dir,
            model_revision=preflight.model_revision,
            adapter_revision=preflight.adapter_revision,
            kind=preflight.kind,
            resume_from=preflight.resume_from,
            loaded_adapter_path=preflight.resume_adapter_path,
        )
    )
    _print_json(
        {
            "completed_examples": result.completed_examples,
            "kind": preflight.kind,
            "latest_checkpoint": str(result.latest_checkpoint),
            "optimizer_steps": result.optimizer_steps,
            "output_dir": str(result.output_dir),
            "status": "complete",
        }
    )
    return 0


def _run_preflight(args: argparse.Namespace) -> _Preflight:
    model_revision = _nonblank(args.model_revision, "model_revision")
    adapter_revision = _nonblank(args.adapter_revision, "adapter_revision")
    config = load_config(args.config, overrides=args.overrides)
    validate_stage6_lora_settings(config.lora, config.training)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    report = audit_dataset(args.dataset_dir)
    _require_passing_audit(report)
    dataset_manifest = _read_json_object(dataset_dir / "dataset_manifest.json")
    if dataset_manifest.get("dataset_schema_version") != OPD_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"OPD dataset schema version must be {OPD_DATASET_SCHEMA_VERSION}"
        )
    if dataset_manifest.get("sample_unit_contract") != OPD_SAMPLE_UNIT_CONTRACT:
        raise ValueError("OPD dataset sample-unit contract is invalid")
    lineage = dataset_manifest.get("policy_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("dataset policy_lineage is missing")
    if lineage.get("model_revision") != model_revision:
        raise ValueError("dataset policy_lineage model_revision does not match --model-revision")
    if lineage.get("adapter_revision") != adapter_revision:
        raise ValueError(
            "dataset policy_lineage adapter_revision does not match --adapter-revision"
        )
    build_id = dataset_manifest.get("dataset_build_id")
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("dataset_build_id is missing")
    examples_per_epoch = _kind_example_count(dataset_manifest, args.kind)
    if examples_per_epoch < 1:
        raise ValueError(f"dataset contains no {args.kind} OPD examples")

    resume_from: Path | None = None
    resume_adapter_path: Path | None = None
    if args.resume_from is None:
        _require_fresh_output(output_dir)
    else:
        resume_from = Path(args.resume_from).resolve()
        resume_adapter_path = _resolve_resume_adapter(
            resume_from,
            output_dir=output_dir,
            config=config,
            dataset_build_id=build_id,
            model_revision=model_revision,
            adapter_revision=adapter_revision,
            kind=args.kind,
        )
    return _Preflight(
        config=config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model_revision=model_revision,
        adapter_revision=adapter_revision,
        kind=args.kind,
        examples_per_epoch=examples_per_epoch,
        dataset_build_id=build_id,
        resume_from=resume_from,
        resume_adapter_path=resume_adapter_path,
    )


def _resolve_resume_adapter(
    checkpoint: Path,
    *,
    output_dir: Path,
    config: ProjectConfig,
    dataset_build_id: str,
    model_revision: str,
    adapter_revision: str,
    kind: str,
) -> Path:
    if not checkpoint.is_dir():
        raise ValueError(f"resume checkpoint does not exist: {checkpoint}")
    if checkpoint.parent.parent != output_dir:
        raise ValueError("resume checkpoint must be under output_dir/checkpoints")
    _require_latest_published_checkpoint(checkpoint)
    manifest = _read_json_object(checkpoint / "checkpoint_manifest.json")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("resume checkpoint schema_version must be 1")
    if manifest.get("dataset_build_id") != dataset_build_id:
        raise ValueError("resume checkpoint dataset_build_id mismatch")
    expected_lineage = {
        "model_revision": model_revision,
        "adapter_revision": adapter_revision,
    }
    if manifest.get("source_lineage") != expected_lineage:
        raise ValueError("resume checkpoint source_lineage mismatch")
    if manifest.get("opd_kind") != kind:
        raise ValueError("resume checkpoint OPD kind mismatch")
    if manifest.get("trainer_start") != {
        **expected_lineage,
        "opd_kind": kind,
    }:
        raise ValueError("resume checkpoint trainer_start mismatch")
    if manifest.get("training_config") != config.training.model_dump(mode="json"):
        raise ValueError("resume checkpoint training_config mismatch")
    if manifest.get("rollout_config") != config.rollout.model_dump(mode="json"):
        raise ValueError("resume checkpoint rollout_config mismatch")
    schedule_fingerprint = manifest.get("schedule_fingerprint")
    if not isinstance(schedule_fingerprint, str) or not schedule_fingerprint.strip():
        raise ValueError("resume checkpoint schedule_fingerprint must not be empty")
    schedule_sha256 = manifest.get("schedule_sha256")
    if not isinstance(schedule_sha256, str) or not schedule_sha256.strip():
        raise ValueError("resume checkpoint schedule_sha256 must not be empty")
    if schedule_fingerprint != schedule_sha256:
        raise ValueError("resume checkpoint schedule fingerprints do not match")
    total_examples = _positive_manifest_int(manifest, "total_examples")
    completed_examples = _positive_manifest_int(manifest, "completed_examples")
    optimizer_steps = _positive_manifest_int(manifest, "optimizer_steps")
    if completed_examples > total_examples:
        raise ValueError("resume checkpoint completed_examples exceeds total_examples")
    if optimizer_steps > completed_examples:
        raise ValueError("resume checkpoint optimizer_steps exceeds completed_examples")

    raw_adapter_path = manifest.get("adapter_path")
    if not isinstance(raw_adapter_path, str) or not raw_adapter_path:
        raise ValueError("resume checkpoint adapter_path is invalid")
    adapter_path = (checkpoint / raw_adapter_path).resolve()
    try:
        adapter_path.relative_to(checkpoint)
    except ValueError as error:
        raise ValueError("resume checkpoint adapter_path escapes the checkpoint") from error
    if not adapter_path.is_dir():
        raise ValueError("resume checkpoint adapter_path does not exist")
    if not (adapter_path / "adapter_config.json").is_file():
        raise ValueError("resume checkpoint adapter_config.json is missing")
    validate_stage6_adapter_contract(adapter_path)
    adapter_weights = (
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
    )
    if sum(path.is_file() for path in adapter_weights) != 1:
        raise ValueError(
            "resume checkpoint adapter must contain exactly one supported PEFT weight file"
        )
    if not (checkpoint / "optimizer.pt").is_file():
        raise ValueError("resume checkpoint optimizer.pt is missing")
    if not (checkpoint / "rng_state.pt").is_file():
        raise ValueError("resume checkpoint rng_state.pt is missing")
    return adapter_path


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


def _positive_manifest_int(manifest: Mapping[str, Any], field: str) -> int:
    value = manifest.get(field)
    if type(value) is not int or value < 1:
        raise ValueError(f"resume checkpoint {field} must be a positive integer")
    return value


def _require_fresh_output(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"training output path is not a directory: {output_dir}")
    managed = (
        output_dir / "training_generations.jsonl",
        output_dir / "training_metrics.jsonl",
        output_dir / "training_manifest.json",
    )
    if any(path.exists() for path in managed):
        raise FileExistsError("training output already exists; use --resume-from")
    checkpoints = output_dir / "checkpoints"
    if checkpoints.exists() and any(
        not (child.name.startswith(".step-") and ".tmp-" in child.name)
        for child in checkpoints.iterdir()
    ):
        raise FileExistsError("training output already exists; use --resume-from")


def _require_passing_audit(report: AuditReport) -> None:
    if report.passed:
        return
    details = ", ".join(
        f"{error.code}: {error.message}" for error in report.errors
    ) or "unknown audit failure"
    raise ValueError(f"Stage 5 dataset audit failed: {details}")


def _require_cuda_bf16(torch_module: Any, config: ProjectConfig) -> None:
    if config.training.dtype != "bfloat16":
        raise RuntimeError("real OPD training requires training.dtype=bfloat16")
    if not torch_module.cuda.is_available():
        raise RuntimeError("real OPD training requires an available CUDA GPU")
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError("real OPD training requires CUDA BF16 support")


def _seed_training_rng(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _load_training_runtime() -> Any:
    import torch

    from tau3_evolver.models.qwen35 import (
        load_qwen35_tokenizer,
        load_shared_qwen35_policy,
    )
    from tau3_evolver.slow_loop.training import OPDTrainer, TrainingRequest

    return SimpleNamespace(
        torch=torch,
        load_qwen35_tokenizer=load_qwen35_tokenizer,
        load_shared_qwen35_policy=load_shared_qwen35_policy,
        OPDTrainer=OPDTrainer,
        TrainingRequest=TrainingRequest,
    )


def _preflight_summary(preflight: _Preflight) -> dict[str, Any]:
    return {
        "adapter_revision": preflight.adapter_revision,
        "dataset_build_id": preflight.dataset_build_id,
        "dataset_dir": str(preflight.dataset_dir),
        "dry_run": True,
        "examples_per_epoch": preflight.examples_per_epoch,
        "kind": preflight.kind,
        "model_id": preflight.config.model.base_model,
        "model_revision": preflight.model_revision,
        "output_dir": str(preflight.output_dir),
        "resume_adapter_path": (
            str(preflight.resume_adapter_path)
            if preflight.resume_adapter_path is not None
            else None
        ),
        "resume_from": (
            str(preflight.resume_from) if preflight.resume_from is not None else None
        ),
        "rollout_config": preflight.config.rollout.model_dump(mode="json"),
        "training_config": preflight.config.training.model_dump(mode="json"),
        "total_examples": (
            preflight.examples_per_epoch
            * preflight.config.training.num_train_epochs
        ),
    }


def _kind_example_count(manifest: Mapping[str, Any], kind: str) -> int:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("dataset artifact metadata is missing")
    metadata = artifacts.get(f"datasets/{kind}.jsonl")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"dataset artifact metadata is missing for {kind}")
    count = metadata.get("line_count")
    if type(count) is not int or count < 0:
        raise ValueError(f"dataset artifact line_count is invalid for {kind}")
    return count


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON value must be an object: {path}")
    return value


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _print_json(value: Any) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
