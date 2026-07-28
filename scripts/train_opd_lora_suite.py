from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import uuid

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.pipeline.sampling import OPD_KINDS
from tau3_retail_evolver.slow_loop.audit import audit_dataset


_SCHEMA_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train four independent Qwen3.5 OPD LoRA capability adapters."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides=args.overrides)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_manifest = _read_json(dataset_dir / "dataset_manifest.json")
    audit = audit_dataset(dataset_dir)
    if not audit.passed:
        detail = ", ".join(f"{item.code}: {item.message}" for item in audit.errors)
        raise ValueError(f"Stage 5 dataset audit failed: {detail}")

    lineage = dataset_manifest.get("policy_lineage")
    expected_lineage = {
        "model_revision": args.model_revision,
        "adapter_revision": args.adapter_revision,
    }
    if not isinstance(lineage, Mapping) or any(
        lineage.get(field) != value for field, value in expected_lineage.items()
    ):
        raise ValueError("dataset policy lineage does not match suite revisions")

    counts = {
        kind: _kind_count(dataset_manifest, kind)
        for kind in OPD_KINDS
    }
    empty = [kind for kind, count in counts.items() if count < 1]
    if empty:
        raise ValueError(
            "four-LoRA training requires nonempty datasets: " + ", ".join(empty)
        )
    order = tuple(sorted(OPD_KINDS, key=lambda kind: (counts[kind], kind)))
    base_manifest = {
        "schema_version": _SCHEMA_VERSION,
        "status": "in_progress",
        "dataset_build_id": dataset_manifest.get("dataset_build_id"),
        "source_lineage": expected_lineage,
        "num_train_epochs": config.training.num_train_epochs,
        "training_order": list(order),
        "kinds": {
            kind: {
                "examples_per_epoch": counts[kind],
                "total_examples": counts[kind] * config.training.num_train_epochs,
                "output_dir": kind,
                "status": "pending",
            }
            for kind in OPD_KINDS
        },
        "total_examples": sum(counts.values()) * config.training.num_train_epochs,
    }
    if args.dry_run:
        _print_json({**base_manifest, "dry_run": True})
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = output_dir / "suite_manifest.json"
    suite = _load_or_initialize_suite(suite_path, base_manifest)
    for kind in order:
        kind_output = output_dir / kind
        completed = _completed_kind(kind_output, kind=kind)
        if completed is None:
            command = [
                sys.executable,
                "-m",
                "scripts.train_opd_lora",
                "--config",
                str(args.config),
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(kind_output),
                "--model-revision",
                args.model_revision,
                "--adapter-revision",
                args.adapter_revision,
                "--kind",
                kind,
            ]
            for override in args.overrides:
                command.extend(("--set", override))
            resume = _latest_checkpoint(kind_output)
            if resume is not None:
                command.extend(("--resume-from", str(resume)))
            suite["kinds"][kind]["status"] = "in_progress"
            _write_json_atomic(suite_path, suite)
            completed_process = subprocess.run(command, check=False)
            if completed_process.returncode != 0:
                suite["kinds"][kind]["status"] = "interrupted"
                _write_json_atomic(suite_path, suite)
                raise RuntimeError(
                    f"{kind} LoRA training exited with code "
                    f"{completed_process.returncode}"
                )
            completed = _completed_kind(kind_output, kind=kind)
            if completed is None:
                raise RuntimeError(f"{kind} LoRA training did not publish completion")
        _prune_completed_kind_checkpoints(kind_output)
        completed["retained_checkpoints"] = 1
        suite["kinds"][kind] = completed
        _write_json_atomic(suite_path, suite)

    suite["status"] = "complete"
    suite["adapter_bundle_revision"] = _bundle_revision(suite["kinds"])
    _write_json_atomic(suite_path, suite)
    _publish_bundle_manifest(output_dir, suite)
    _print_json(suite)
    return 0


def _load_or_initialize_suite(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        suite = json.loads(json.dumps(expected))
        _write_json_atomic(path, suite)
        return suite
    suite = _read_json(path)
    for field in (
        "schema_version",
        "dataset_build_id",
        "source_lineage",
        "num_train_epochs",
        "training_order",
        "total_examples",
    ):
        if suite.get(field) != expected.get(field):
            raise ValueError(f"existing suite manifest {field} mismatch")
    return suite


def _completed_kind(output_dir: Path, *, kind: str) -> dict[str, Any] | None:
    path = output_dir / "training_manifest.json"
    if not path.is_file():
        return None
    manifest = _read_json(path)
    if manifest.get("status") != "complete":
        return None
    if manifest.get("opd_kind") != kind:
        raise ValueError(f"{kind} training manifest OPD kind mismatch")
    relative = manifest.get("latest_checkpoint")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{kind} training manifest checkpoint is missing")
    checkpoint = (output_dir / relative).resolve()
    try:
        checkpoint.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError(f"{kind} checkpoint escapes its output directory") from error
    checkpoint_manifest = _read_json(checkpoint / "checkpoint_manifest.json")
    if checkpoint_manifest.get("opd_kind") != kind:
        raise ValueError(f"{kind} checkpoint OPD kind mismatch")
    return {
        "adapter_revision": checkpoint_manifest.get("adapter_revision"),
        "completed_examples": manifest.get("completed_examples"),
        "examples_per_epoch": (
            manifest.get("total_examples")
            // manifest["training_config"]["num_train_epochs"]
        ),
        "latest_checkpoint": checkpoint.relative_to(output_dir.parent).as_posix(),
        "output_dir": kind,
        "status": "complete",
        "total_examples": manifest.get("total_examples"),
    }


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.is_dir():
        return None
    published = [
        child.resolve()
        for child in checkpoints.iterdir()
        if child.is_dir() and (child / "checkpoint_manifest.json").is_file()
    ]
    if not published:
        return None
    return max(published, key=lambda path: int(path.name.removeprefix("step-")))


def _prune_completed_kind_checkpoints(output_dir: Path) -> None:
    manifest = _read_json(output_dir / "training_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("checkpoint pruning requires completed child training")
    relative = manifest.get("latest_checkpoint")
    if not isinstance(relative, str) or not relative:
        raise ValueError("completed child training has no latest checkpoint")
    latest = (output_dir / relative).resolve()
    checkpoints = output_dir / "checkpoints"
    for child in checkpoints.iterdir():
        if (
            child.resolve() != latest
            and child.is_dir()
            and re.fullmatch(r"step-\d{8}", child.name)
            and (child / "checkpoint_manifest.json").is_file()
        ):
            shutil.rmtree(child)


def _kind_count(manifest: Mapping[str, Any], kind: str) -> int:
    artifacts = manifest.get("artifacts")
    metadata = (
        artifacts.get(f"datasets/{kind}.jsonl")
        if isinstance(artifacts, Mapping)
        else None
    )
    count = metadata.get("line_count") if isinstance(metadata, Mapping) else None
    if type(count) is not int or count < 0:
        raise ValueError(f"dataset artifact line_count is invalid for {kind}")
    return count


def _bundle_revision(kinds: Mapping[str, Any]) -> str:
    payload = {
        kind: {
            "adapter_revision": kinds[kind]["adapter_revision"],
            "latest_checkpoint": kinds[kind]["latest_checkpoint"],
        }
        for kind in OPD_KINDS
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"opd-four-lora-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _publish_bundle_manifest(
    output_dir: Path,
    suite: Mapping[str, Any],
) -> None:
    kinds = suite.get("kinds")
    if not isinstance(kinds, Mapping) or set(kinds) != set(OPD_KINDS):
        raise ValueError("complete suite must contain exactly four OPD kinds")
    adapter_checkpoints: dict[str, str] = {}
    adapter_revisions: dict[str, str] = {}
    completed_examples = 0
    for kind in OPD_KINDS:
        record = kinds[kind]
        if not isinstance(record, Mapping) or record.get("status") != "complete":
            raise ValueError(f"{kind} LoRA is not complete")
        relative = record.get("latest_checkpoint")
        revision = record.get("adapter_revision")
        count = record.get("completed_examples")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{kind} LoRA checkpoint is missing")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"{kind} LoRA revision is missing")
        if type(count) is not int or count < 1:
            raise ValueError(f"{kind} completed example count is invalid")
        checkpoint = (output_dir / relative).resolve()
        try:
            checkpoint.relative_to(output_dir.resolve())
        except ValueError as error:
            raise ValueError(f"{kind} LoRA checkpoint escapes suite output") from error
        checkpoint_manifest = _read_json(checkpoint / "checkpoint_manifest.json")
        if checkpoint_manifest.get("status") != "checkpoint":
            raise ValueError(f"{kind} LoRA checkpoint is not published")
        if checkpoint_manifest.get("opd_kind") != kind:
            raise ValueError(f"{kind} LoRA checkpoint kind mismatch")
        if checkpoint_manifest.get("adapter_revision") != revision:
            raise ValueError(f"{kind} LoRA checkpoint revision mismatch")
        adapter_checkpoints[kind] = relative
        adapter_revisions[kind] = revision
        completed_examples += count

    bundle_revision = suite.get("adapter_bundle_revision")
    if not isinstance(bundle_revision, str) or not bundle_revision:
        raise ValueError("adapter bundle revision is missing")
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "dataset_build_id": suite.get("dataset_build_id"),
        "source_lineage": suite.get("source_lineage"),
        "adapter_revision": bundle_revision,
        "adapter_bundle_revision": bundle_revision,
        "adapter_checkpoints": adapter_checkpoints,
        "adapter_revisions": adapter_revisions,
        "completed_examples": completed_examples,
        "total_examples": suite.get("total_examples"),
        "num_train_epochs": suite.get("num_train_epochs"),
        "training_order": suite.get("training_order"),
        "kinds": kinds,
    }
    if manifest["completed_examples"] != manifest["total_examples"]:
        raise ValueError("four-LoRA completed example count does not match suite total")
    _write_json_atomic(output_dir / "training_manifest.json", manifest)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON value must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
