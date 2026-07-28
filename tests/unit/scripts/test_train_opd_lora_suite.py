from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from scripts import train_opd_lora_suite
from tau3_retail_evolver.slow_loop.audit import AuditReport
from tau3_retail_evolver.slow_loop.examples import (
    OPD_DATASET_SCHEMA_VERSION,
    OPD_SAMPLE_UNIT_CONTRACT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
COUNTS = {"sel": 198, "act": 96, "write": 50, "maint": 7}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_dataset(root: Path) -> None:
    _write_json(
        root / "dataset_manifest.json",
        {
            "dataset_build_id": "dataset-a",
            "dataset_schema_version": OPD_DATASET_SCHEMA_VERSION,
            "sample_unit_contract": OPD_SAMPLE_UNIT_CONTRACT,
            "policy_lineage": {
                "model_revision": "model-a",
                "adapter_revision": "zero-impact-init-v1",
            },
            "artifacts": {
                f"datasets/{kind}.jsonl": {
                    "line_count": count,
                    "sha256": kind * 16,
                }
                for kind, count in COUNTS.items()
            },
        },
    )


def _argv(dataset: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--config",
        str(DEFAULT_CONFIG),
        "--dataset-dir",
        str(dataset),
        "--output-dir",
        str(output),
        "--model-revision",
        "model-a",
        "--adapter-revision",
        "zero-impact-init-v1",
        *extra,
    ]


def _passing_audit() -> AuditReport:
    return AuditReport(
        dataset_build_id="dataset-a",
        passed=True,
        checked_artifacts=tuple(
            f"datasets/{kind}.jsonl" for kind in COUNTS
        ),
        errors=(),
    )


def _value(command: Sequence[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_dry_run_reports_natural_four_lora_counts(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    monkeypatch.setattr(
        train_opd_lora_suite,
        "audit_dataset",
        lambda path: _passing_audit(),
    )

    assert train_opd_lora_suite.main(
        _argv(dataset, tmp_path / "output", "--dry-run")
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["training_order"] == ["maint", "write", "act", "sel"]
    assert summary["total_examples"] == 1053
    assert {
        kind: summary["kinds"][kind]["total_examples"]
        for kind in COUNTS
    } == {
        "sel": 594,
        "act": 288,
        "write": 150,
        "maint": 21,
    }


def test_suite_trains_four_independent_outputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_dataset(dataset)
    monkeypatch.setattr(
        train_opd_lora_suite,
        "audit_dataset",
        lambda path: _passing_audit(),
    )
    calls: list[str] = []

    def fake_run(command: Sequence[str], *, check: bool) -> Any:
        assert check is False
        kind = _value(command, "--kind")
        calls.append(kind)
        kind_output = Path(_value(command, "--output-dir"))
        checkpoint = kind_output / "checkpoints" / "step-00000001"
        _write_json(
            kind_output
            / "checkpoints"
            / "step-00000000"
            / "checkpoint_manifest.json",
            {"status": "checkpoint"},
        )
        adapter = checkpoint / "adapter"
        _write_json(adapter / "adapter_config.json", {"r": 32})
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        _write_json(
            checkpoint / "checkpoint_manifest.json",
            {
                "adapter_path": "adapter",
                "adapter_revision": f"opd-{kind}-step-00000001",
                "dataset_build_id": "dataset-a",
                "opd_kind": kind,
                "source_lineage": {
                    "model_revision": "model-a",
                    "adapter_revision": "zero-impact-init-v1",
                },
                "status": "checkpoint",
            },
        )
        _write_json(
            kind_output / "training_manifest.json",
            {
                "adapter_revision": f"opd-{kind}-step-00000001",
                "completed_examples": COUNTS[kind] * 3,
                "latest_checkpoint": "checkpoints/step-00000001",
                "opd_kind": kind,
                "status": "complete",
                "total_examples": COUNTS[kind] * 3,
                "training_config": {"num_train_epochs": 3},
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_opd_lora_suite.subprocess, "run", fake_run)

    assert train_opd_lora_suite.main(_argv(dataset, output)) == 0

    assert calls == ["maint", "write", "act", "sel"]
    suite = json.loads((output / "suite_manifest.json").read_text(encoding="utf-8"))
    assert suite["status"] == "complete"
    assert suite["adapter_bundle_revision"].startswith("opd-four-lora-")
    assert all(suite["kinds"][kind]["status"] == "complete" for kind in COUNTS)
    bundle = json.loads(
        (output / "training_manifest.json").read_text(encoding="utf-8")
    )
    assert bundle["schema_version"] == 2
    assert bundle["status"] == "complete"
    assert bundle["completed_examples"] == 1053
    assert set(bundle["adapter_checkpoints"]) == set(COUNTS)
    assert all(
        [
            child.name
            for child in (output / kind / "checkpoints").iterdir()
            if child.is_dir()
        ]
        == ["step-00000001"]
        for kind in COUNTS
    )
