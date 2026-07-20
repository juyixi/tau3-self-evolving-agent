from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import train_opd_lora
from tau3_retail_evolver.slow_loop.audit import AuditError, AuditReport
from tau3_retail_evolver.slow_loop.trainer import TrainingRequest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def test_importing_training_cli_does_not_import_qwen_loader() -> None:
    module_names = (
        "scripts.train_opd_lora",
        "tau3_retail_evolver.models",
        "tau3_retail_evolver.models.lora",
        "tau3_retail_evolver.models.qwen35",
    )
    saved_modules = {
        name: sys.modules.pop(name)
        for name in module_names
        if name in sys.modules
    }
    try:
        importlib.import_module("scripts.train_opd_lora")

        assert "tau3_retail_evolver.models.qwen35" not in sys.modules
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _write_dataset_manifest(
    dataset_dir: Path,
    *,
    model_revision: str = "model-commit-a",
    adapter_revision: str = "adapter-a",
) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_build_id": "dataset-a",
                "dataset_schema_version": 1,
                "policy_lineage": {
                    "model_revision": model_revision,
                    "adapter_revision": adapter_revision,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _passing_audit() -> AuditReport:
    return AuditReport(
        dataset_build_id="dataset-a",
        passed=True,
        checked_artifacts=("datasets/sel.jsonl",),
        errors=(),
    )


def _argv(dataset_dir: Path, output_dir: Path, *extra: str) -> list[str]:
    return [
        "--config",
        str(DEFAULT_CONFIG),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--model-revision",
        "model-commit-a",
        "--adapter-revision",
        "adapter-a",
        *extra,
    ]


def test_cli_requires_dataset_output_and_both_revisions() -> None:
    with pytest.raises(SystemExit):
        train_opd_lora.parse_args([])


def test_cli_parses_config_repeated_overrides_resume_and_dry_run(tmp_path: Path) -> None:
    args = train_opd_lora.parse_args(
        _argv(
            tmp_path / "dataset",
            tmp_path / "output",
            "--set",
            "training.num_train_epochs=1",
            "--set",
            "training.per_device_batch_size=1",
            "--resume-from",
            str(tmp_path / "output" / "checkpoints" / "step-00000001"),
            "--dry-run",
        )
    )

    assert args.config == DEFAULT_CONFIG
    assert args.overrides == [
        "training.num_train_epochs=1",
        "training.per_device_batch_size=1",
    ]
    assert args.resume_from == tmp_path / "output" / "checkpoints" / "step-00000001"
    assert args.dry_run is True


def test_dry_run_audits_lineage_resolves_settings_and_prints_canonical_json_without_qwen_or_peft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    audit_calls: list[Path] = []
    monkeypatch.setattr(
        train_opd_lora,
        "audit_dataset",
        lambda path: audit_calls.append(path) or _passing_audit(),
    )
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "peft" or name.endswith(".models.qwen35"):
            raise AssertionError(f"dry-run imported training weights dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    returncode = train_opd_lora.main(
        _argv(
            dataset_dir,
            output_dir,
            "--set",
            "training.num_train_epochs=1",
            "--set",
            "training.per_device_batch_size=1",
            "--dry-run",
        )
    )

    raw_summary = capsys.readouterr().out
    summary = json.loads(raw_summary)
    assert returncode == 0
    assert audit_calls == [dataset_dir]
    assert summary == {
        "adapter_revision": "adapter-a",
        "dataset_build_id": "dataset-a",
        "dataset_dir": str(dataset_dir.resolve()),
        "dry_run": True,
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "model-commit-a",
        "output_dir": str(output_dir.resolve()),
        "resume_adapter_path": None,
        "resume_from": None,
        "rollout_config": {
            "max_episode_steps": 40,
            "temperature": 1.0,
            "top_p": 0.95,
        },
        "training_config": {
            "dtype": "bfloat16",
            "generation_max_new_tokens": 512,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "learning_rate": 1e-5,
            "loss_type": "forward_kl",
            "max_sequence_length": 8192,
            "num_train_epochs": 1,
            "per_device_batch_size": 1,
            "seed": 42,
            "target_modules": "all-linear",
        },
    }
    assert raw_summary == json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def test_preflight_requires_a_passing_stage5_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset_manifest(dataset_dir)
    monkeypatch.setattr(
        train_opd_lora,
        "audit_dataset",
        lambda path: AuditReport(
            dataset_build_id="dataset-a",
            passed=False,
            checked_artifacts=(),
            errors=(AuditError(code="hash_mismatch", message="bad hash"),),
        ),
    )

    with pytest.raises(ValueError, match="hash_mismatch"):
        train_opd_lora.main(_argv(dataset_dir, tmp_path / "output", "--dry-run"))


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ("lora.lora_r=16", "lora_r=32"),
        ("lora.lora_alpha=32", "lora_alpha=64"),
        ("lora.lora_dropout=0.1", "lora_dropout=0.05"),
        ("training.target_modules=q_proj", "target_modules='all-linear'"),
    ),
)
def test_dry_run_rejects_stage6_lora_setting_deviations_before_audit_or_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: str,
    match: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset_manifest(dataset_dir)
    audit_calls: list[Path] = []
    monkeypatch.setattr(
        train_opd_lora,
        "audit_dataset",
        lambda path: audit_calls.append(path) or _passing_audit(),
    )
    monkeypatch.setattr(
        train_opd_lora,
        "_load_training_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must not load")),
    )

    with pytest.raises(ValueError, match=match):
        train_opd_lora.main(
            _argv(
                dataset_dir,
                tmp_path / "output",
                "--set",
                override,
                "--dry-run",
            )
        )

    assert audit_calls == []


@pytest.mark.parametrize(
    ("model_revision", "adapter_revision", "match"),
    (
        ("other-model", "adapter-a", "model_revision"),
        ("model-commit-a", "other-adapter", "adapter_revision"),
    ),
)
def test_preflight_rejects_dataset_policy_lineage_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_revision: str,
    adapter_revision: str,
    match: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset_manifest(
        dataset_dir,
        model_revision=model_revision,
        adapter_revision=adapter_revision,
    )
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match=match):
        train_opd_lora.main(_argv(dataset_dir, tmp_path / "output", "--dry-run"))


def test_fresh_preflight_rejects_existing_training_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    output_dir.mkdir()
    (output_dir / "training_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(FileExistsError, match="resume"):
        train_opd_lora.main(_argv(dataset_dir, output_dir, "--dry-run"))


def test_fresh_preflight_rejects_an_existing_non_directory_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    output_dir.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(NotADirectoryError, match="not a directory"):
        train_opd_lora.main(_argv(dataset_dir, output_dir, "--dry-run"))


def _write_resume_checkpoint(
    output_dir: Path,
    *,
    adapter_path: str = "adapter/shared_policy",
    manifest_overrides: dict[str, Any] | None = None,
    missing_fields: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    checkpoint = output_dir / "checkpoints" / "step-00000001"
    adapter = checkpoint / "adapter" / "shared_policy"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "adapter_path": adapter_path,
        "completed_examples": 2,
        "dataset_build_id": "dataset-a",
        "optimizer_steps": 1,
        "rollout_config": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_episode_steps": 40,
        },
        "schedule_fingerprint": "a" * 64,
        "schedule_sha256": "a" * 64,
        "schema_version": 1,
        "source_lineage": {
            "model_revision": "model-commit-a",
            "adapter_revision": "adapter-a",
        },
        "total_examples": 4,
        "trainer_start": {
            "model_revision": "model-commit-a",
            "adapter_revision": "adapter-a",
        },
        "training_config": {
            "seed": 42,
            "dtype": "bfloat16",
            "target_modules": "all-linear",
            "max_sequence_length": 8192,
            "gradient_checkpointing": True,
            "learning_rate": 1e-5,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 3,
            "generation_max_new_tokens": 512,
            "loss_type": "forward_kl",
        },
    }
    manifest.update(manifest_overrides or {})
    for field in missing_fields:
        manifest.pop(field)
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer state")
    (checkpoint / "rng_state.pt").write_bytes(b"rng state")
    return checkpoint, adapter.resolve()


def test_resume_dry_run_reads_manifest_and_resolves_exact_adapter_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, adapter = _write_resume_checkpoint(output_dir)
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    train_opd_lora.main(
        _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["resume_from"] == str(checkpoint.resolve())
    assert summary["resume_adapter_path"] == str(adapter)


@pytest.mark.parametrize("adapter_path", ("../escape", "missing/shared_policy"))
def test_resume_preflight_rejects_escaping_or_missing_adapter_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_path: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, _ = _write_resume_checkpoint(output_dir, adapter_path=adapter_path)
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match="adapter_path"):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "dataset_build_id",
        "source_lineage",
        "trainer_start",
        "training_config",
        "rollout_config",
        "schedule_fingerprint",
        "schedule_sha256",
        "total_examples",
        "completed_examples",
        "optimizer_steps",
        "adapter_path",
    ),
)
def test_resume_preflight_rejects_a_checkpoint_missing_a_required_manifest_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, _ = _write_resume_checkpoint(output_dir, missing_fields=(field,))
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match=field.replace("_", "[ _]")):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"schema_version": True}, "schema_version"),
        ({"source_lineage": {}}, "source_lineage"),
        ({"schedule_fingerprint": " "}, "schedule_fingerprint"),
        ({"schedule_sha256": " "}, "schedule_sha256"),
        ({"schedule_sha256": "b" * 64}, "schedule"),
        ({"total_examples": 0}, "total_examples"),
        ({"completed_examples": True}, "completed_examples"),
        ({"completed_examples": 5}, "completed_examples"),
        ({"optimizer_steps": 0}, "optimizer_steps"),
        ({"optimizer_steps": 3}, "optimizer_steps"),
    ),
)
def test_resume_preflight_rejects_invalid_checkpoint_manifest_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, Any],
    match: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, _ = _write_resume_checkpoint(
        output_dir,
        manifest_overrides=overrides,
    )
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match=match):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


def test_resume_preflight_requires_optimizer_state_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, _ = _write_resume_checkpoint(output_dir)
    (checkpoint / "optimizer.pt").unlink()
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match="optimizer.pt"):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


def test_resume_preflight_requires_rng_state_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, _ = _write_resume_checkpoint(output_dir)
    (checkpoint / "rng_state.pt").unlink()
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match="rng_state.pt"):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


def test_resume_preflight_requires_adapter_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, adapter = _write_resume_checkpoint(output_dir)
    (adapter / "adapter_config.json").unlink()
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match="adapter_config.json"):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


@pytest.mark.parametrize("layout", ("missing", "invalid"))
def test_resume_preflight_requires_a_valid_stage6_adapter_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    layout: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, adapter = _write_resume_checkpoint(output_dir)
    contract = adapter / "stage6_adapter_contract.json"
    if layout == "missing":
        contract.unlink()
    else:
        contract.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "requested_target_modules": "q_proj",
                    "resolved_target_modules": ["q_proj"],
                    "options": {},
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match="adapter contract|stage6_adapter_contract"):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


@pytest.mark.parametrize("layout", ("missing", "ambiguous"))
def test_resume_preflight_requires_exactly_one_supported_adapter_weight_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    layout: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, adapter = _write_resume_checkpoint(output_dir)
    if layout == "missing":
        (adapter / "adapter_model.safetensors").unlink()
    else:
        (adapter / "adapter_model.bin").write_bytes(b"other adapter weights")
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())

    with pytest.raises(ValueError, match="exactly one supported PEFT weight"):
        train_opd_lora.main(
            _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint), "--dry-run")
        )


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        del seed


class _FakeModel:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    def to(self, device: str) -> "_FakeModel":
        self.events.append(("model.to", device))
        return self


def _runtime(events: list[Any]) -> Any:
    model = _FakeModel(events)

    def load_tokenizer(model_id: str, *, revision: str) -> object:
        events.append(("tokenizer", model_id, revision))
        return "tokenizer"

    def load_policy(*args: Any, revision: str, adapter_path: Path | None = None) -> Any:
        events.append(("policy", revision, adapter_path))
        return model

    class FakeTrainer:
        def __init__(self, loaded_model: Any, tokenizer: Any, training: Any, rollout: Any) -> None:
            events.append(("trainer", loaded_model, tokenizer))

        def train(self, request: TrainingRequest) -> Any:
            events.append(("train", request))
            return SimpleNamespace(
                output_dir=request.output_dir.resolve(),
                latest_checkpoint=request.output_dir.resolve() / "checkpoints" / "step-1",
                completed_examples=1,
                optimizer_steps=1,
            )

    return SimpleNamespace(
        torch=SimpleNamespace(
            cuda=_FakeCuda(),
            bfloat16="bfloat16",
            manual_seed=lambda seed: None,
        ),
        load_qwen35_tokenizer=load_tokenizer,
        load_shared_qwen35_policy=load_policy,
        OPDTrainer=FakeTrainer,
        TrainingRequest=TrainingRequest,
    )


def test_real_fresh_mode_pins_both_loaders_moves_policy_to_cuda_and_trains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    events: list[Any] = []
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())
    monkeypatch.setattr(train_opd_lora, "_load_training_runtime", lambda: _runtime(events))

    assert train_opd_lora.main(_argv(dataset_dir, output_dir)) == 0

    assert events[:3] == [
        ("tokenizer", "Qwen/Qwen3.5-9B", "model-commit-a"),
        ("policy", "model-commit-a", None),
        ("model.to", "cuda"),
    ]
    request = next(event[1] for event in events if event[0] == "train")
    assert request.loaded_adapter_path is None
    assert request.resume_from is None


def test_real_mode_seeds_python_torch_and_cuda_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    events: list[Any] = []
    runtime = _runtime(events)
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())
    monkeypatch.setattr(train_opd_lora, "_load_training_runtime", lambda: runtime)
    monkeypatch.setattr(
        "random.seed",
        lambda seed: events.append(("python.seed", seed)),
    )
    runtime.torch.manual_seed = lambda seed: events.append(("torch.manual_seed", seed))
    runtime.torch.cuda.manual_seed_all = lambda seed: events.append(
        ("cuda.manual_seed_all", seed)
    )

    train_opd_lora.main(_argv(dataset_dir, output_dir))

    assert events[:4] == [
        ("python.seed", 42),
        ("torch.manual_seed", 42),
        ("cuda.manual_seed_all", 42),
        ("tokenizer", "Qwen/Qwen3.5-9B", "model-commit-a"),
    ]


def test_real_resume_loads_manifest_adapter_before_qwen_and_passes_exact_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset_manifest(dataset_dir)
    checkpoint, adapter = _write_resume_checkpoint(output_dir)
    events: list[Any] = []
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())
    real_read = train_opd_lora._read_json_object

    def read_json(path: Path) -> dict[str, Any]:
        if path.name == "checkpoint_manifest.json":
            events.append(("checkpoint_manifest", path.resolve()))
        return real_read(path)

    monkeypatch.setattr(train_opd_lora, "_read_json_object", read_json)
    monkeypatch.setattr(
        train_opd_lora,
        "_load_training_runtime",
        lambda: events.append(("runtime",)) or _runtime(events),
    )

    train_opd_lora.main(
        _argv(dataset_dir, output_dir, "--resume-from", str(checkpoint))
    )

    assert events.index(("checkpoint_manifest", checkpoint / "checkpoint_manifest.json")) < events.index(
        ("runtime",)
    )
    assert ("policy", "model-commit-a", adapter) in events
    request = next(event[1] for event in events if event[0] == "train")
    assert request.resume_from == checkpoint.resolve()
    assert request.loaded_adapter_path == adapter


@pytest.mark.parametrize(
    ("available", "bf16", "match"),
    ((False, True, "CUDA"), (True, False, "BF16")),
)
def test_real_mode_fails_clearly_without_cuda_or_bf16(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    available: bool,
    bf16: bool,
    match: str,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset_manifest(dataset_dir)
    monkeypatch.setattr(train_opd_lora, "audit_dataset", lambda path: _passing_audit())
    runtime = _runtime([])
    runtime.torch.cuda = SimpleNamespace(
        is_available=lambda: available,
        is_bf16_supported=lambda: bf16,
    )
    monkeypatch.setattr(train_opd_lora, "_load_training_runtime", lambda: runtime)

    with pytest.raises(RuntimeError, match=match):
        train_opd_lora.main(_argv(dataset_dir, tmp_path / "output"))
