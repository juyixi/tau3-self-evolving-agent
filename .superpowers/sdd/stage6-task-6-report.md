# Stage 6 Task 6 Report

## Status

Stage 6 Task 6 is complete for the local toy, CLI, dry-run, resume, and
adapter-only contracts. The opt-in real Qwen3.5-9B GPU smoke is implemented but
was not run because `RUN_OPD_GPU_SMOKE=1` was not set and no download was
performed. The full Stage 6 hardware gate therefore remains pending.

Feature commit: `6eb5904` (`feat: expose stage 6 opd training workflow`)

## Files

- `README.md`
- `docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md`
- `docs/superpowers/plans/2026-07-20-stage-6-shared-policy-opd.md`
- `pyproject.toml`
- `scripts/train_opd_lora.py`
- `src/tau3_retail_evolver/models/qwen35.py`
- `tests/integration/test_opd_gpu_smoke.py`
- `tests/unit/models/test_lora_config.py`
- `tests/unit/scripts/test_train_opd_lora.py`
- `tests/unit/slow_loop/test_loss.py`
- `tests/unit/slow_loop/test_opd_step.py`
- `tests/unit/test_config.py`

## Implementation

- Added the production `python -m scripts.train_opd_lora` entry point with all
  required arguments, repeated config overrides, fresh/resume modes, and a
  canonical JSON dry-run summary.
- Dry-run performs the full Stage 5 `audit_dataset` audit, requires a passing
  report, verifies policy lineage, and validates output/checkpoint/config
  contracts without importing Qwen or PEFT-facing runtime modules.
- Fresh real mode loads one zero-impact shared LoRA policy. Resume reads
  `checkpoint_manifest.json` before the training runtime, resolves and contains
  `adapter_path`, loads that adapter, and passes the exact resolved path as
  `TrainingRequest.loaded_adapter_path`.
- Processor and model loaders accept `revision` and pass it to
  `from_pretrained`; cache-only loading is available for the no-download smoke.
- Real mode requires BF16 CUDA and moves the loaded model to CUDA. It contains
  no vLLM or OpenAI-compatible API path.
- The GPU smoke trains one example and records finite KL, nonzero LoRA gradient
  norm, LoRA update evidence, zero base gradients, adapter-only artifacts, and
  adapter reload success.
- Strengthened the Task 3 full-vocabulary hand calculation and added
  teacher/loss exception mode-restoration coverage.
- Added complete AutoDL install, cache, preflight, real, resume, and opt-in smoke
  commands. The staged plan records the local pass and pending hardware gate.

## TDD Evidence

Observed red failures before production changes:

- CLI suite failed during collection because `scripts.train_opd_lora` did not
  exist.
- Loader tests failed with `unexpected keyword argument 'revision'`.
- Dependency test failed because the training extra still used
  `transformers>=4.44` and `peft>=0.12`.

Green verification:

```text
conda run -n tau3-bench python -m pytest tests/unit/scripts/test_train_opd_lora.py -q
14 passed in 1.96s

conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py tests/unit/slow_loop/test_loss.py tests/unit/slow_loop/test_opd_step.py tests/unit/slow_loop/test_trainer.py tests/unit/test_config.py tests/integration/test_opd_gpu_smoke.py -q
86 passed, 1 skipped in 3.43s

conda run -n tau3-bench python -m pytest -q
570 passed, 5 skipped in 18.87s

conda run -n tau3-bench python -m compileall -q scripts src tests
exit 0

git diff --cached --check
exit 0
```

The pre-change baseline was `553 passed, 4 skipped in 18.93s`. Ruff was not
installed in `tau3-bench`; it was not downloaded or installed for this task.

## Self-Review

- Confirmed dry-run returns before `_load_training_runtime` and has a guarded
  import test for Qwen/PEFT.
- Confirmed resume manifest and adapter resolution occur before Qwen loading,
  reject path escape/missing adapter, and validate dataset, trainer, training,
  and rollout lineage.
- Confirmed both processor and model receive the CLI model revision and the
  loaded model is moved to CUDA before trainer construction.
- Confirmed fresh mode passes no adapter path, preserving zero-impact LoRA
  initialization, while resume passes the same resolved path to loader and
  trainer request.
- Confirmed no base-model save path or vLLM API path was introduced.
- `git diff --check` reported only expected Windows LF-to-CRLF notices before
  staging and no staged whitespace errors.

## GPU Status And Concerns

- `tests/integration/test_opd_gpu_smoke.py` skipped by default as designed.
- The test uses `local_files_only=True`, so opting in cannot download weights.
- The current machine did not run the real Qwen3.5-9B BF16 step. Real model API,
  VRAM fit, finite KL, gradient/update, artifact, and reload assertions remain
  pending on an AutoDL GPU with the immutable revision already cached.
- PEFT and Accelerate are not installed in the local baseline environment, so
  optional dependency behavior was tested through lazy-import and fake-module
  contracts rather than by installing packages.
