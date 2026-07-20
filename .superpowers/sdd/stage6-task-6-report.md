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

## Fix Pass

Fix implementation commit: `5e32866` (`fix: harden stage 6 training preflight`)

### Important Findings Resolved

- Exposed pure `validate_stage6_lora_settings(lora_config, training_config)`
  from `models/lora.py`. CLI preflight calls it immediately after
  `load_config`, and both LoRA construction and attachment reuse it. The module
  imports no Torch, PEFT, or Transformers runtime dependency.
- Dry-run now rejects deviations in LoRA rank, alpha, dropout, and target
  modules before dataset audit or runtime loader access.
- Resume preflight now requires a complete version-1 checkpoint manifest:
  dataset build identity, source lineage, trainer start lineage, full training
  and rollout configs, nonblank schedule SHA-256, valid and bounded example and
  optimizer counts, a contained existing adapter directory, and an
  `optimizer.pt` file.
- Checkpoint publication now includes `schedule_sha256` while retaining the
  existing schedule fingerprint field for compatibility. Direct trainer
  resume verifies the new field as well.
- Fresh mode now rejects an existing output path that is not a directory.
- The model package uses lazy Qwen loader exports, preserving the Torch-free
  CLI preflight import boundary. This was covered with a simple in-process test
  rather than subprocess machinery.

Files changed in the fix commit:

- `scripts/train_opd_lora.py`
- `src/tau3_retail_evolver/models/__init__.py`
- `src/tau3_retail_evolver/models/lora.py`
- `src/tau3_retail_evolver/slow_loop/trainer.py`
- `tests/unit/models/test_lora_config.py`
- `tests/unit/scripts/test_train_opd_lora.py`
- `tests/unit/slow_loop/test_trainer.py`

### Fix TDD Evidence

Tests were written first and observed failing for each reported behavior:

- Public-validator tests failed because
  `validate_stage6_lora_settings` did not exist.
- Dry-run deviation cases reached audit/runtime instead of failing during
  config preflight.
- Incomplete and invalid resume manifests were accepted, including missing
  schedule/count fields and absent `optimizer.pt`.
- An existing file was accepted as a fresh output path.
- The isolated import-boundary test showed eager Qwen module loading through
  `models/__init__.py`.
- The trainer manifest test failed with missing `schedule_sha256`.

Green verification after the fixes:

```text
conda run -n tau3-bench python -m pytest tests/unit/scripts/test_train_opd_lora.py tests/unit/models/test_lora_config.py tests/unit/slow_loop/test_trainer.py -q
86 passed in 4.20s

conda run -n tau3-bench python -m pytest -q
601 passed, 5 skipped in 20.49s

conda run -n tau3-bench python -m compileall -q scripts src tests
exit 0

git diff --cached --check
exit 0
```

### Fix Self-Review And Remaining Concerns

- Confirmed resume validation is Torch-free and does not import the trainer.
- Confirmed strict integer validation rejects booleans and enforces
  `optimizer_steps <= completed_examples <= total_examples`.
- Confirmed adapter paths remain contained by the checkpoint and both adapter
  and optimizer artifacts have the required filesystem types.
- Confirmed package-level Qwen loader exports still resolve after conversion
  to lazy imports.
- The opt-in GPU smoke remains skipped because `RUN_OPD_GPU_SMOKE=1` was not
  set. No model download was attempted; real Qwen BF16 hardware validation is
  still the outstanding Stage 6 gate.
- Ruff remains unavailable in the existing environment and was not installed.
- One redundant post-verification loader run encountered Windows `WinError 5`
  while pytest cleaned temporary directories. The isolated affected contract
  passed with cache handling disabled; the clean focused and full-suite runs
  above remain the authoritative results.

## Final Preflight Fix

Final implementation commit: `d826bce`
(`fix: validate resumable adapter artifacts`)

### Findings Resolved

- Resume preflight now requires nonblank `schedule_fingerprint` and
  `schedule_sha256` fields and rejects any value mismatch. This exactly matches
  the checkpoint writer and trainer resume contract.
- A resolved resume adapter directory must contain an
  `adapter_config.json` file and exactly one supported PEFT weight file:
  `adapter_model.safetensors` or `adapter_model.bin`.
- Valid resume fixtures now model the complete writer output. Rejection tests
  cover a missing adapter directory, missing config, missing weights, and the
  ambiguous two-weight-file case.
- All new checks remain in CLI preflight and do not import Torch, PEFT,
  Transformers, or the trainer.

### Final TDD Evidence

The updated tests were run before production changes and produced six expected
failures: missing/blank/mismatched schedule fingerprint validation, missing
adapter config, missing adapter weights, and ambiguous adapter weights.

```text
conda run -n tau3-bench python -m pytest tests/unit/scripts/test_train_opd_lora.py -q -p no:cacheprovider
6 failed, 40 passed in 2.38s

conda run -n tau3-bench python -m pytest tests/unit/scripts/test_train_opd_lora.py -q -p no:cacheprovider
46 passed in 2.30s

conda run -n tau3-bench python -m pytest tests/unit/scripts/test_train_opd_lora.py tests/unit/models/test_lora_config.py tests/unit/slow_loop/test_trainer.py -q -p no:cacheprovider
92 passed in 3.27s

conda run -n tau3-bench python -m pytest -q -p no:cacheprovider
607 passed, 5 skipped in 11.71s

conda run -n tau3-bench python -m compileall -q scripts src tests
exit 0

git diff --check
exit 0
```

### Final Concerns

- The five default skips still include the opt-in Qwen GPU smoke. No model
  download or real Qwen BF16 training was performed in this fix pass.
- Pytest cache handling remained disabled for verification to avoid the
  previously documented Windows cache-directory permission issue.

## Whole-Branch Fix

Whole-branch implementation commit: `d9d03ce`
(`fix: harden Stage 6 training contracts`)

### Review Requirements Resolved

- The LoRA builder now passes the complete explicit Stage 6 PEFT contract:
  `bias="none"`, `use_rslora=False`, `use_dora=False`,
  `modules_to_save=None`, and empty rank/alpha patterns in addition to the
  pinned rank, alpha, dropout, zero-impact initializer, causal-LM task type,
  and logical `all-linear` target.
- Adapter checkpoints now write `stage6_adapter_contract.json` beside the PEFT
  adapter config and weights. The dependency-free validator requires exact
  schema/options, logical `all-linear`, and a nonempty, unique, sorted list of
  resolved concrete targets.
- Adapter state is validated before save. Reload validates the JSON contract
  before invoking PEFT, then validates the loaded PEFT config and requires its
  resolved targets to match the contract. CLI resume preflight uses the same
  pure validator without importing Torch, PEFT, Transformers, or the trainer.
- The Qwen text path now uses `AutoTokenizer`; `load_qwen35_tokenizer` is
  exported through the model package and CLI, while `load_qwen35_processor`
  remains a compatibility alias returning the text tokenizer. No positional
  text is passed through an `AutoProcessor` contract.
- Forward KL promotes the complete selected student and detached teacher
  full-vocabulary logits to FP32 before log-softmax. The 65,536-vocabulary
  BF16 near-equality regression verifies finite FP32 loss and gradients against
  an independent FP32 reference.
- Fresh CLI execution seeds Python, Torch CPU, and Torch CUDA before tokenizer,
  model, and adapter construction. Fresh trainer execution seeds again before
  schedule sampling; resume restores checkpoint RNG state before stochastic
  generation.
- Checkpoints atomically include `rng_state.pt`; direct trainer resume and pure
  CLI artifact preflight both require it. The trusted project-owned RNG file is
  loaded with explicit `weights_only=False`, and Python state is structurally
  validated before `random.setstate`.
- The stochastic fresh-versus-resume regression constructs a newly loaded
  resumed model and verifies identical generated response IDs and exact final
  LoRA weights against uninterrupted training.
- Production gradient checkpointing now enables input gradients as required by
  frozen-base LoRA training. The opt-in GPU smoke uses
  `gradient_checkpointing=True` and the text tokenizer path.

Files changed in the whole-branch implementation commit:

- `scripts/train_opd_lora.py`
- `src/tau3_retail_evolver/models/__init__.py`
- `src/tau3_retail_evolver/models/lora.py`
- `src/tau3_retail_evolver/models/qwen35.py`
- `src/tau3_retail_evolver/slow_loop/loss.py`
- `src/tau3_retail_evolver/slow_loop/trainer.py`
- `tests/integration/test_opd_gpu_smoke.py`
- `tests/integration/test_qwen35_loader.py`
- `tests/unit/models/test_lora_config.py`
- `tests/unit/scripts/test_train_opd_lora.py`
- `tests/unit/slow_loop/test_loss.py`
- `tests/unit/slow_loop/test_trainer.py`

### Whole-Branch TDD Evidence

The first shell-level focused invocation used an unrelated Python 3.14 pytest
and failed collection because Torch was unavailable. The project plan identified
`tau3-bench` as the required environment; the authoritative unchanged focused
red run was then:

```text
conda run -n tau3-bench python -m pytest -q tests/unit/models/test_lora_config.py tests/integration/test_qwen35_loader.py tests/unit/scripts/test_train_opd_lora.py tests/unit/slow_loop/test_loss.py tests/unit/slow_loop/test_trainer.py tests/integration/test_opd_gpu_smoke.py --basetemp=.pytest-tmp/whole-fix-red
15 failed, 102 passed, 2 skipped in 7.08s
```

All 15 failures were the incomplete adapter contract: missing explicit PEFT
options, missing persisted/validated contract, incomplete reload checks, and
missing CLI contract preflight. Two additional focused regressions were added
for the explicit trusted RNG load and pre-`random.setstate` validation; both
were observed failing before production changes.

Final green verification:

```text
conda run -n tau3-bench python -m pytest -q tests/unit/models/test_lora_config.py tests/integration/test_qwen35_loader.py tests/unit/scripts/test_train_opd_lora.py tests/unit/slow_loop/test_loss.py tests/unit/slow_loop/test_trainer.py tests/integration/test_opd_gpu_smoke.py --basetemp=.pytest-tmp/whole-fix-focused
119 passed, 2 skipped in 4.59s

conda run -n tau3-bench python -m pytest -q --basetemp=.pytest-tmp/whole-fix-full
627 passed, 5 skipped in 14.63s

conda run -n tau3-bench python -m compileall -q scripts src tests
exit 0

git diff --check
exit 0 (only expected Windows LF-to-CRLF notices)

git diff --cached --check
exit 0
```

### Whole-Branch Review And Concerns

- A final production/test diff review found no unresolved correctness or
  scope issues. The implementation commit contains only the requested Stage 6
  production and regression-test changes.
- The five full-suite skips are expected opt-in integrations. The changed
  focused set skipped both GPU smoke cases; no model was downloaded.
- Real Qwen3.5-9B BF16 execution, VRAM fit, checkpointed backward, adapter
  reload, and exact PEFT-version behavior remain pending on the intended GPU
  host with the immutable model revision already cached.

## Final Review Fixes

The final whole-branch review identified two remaining correctness gaps. Both
were reproduced with failing tests before production changes:

- Selecting an older published checkpoint could truncate committed JSONL logs
  and then fail while trying to overwrite a newer checkpoint. CLI preflight
  and direct trainer resume now require the highest published `step-*`
  checkpoint before any log repair. The regression also verifies that rejected
  resume attempts leave both training logs byte-for-byte unchanged.
- The suffix-only adapter contract could not distinguish full `all-linear`
  coverage from layer-filtered coverage. Contract schema 2 now stores the full
  eligible linear-module instance list and PEFT's actual targeted-module
  instance list. Fresh attach, save, and reload require exact equality, while
  reload recomputes eligibility from the pinned base model. PEFT exclusion,
  layer filtering, token targeting, layer replication, LoRA bias, and direct
  parameter targeting must all retain their standard disabled values.

TDD evidence:

```text
stale checkpoint red: 2 failed
stale checkpoint green: 2 passed
adapter coverage red: 9 failed, 14 passed
adapter coverage green: 23 passed
focused final: 121 passed, 2 skipped
full final: 637 passed, 5 skipped
compileall: exit 0
```

The five full-suite skips still include the opt-in GPU paths. No model was
downloaded during this review pass; real Qwen3.5-9B BF16 execution remains the
only pending Stage 6 validation gate.
