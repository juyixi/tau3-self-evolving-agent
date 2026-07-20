# Stage 6 Task 4 Report

## Scope

Implemented the Qwen3.5 shared-policy loader and a zero-impact PEFT LoRA adapter lifecycle.
The implementation creates or reloads one adapter named `shared_policy`; it never creates a
teacher model copy.

## Files

- Created `src/tau3_retail_evolver/models/lora.py`
- Created `src/tau3_retail_evolver/models/qwen35.py`
- Updated `src/tau3_retail_evolver/models/__init__.py`
- Created `tests/unit/models/test_lora_config.py`
- Created `tests/integration/test_qwen35_loader.py`

## TDD Evidence

### Red

Command:

```powershell
conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py -q
```

Observed result before implementation: `9 failed in 1.63s`. Each failure correctly reported
that `tau3_retail_evolver.models.lora` or `tau3_retail_evolver.models.qwen35` did not exist.
No model was downloaded.

### Green

Unit lifecycle command after implementation:

```powershell
conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py -q
```

Observed result: `9 passed in 1.55s`.

Required focused command after adding opt-in integration coverage:

```powershell
conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py tests/integration/test_qwen35_loader.py -q
```

Observed result: `9 passed, 1 skipped in 1.57s`. The integration test is skipped unless
`RUN_QWEN35_INTEGRATION=1`, so it did not download or load a Qwen model here.

## Behavior Covered

- Builds PEFT `LoraConfig` from the project `LoraConfig` and `TrainingConfig` with
  `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `target_modules="all-linear"`,
  `task_type="CAUSAL_LM"`, and `init_lora_weights=True`.
- Rejects `False`, Gaussian, and PiSSA initializers because they do not guarantee zero impact.
- Lazily imports optional training dependencies; unit tests inject their narrow module boundary.
- Maps the configured dtype, enables gradient checkpointing, and disables `use_cache`.
- Creates or reloads exactly one trainable adapter, freezes every non-LoRA parameter, and
  validates that only LoRA parameters remain trainable.
- Saves PEFT adapter state only and rejects any state dict entry that is not a LoRA tensor.
- The opt-in real-Qwen test checks frozen base parameters, zero LoRA-B initialization, finite
  forward logits, and adapter reload.

## Self-Review

- `git diff --check` completed without whitespace errors.
- Optional imports remain inside runtime helpers, so ordinary policy imports do not require
  `peft`, `transformers`, or `torch`.
- The checkpoint helper intentionally treats every non-`lora_` state key as a base-model tensor;
  this is strict by design for the requested adapter-only artifact policy.

## Concerns

- The real Qwen test was not run because `RUN_QWEN35_INTEGRATION` was not set and the task
  explicitly prohibits downloading models during this run. It requires a locally available or
  explicitly permitted Qwen3.5 model plus suitable GPU memory when enabled.
- This task did not run the full repository suite; the requested focused unit and integration
  command is recorded above.

## Commit

Commit message: `feat: load qwen35 with zero impact lora`

## Fix: Review Findings

### Changes

- `save_adapter_checkpoint` now requests adapter state with
  `get_peft_model_state_dict(model, adapter_name="shared_policy")`.
- The checkpoint save explicitly selects `shared_policy`. For PEFT's non-default adapter
  layout, it now returns `destination/shared_policy`, verifies that this directory contains
  `adapter_config.json`, and the load path is passed through unchanged to
  `PeftModel.from_pretrained`.
- `build_lora_config` now rejects every Stage 6 configuration drift before constructing a PEFT
  config: `use_peft=True`, `r=32`, `alpha=64`, `dropout=0.05`,
  `target_modules="all-linear"`, and `init_lora_weights=True` are mandatory.

### Review-Fix TDD Evidence

Red command:

```powershell
conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py -q
```

Observed before the fix: `8 failed, 7 passed in 1.55s`.

- Five failures showed that non-mandatory Stage 6 LoRA values were accepted.
- Three failures showed that `get_peft_model_state_dict` was called without its required
  `adapter_name` keyword. The test double deliberately rejects that omission.

Green unit command:

```powershell
conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py -q
```

Observed after the fix: `15 passed in 1.42s`.

Focused verification command:

```powershell
conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py tests/integration/test_qwen35_loader.py -q
```

Observed result: `15 passed, 1 skipped in 1.45s`.

### Self-Review

- The fake PEFT boundary now requires `adapter_name` and emulates PEFT's non-default adapter
  subdirectory. The unit round trip proves that the exact returned directory is reloaded.
- Invalid Stage 6 values fail before `RecordingLoraConfig` is constructed.
- `git diff --check` reported no whitespace errors.

### Remaining Concern

The real PEFT/Qwen integration remains opt-in and skipped by default. It has not been run in
this environment because `RUN_QWEN35_INTEGRATION` is unset and no model download is allowed.
