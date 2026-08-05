# Stage 6 Shared-Policy OPD Slow Loop Implementation Plan

> **2026-07-28 修订：** 本计划中“一个 LoRA 混合四类样本”和“equal kind
> round-robin”已被真实实验否决。现行设计以
> [四能力 LoRA OPSD 训练设计](../../four-lora-opd-training.md) 为准：
> `sel/act/write/maint` 分别训练独立 LoRA，每类保持自然样本数。本文后续旧任务
> 记录仅保留为实现历史。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a zero-impact Qwen3.5-9B LoRA adapter with online student generation and full-vocabulary `KL(teacher || student)` on Stage 5 OPD examples.

**Architecture:** Stage 6 loads one frozen Qwen base and one current LoRA adapter. The student generates each completion from public input; the same model and adapter then run a stop-gradient privileged teacher forward and a gradient-bearing public student forward over the identical student token prefix. Training artifacts contain only adapter state, lineage manifests, and append-only JSONL generation metrics.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, Pydantic, pytest.

## Global Constraints

- Base model is exactly `Qwen/Qwen3.5-9B` and remains frozen.
- PEFT is mandatory with `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, and zero-impact initialization.
- Teacher and student use the same Python model object and current LoRA parameter storage.
- Teacher loads LoRA but runs in eval mode under `torch.no_grad()`; its logits are detached.
- Student samples on-policy from public input before either KL-scoring sequence is built.
- Loss is full-vocabulary token-level `KL(teacher || student)` on aligned student response positions only.
- Stage 5 source model and adapter revisions must match the trainer start revisions.
- Checkpoints save adapter files only, never Qwen base weights.
- Initial implementation targets BF16, gradient checkpointing, sequence length 8192, batch size 2, accumulation 4, learning rate `1e-5`, and three epochs; all are configurable.

---

### Task 1: Stage 6 Configuration and Training Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/tau3_evolver/config.py`
- Modify: `configs/default.yaml`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `TrainingConfig` fields consumed by the loader, generator, trainer, and CLI.

- [x] **Step 1: Write failing configuration tests**

```python
assert config.training.dtype == "bfloat16"
assert config.training.max_sequence_length == 8192
assert config.training.loss_type == "forward_kl"
assert config.training.target_modules == "all-linear"
```

- [x] **Step 2: Run the focused tests**

Run: `conda run -n tau3-bench python -m pytest tests/unit/test_config.py -q`

Expected: FAIL because the Stage 6 fields do not exist.

- [x] **Step 3: Add exact validated settings and optional dependencies**

```python
class TrainingConfig(_ConfigModel):
    seed: int = 42
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    target_modules: str | tuple[str, ...] = "all-linear"
    max_sequence_length: int = Field(default=8192, ge=128)
    gradient_checkpointing: StrictBool = True
    learning_rate: float = Field(default=1e-5, gt=0)
    per_device_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_train_epochs: int = Field(default=3, ge=1)
    generation_max_new_tokens: int = Field(default=512, ge=1)
    loss_type: Literal["forward_kl"] = "forward_kl"
```

Add a `training` optional-dependency group containing compatible `torch`, `transformers`, `peft`, `accelerate`, and `safetensors` requirements without forcing them into baseline-only installs.

- [x] **Step 4: Run focused tests and commit**

Run: `conda run -n tau3-bench python -m pytest tests/unit/test_config.py -q`

Expected: PASS.

Commit: `feat: configure stage 6 opd training`

---

### Task 2: Deterministic OPD Prompt Encoding and Prefix Alignment

**Files:**
- Create: `src/tau3_evolver/slow_loop/alignment.py`
- Create: `tests/unit/slow_loop/test_alignment.py`

**Interfaces:**
- Consumes: `OPDExample`, a Transformers-compatible tokenizer/processor, and generated response token IDs.
- Produces: `AlignedOPDBatch`, `render_public_prompt`, `render_teacher_prompt`, and `build_aligned_batch`.

- [x] **Step 1: Write toy-tokenizer tests**

```python
batch = build_aligned_batch(example, tokenizer, response_ids=(41, 42), max_length=32)
assert batch.student_response_positions.shape == batch.teacher_response_positions.shape
assert batch.student_input_ids[batch.student_response_positions].tolist() == [41, 42]
assert batch.teacher_input_ids[batch.teacher_response_positions].tolist() == [41, 42]
```

Cover all four `kind` values, deterministic JSON rendering, prompt-token masking, paired truncation, and rejection when the response alone exceeds `max_length`.

- [x] **Step 2: Verify failure**

Run: `conda run -n tau3-bench python -m pytest tests/unit/slow_loop/test_alignment.py -q`

Expected: FAIL because `alignment.py` is absent.

- [x] **Step 3: Implement explicit prompt and position mapping**

```python
@dataclass(frozen=True, slots=True)
class AlignedOPDBatch:
    student_input_ids: Tensor
    student_attention_mask: Tensor
    teacher_input_ids: Tensor
    teacher_attention_mask: Tensor
    student_response_positions: Tensor
    teacher_response_positions: Tensor
```

Serialize `public_input`, `privileged_hindsight`, and `response_schema` as canonical JSON instructions. Encode public and privileged prompts independently, append the same response IDs, truncate only from the left side of each prompt, and preserve every response token.

- [x] **Step 4: Run focused tests and commit**

Run: `conda run -n tau3-bench python -m pytest tests/unit/slow_loop/test_alignment.py -q`

Expected: PASS.

Commit: `feat: align public and privileged opd prefixes`

---

### Task 3: Forward KL and Shared-Model OPD Step

**Files:**
- Create: `src/tau3_evolver/slow_loop/loss.py`
- Create: `src/tau3_evolver/slow_loop/opd_step.py`
- Create: `tests/unit/slow_loop/test_loss.py`
- Create: `tests/unit/slow_loop/test_opd_step.py`

**Interfaces:**
- Produces: `token_forward_kl`, `OPDStepResult`, and `shared_policy_opd_step`.

- [x] **Step 1: Write hand-computed loss tests**

```python
loss = token_forward_kl(student_logits, teacher_logits, student_positions, teacher_positions)
expected = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1).mean()
torch.testing.assert_close(loss, expected)
```

Also assert identical logits give zero, prompt positions do not affect loss, full vocabulary contributes, and teacher logits have no gradient.

- [x] **Step 2: Write observable shared-model tests**

Use one toy causal module that records `self.training`, `torch.is_grad_enabled()`, input IDs, and parameter data pointers. Assert teacher runs first in eval/no-grad, student runs second with gradients, both calls use one object and storage, and the original mode is restored.

- [x] **Step 3: Verify failure**

Run: `conda run -n tau3-bench python -m pytest tests/unit/slow_loop/test_loss.py tests/unit/slow_loop/test_opd_step.py -q`

Expected: FAIL because the modules are absent.

- [x] **Step 4: Implement the paper-exact step**

```python
with torch.no_grad():
    model.eval()
    teacher_logits = model(...).logits.detach()
model.train(original_training)
student_logits = model(...).logits
loss = token_forward_kl(student_logits, teacher_logits, student_positions, teacher_positions)
```

Return detached scalar metrics but leave `loss` attached for caller-controlled accumulation and backward.

- [x] **Step 5: Run focused tests and commit**

Run: `conda run -n tau3-bench python -m pytest tests/unit/slow_loop/test_loss.py tests/unit/slow_loop/test_opd_step.py -q`

Expected: PASS.

Commit: `feat: add shared policy forward kl step`

---

### Task 4: Qwen3.5 and Zero-Impact PEFT Loader

**Files:**
- Create: `src/tau3_evolver/models/qwen35.py`
- Create: `src/tau3_evolver/models/lora.py`
- Modify: `src/tau3_evolver/models/__init__.py`
- Create: `tests/unit/models/test_lora_config.py`
- Create: `tests/integration/test_qwen35_loader.py`

**Interfaces:**
- Produces: `build_lora_config`, `load_qwen35_processor`, `load_shared_qwen35_policy`, `assert_only_lora_trainable`, and `save_adapter_checkpoint`.

- [x] **Step 1: Write mocked PEFT tests**

Assert `LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, init_lora_weights=True, target_modules="all-linear", task_type="CAUSAL_LM")`, reject non-zero-impact initializers, freeze every non-LoRA parameter, and reject saving any base-model tensor.

- [x] **Step 2: Verify failure**

Run: `conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py -q`

Expected: FAIL because the loader modules are absent.

- [x] **Step 3: Implement lazy optional imports and adapter lifecycle**

Load the processor and `AutoModelForCausalLM` from a local path or Hugging Face ID, map configured dtype to `torch.dtype`, enable gradient checkpointing, disable KV cache during training, create or load exactly one adapter, and validate trainable parameter names/counts.

- [x] **Step 4: Add opt-in GPU integration coverage**

Gate real model loading behind `RUN_QWEN35_INTEGRATION=1`; verify Qwen base parameters have no gradients, LoRA B matrices start at zero, one shared forward is finite, and adapter reload succeeds.

- [x] **Step 5: Run focused tests and commit**

Run: `conda run -n tau3-bench python -m pytest tests/unit/models/test_lora_config.py tests/integration/test_qwen35_loader.py -q`

Expected: unit tests PASS and GPU integration SKIP by default.

Commit: `feat: load qwen35 with zero impact lora`

---

### Task 5: Online Generator, Trainer, Adapter Checkpoint, and Resume

**Files:**
- Create: `src/tau3_evolver/slow_loop/trainer.py`
- Create: `tests/unit/slow_loop/test_trainer.py`

**Interfaces:**
- Consumes: Stage 5 `dataset_manifest.json` and `datasets/{sel,act,write,maint}.jsonl`.
- Produces: `OPDTrainer`, `TrainingRequest`, `TrainingResult`, adapter checkpoint directories, `training_generations.jsonl`, `training_metrics.jsonl`, and `training_manifest.json`.

- [x] **Step 1: Write CPU toy-model trainer tests**

Cover online generation, equal kind round-robin sampling, accumulation, final partial accumulation, source revision mismatch, adapter-only save, atomic manifest publication, and resume from the latest completed optimizer step.

- [x] **Step 2: Verify failure**

Run: `conda run -n tau3-bench python -m pytest tests/unit/slow_loop/test_trainer.py -q`

Expected: FAIL because the trainer is absent.

- [x] **Step 3: Implement the minimal trainer**

```python
for example in balanced_examples:
    response_ids = generate_student_response(model, tokenizer, example, generation_config)
    batch = build_aligned_batch(example, tokenizer, response_ids, max_length=config.max_sequence_length)
    result = shared_policy_opd_step(model, batch)
    (result.loss / accumulation_steps).backward()
```

Clip no gradients by default, step only LoRA optimizer parameters, append generation/metric JSONL rows after successful examples, and save adapter checkpoints with exact dataset/model/adapter lineage.

- [x] **Step 4: Run focused tests and commit**

Run: `conda run -n tau3-bench python -m pytest tests/unit/slow_loop/test_trainer.py -q`

Expected: PASS.

Commit: `feat: train and resume opd lora adapters`

---

### Task 6: Training CLI, Documentation, and GPU Smoke Contract

**Files:**
- Create: `scripts/train_opd_lora.py`
- Create: `tests/unit/scripts/test_train_opd_lora.py`
- Create: `tests/integration/test_opd_gpu_smoke.py`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md`

**Interfaces:**
- Produces: `python -m scripts.train_opd_lora` as the Stage 6 entry point.

- [x] **Step 1: Write CLI parsing tests**

Require `--dataset-dir`, `--output-dir`, `--model-revision`, and `--adapter-revision`; support `--config`, repeated `--set`, `--resume-from`, and `--dry-run`.

- [x] **Step 2: Verify failure**

Run: `conda run -n tau3-bench python -m pytest tests/unit/scripts/test_train_opd_lora.py -q`

Expected: FAIL because the script is absent.

- [x] **Step 3: Implement CLI and dry-run preflight**

Dry-run loads and audits the Stage 5 dataset, checks lineage and output paths, resolves training settings, and prints a JSON summary without importing PEFT or loading Qwen weights. Real mode loads the local Transformers model and calls `OPDTrainer.train()`.

- [x] **Step 4: Add and execute the opt-in GPU smoke contract**

Gate with `RUN_OPD_GPU_SMOKE=1`; train one example and assert finite KL, non-zero LoRA gradient norm, zero base gradient tensors, adapter-only artifacts, and exact tensor equality after reload. The gate passed on 2026-07-23 with Qwen3.5-9B/BF16 on an RTX 4090 48GB. The first execution exposed an empty adapter artifact caused by passing an already-filtered PEFT state dict back through `save_pretrained`; the save path and reload assertion were corrected before the final passing run.

- [x] **Step 5: Document AutoDL execution**

Document that the vLLM server must be stopped before training, the Hugging Face cache/model path may be reused, and the remote command must install the `training` extra.

- [x] **Step 6: Run the complete suite**

Run: `conda run -n tau3-bench python -m pytest -q`

Expected: all unit tests PASS; Tau2, Qwen3.5, and GPU integration tests SKIP unless explicitly enabled.

- [x] **Step 7: Commit**

Commit: `feat: expose stage 6 opd training workflow`

## Self-Review

- Spec coverage: loader, zero-impact LoRA, shared teacher/student storage, online response generation, same-prefix alignment, full-vocabulary forward KL, adapter-only artifacts, resume, and GPU smoke are assigned to explicit tasks.
- Placeholder scan: no TBD/TODO or deferred implementation placeholders are present.
- Type consistency: Stage 5 `OPDExample` flows through `build_aligned_batch` into `shared_policy_opd_step`; the trainer and CLI consume the same typed interfaces.
