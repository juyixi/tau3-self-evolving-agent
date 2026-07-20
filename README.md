# Tau3 Retail Evolver

This repository builds auditable retail learning datasets and trains a shared
Qwen3.5-9B LoRA policy with online policy distillation (OPD).

## AutoDL OPD training

Run these commands from the repository root on an AutoDL instance. Training
loads Qwen directly with Transformers; it never uses the vLLM OpenAI-compatible
API. Stop the vLLM server first so it releases GPU memory:

```bash
pkill -f 'vllm serve' || true
nvidia-smi
```

Install the project with the training extra. The lower bounds include released
Qwen3.5 support and are intentionally not represented by a generated lockfile:

```bash
python -m pip install -U pip
python -m pip install -e '.[training]'
```

Reuse the Hugging Face cache already populated by rollout or inference jobs.
`MODEL_REVISION` must be the same immutable Qwen commit recorded in the Stage 5
dataset lineage. After the model and that revision are cached, `HF_HUB_OFFLINE=1`
also makes accidental downloads fail immediately.

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export MODEL_REVISION='<qwen-hugging-face-commit>'
export ADAPTER_REVISION='<stage-5-source-adapter-revision>'
export DATASET_DIR='/root/autodl-tmp/tau3-retail-evolver/runs/opd-dataset-0001'
export OUTPUT_DIR='/root/autodl-tmp/tau3-retail-evolver/runs/opd-training-0001'
```

An existing local cache or copied Hugging Face snapshot may be placed under
`HF_HOME`; no vLLM server is needed. Keep the dataset's source-run and memory
snapshot artifacts at the project-relative paths recorded by its manifest so
the Stage 5 audit can reconstruct them.

Always run the preflight first. It loads config and overrides, performs the full
Stage 5 audit, verifies model and adapter lineage, validates fresh/resume output
state, and prints canonical JSON without importing PEFT or loading Qwen.

```bash
python -m scripts.train_opd_lora \
  --config configs/default.yaml \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-revision "$MODEL_REVISION" \
  --adapter-revision "$ADAPTER_REVISION" \
  --set training.per_device_batch_size=1 \
  --set training.gradient_accumulation_steps=8 \
  --dry-run
```

Run real training only after the dry-run succeeds. Real mode requires a
CUDA GPU with BF16 support, loads one shared zero-impact LoRA policy, and moves
it to CUDA before training.

```bash
python -m scripts.train_opd_lora \
  --config configs/default.yaml \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-revision "$MODEL_REVISION" \
  --adapter-revision "$ADAPTER_REVISION" \
  --set training.per_device_batch_size=1 \
  --set training.gradient_accumulation_steps=8
```

Resume from a completed optimizer-step checkpoint. The CLI reads
`checkpoint_manifest.json`, resolves its adapter path, reloads that exact
adapter, and then restores trainer and optimizer state.

```bash
export CHECKPOINT="$OUTPUT_DIR/checkpoints/step-00000001"

python -m scripts.train_opd_lora \
  --config configs/default.yaml \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-revision "$MODEL_REVISION" \
  --adapter-revision "$ADAPTER_REVISION" \
  --set training.per_device_batch_size=1 \
  --set training.gradient_accumulation_steps=8 \
  --resume-from "$CHECKPOINT" \
  --dry-run

python -m scripts.train_opd_lora \
  --config configs/default.yaml \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-revision "$MODEL_REVISION" \
  --adapter-revision "$ADAPTER_REVISION" \
  --set training.per_device_batch_size=1 \
  --set training.gradient_accumulation_steps=8 \
  --resume-from "$CHECKPOINT"
```

The opt-in smoke uses cached files only and performs no downloads:

```bash
export RUN_OPD_GPU_SMOKE=1
export OPD_GPU_SMOKE_MODEL_REVISION="$MODEL_REVISION"
python -m pytest tests/integration/test_opd_gpu_smoke.py -q
```
