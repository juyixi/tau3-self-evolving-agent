# Tau3 Retail 留出集评测协议

## 1. 评测边界

Stage 8 只使用固定后的模型、LoRA checkpoint、Memory snapshot 和运行参数评测官方 Retail 留出集：

- 主评测 split 固定为 `test`，共 40 个官方任务。
- 不切分 dev，也不使用 test reward 调参、选择 checkpoint 或更新 Memory/模型训练状态。
- `base` 仅通过 `--official-base-reproduction` 运行，并在报告中单独标记。
- 评测入口不创建 optimizer，不执行 attribution，不构建 OPD dataset，也不保存 checkpoint。

2026-07-23 至 2026-07-24 已完成的真实五任务 Memory-enabled build/audit、Qwen3.5-9B BF16 单步更新和 Stage 7 五任务恢复验收属于训练侧前置 gate，不需要在 Stage 8 重复执行。

## 2. Memory 协议

### `no_memory`

不打开 Memory repository，不加载 Embedding 模型，也不执行 retrieve、select、write 或 maintenance。该协议同时用于基础 Qwen 与训练后 LoRA 的无 Memory 对照。

### `test_static`

只读加载一个 train 产生的冻结 snapshot：

```text
history/agents/retail/memory/snapshots/<memory_snapshot_id>/
```

每个 trial 都读取同一个 snapshot。Repository 拒绝所有写操作，评测过程也不会生成 write 或 maintenance 事件。

### `test_streaming`

每个 trial 都从空 Memory 开始，并只在该 trial 内执行 retrieve、select、write 和每 30 个已完成任务一次的 maintenance：

```text
history/evaluations/<run_id>/retail/quarantine/trial-000/
history/evaluations/<run_id>/retail/quarantine/trial-001/
...
```

不同 seed 的 trial 互不继承 test Memory。训练 source loader 统一拒绝任何位于 `history/evaluations/` 下的路径，因此 quarantine 不能回流 attribution、OPD dataset 或后续 iteration。

## 3. Trial、任务顺序与 Seed

不传 `--task-id` 时，CLI 按固定 Tau2 split 中的官方顺序运行全部 40 个 test task。报告默认运行四个 trial，使用 `training.seed` 起始的四个连续 seed；默认配置即 `42, 43, 44, 45`。

执行顺序为 trial-major：

```text
seed 42: task 1 ... task 40
seed 43: task 1 ... task 40
seed 44: task 1 ... task 40
seed 45: task 1 ... task 40
```

单任务或五任务 smoke 必须显式使用 `--num-trials 1`。若传入 `--seed`，其数量必须与 `--num-trials` 完全相等，且 seed 必须唯一。

## 4. 运行前提

控制机需要配置：

```powershell
$env:QWEN_BASE_URL = "http://<qwen-host>:8000/v1"
$env:QWEN_MODEL_REVISION = "<fixed-qwen-revision>"
$env:QWEN_API_KEY = "EMPTY"
$env:OPENROUTER_API_KEY = "<openrouter-key>"
$env:DEEPSEEK_API_KEY = "<deepseek-key>"
```

`evaluate_retail` 不会把 LoRA 热加载到 vLLM。执行训练后 LoRA 实验前，Qwen endpoint 必须已经实际加载相应 adapter；`--adapter-revision` 和 `--checkpoint` 只记录并约束实验 provenance，不能替代服务端加载。

## 5. Smoke 与完整评测

基础 Qwen 单任务 smoke：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol no_memory `
  --task-id 5 `
  --num-trials 1 `
  --run-id eval-base-smoke `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION
```

训练后 LoRA 无 Memory 完整评测：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol no_memory `
  --run-id eval-lora-no-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "<adapter-revision>" `
  --checkpoint "runs/iterations/<iteration>/checkpoints/<step>"
```

训练后 LoRA 加冻结 train Memory：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol test_static `
  --run-id eval-lora-static `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "<adapter-revision>" `
  --checkpoint "runs/iterations/<iteration>/checkpoints/<step>" `
  --memory-snapshot "history/agents/retail/memory/snapshots/<snapshot-id>"
```

训练后 LoRA 流式评测：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol test_streaming `
  --run-id eval-lora-streaming `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "<adapter-revision>" `
  --checkpoint "runs/iterations/<iteration>/checkpoints/<step>"
```

官方 `base=train+test` 复现只允许无 Memory：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol no_memory `
  --official-base-reproduction `
  --run-id eval-official-base `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION
```

每次运行必须使用新的 safe-slug `run_id`。失败或已存在的 run 不允许覆盖。

## 6. 评测产物

每个 run 位于 `runs/<run_id>/`：

- `manifest.json`：不可变运行参数、模型/adapter、Tau2/split revision、任务顺序、seed set、用户模拟器、NL evaluator、protocol 和副作用能力。
- `rollouts/events.jsonl`：逐 episode canonical 事件，评测事件带 `mode=evaluate`、`trial_index` 和 seed。
- `evaluation_report.json`：官方 reward/`reward_info`、逐任务与逐 episode 指标、失败分类及 Memory snapshot provenance。

汇总指标包括 task/trial/episode/completed count、mean reward、success rate、step count、environment/tool parse-error rate、模型 response parse-error rate、合并 parse-error rate 和 failure category。成功定义为官方 terminal reward 等于 `1.0`；reward 本身及 `reward_info` 始终来自 Tau2 evaluator，项目不重新实现官方评分。

## 7. 对照实验比较

至少生成以下四份报告：

1. 基础 Qwen，`no_memory`。
2. 训练后 LoRA，`no_memory`。
3. 训练后 LoRA，`test_static`。
4. 训练后 LoRA，`test_streaming`。

比较命令：

```powershell
python -m scripts.compare_evaluations `
  --report base_qwen=runs/eval-base/evaluation_report.json `
  --report trained_no_memory=runs/eval-lora-no-memory/evaluation_report.json `
  --report trained_static=runs/eval-lora-static/evaluation_report.json `
  --report trained_streaming=runs/eval-lora-streaming/evaluation_report.json `
  --baseline-label base_qwen `
  --output runs/evaluation-comparison.json
```

Memory protocol、adapter、checkpoint 和 Memory snapshot 是处理变量，可以不同。比较器要求以下控制变量完全一致：split/base 标记、基础模型及 revision、Tau2 commit、split hash、精确 task IDs/order、seed set、trial 数、用户模拟器、NL evaluator、最大步数和模型服务合同。

## 8. 验收

本地合成端到端测试：

```powershell
python -m pytest tests/integration/test_stage8_retail_evaluation.py -q
```

在具备真实服务与凭证的机器上启用单任务官方 test smoke：

```powershell
$env:RUN_RETAIL_EVALUATION_INTEGRATION = "1"
$env:RETAIL_EVALUATION_CONFIG = "configs/autodl-deepseek-eval.yaml"
python -m pytest tests/integration/test_stage8_retail_evaluation.py `
  -m tau2_integration -q
```

`RETAIL_EVALUATION_CONFIG` 未设置时默认使用 `configs/default.yaml`。测试会读取该配置中的 `evaluation.nl_assertions.api_key_env`，因此 OpenRouter 和 DeepSeek 评估器都可以使用各自的凭证，不会硬编码 provider。

完整 Stage 8 gate 只有在四份官方 40-task、4-trial 报告生成并通过受控比较后才完成。selection/writing/maintenance 单模块消融仅在主实验稳定后追加。
