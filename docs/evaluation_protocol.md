# Tau3 Retail Stage 8 实验与评测协议

## 1. 实验目标

Stage 8 使用一个严格的 2×2 因子实验，分别测量 Fast Loop Memory、OPD Slow Loop 权重更新以及二者协同带来的收益：

| Cell | 模型 | Memory | 说明 |
|---|---|---|---|
| A `base_no_memory` | 基础 Qwen3.5-9B | 关闭 | 裸模型 baseline |
| B `base_with_memory` | 基础 Qwen3.5-9B | 冻结 S1 | 纯 Fast Loop Memory 收益 |
| C `opd_with_memory` | OPD checkpoint C1 | 冻结 S1 | 完整系统 |
| D `opd_no_memory` | OPD checkpoint C1 | 关闭 | 权重内化收益 |

五个主要对比为：

- `B-A`：基础模型的 Memory 收益。
- `C-B`：已有 Memory 时的 OPD 收益。
- `D-A`：不带 Memory 时的 OPD 知识内化收益。
- `C-D`：训练后模型的 Memory 收益。
- `C-A`：完整系统收益。

协同效应定义为 `(C-D)-(B-A)`。

主评测只使用 40 个官方 `test` 任务，不切分 dev，也不使用 test 结果调参、选择 checkpoint、更新模型或修改训练 Memory。Tau2 `base=train+test` 复现仍可通过独立 flag 执行，但不进入上述主实验。

## 2. Fast Loop 三次 Train Pass

一个正式 Stage 8 iteration 默认在全部 74 个官方 `train` 任务上运行三次，共产生 222 个 episode：

```text
Pass 1: 74 tasks, seed 42, input S0 → output S0.1
Pass 2: 74 tasks, seed 43, input S0.1 → output S0.2
Pass 3: 74 tasks, seed 44, input S0.2 → output S1
```

三次 pass 必须满足：

- 使用同一个未更新的基础 Qwen/零影响 LoRA checkpoint。
- Memory 在 pass 之间持续积累。
- 每次使用不同固定 seed，并以该 seed 确定性打乱 74 个任务的顺序。
- 三次全部完成后才冻结最终 snapshot S1、构建 OPD 数据并运行 Slow Loop。
- Slow Loop 更新后不得再把新 rollout 混入这一 iteration 的旧数据集。

三次 pass 会重复相同的 74 个 task ID，这是有意的重复采样。每个 pass 内 task ID
必须唯一，跨 pass 的 episode 则以 `run_id:task_id` 作为唯一身份，因此 222 个
episode 会全部保留并进入归因与 OPD 数据构建。

`adapter_revision` 在 rollout 和训练入口中表示训练开始前的 source policy
revision。初始轮统一记录为 `zero-impact-init-v1`；它不是训练输出 C1 的
revision。零影响 LoRA 的初始增量为零，因此 rollout 服务端加载该 adapter
或直接使用裸基础权重在数值语义上等价。

PowerShell 示例：

```powershell
python -m scripts.run_fast_loop `
  --config configs/default.yaml `
  --split train `
  --all-train-tasks `
  --seed 42 `
  --run-id stage8-train-pass-1 `
  --completed-train-tasks-before 0 `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1

python -m scripts.run_fast_loop `
  --config configs/default.yaml `
  --split train `
  --all-train-tasks `
  --seed 43 `
  --run-id stage8-train-pass-2 `
  --completed-train-tasks-before 74 `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1

python -m scripts.run_fast_loop `
  --config configs/default.yaml `
  --split train `
  --all-train-tasks `
  --seed 44 `
  --run-id stage8-train-pass-3 `
  --completed-train-tasks-before 148 `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1
```

每次 run 的 `fast_loop_summary.json` 会记录 episode、Token、Memory 数量、复用、maintenance 和 snapshot lineage。第三次输出的 `output_memory_snapshot_id` 即 S1。

## 3. OPD 数据与 Slow Loop

三次 source run 按顺序构建同一个 OPD 数据集：

```powershell
python -m scripts.build_opd_dataset `
  --config configs/default.yaml `
  --source-run runs/stage8-train-pass-1 `
  --source-run runs/stage8-train-pass-2 `
  --source-run runs/stage8-train-pass-3 `
  --dataset-build-id stage8-opd-dataset
```

数据审计通过后训练一次 LoRA：

```powershell
python -m scripts.train_opd_lora `
  --config configs/default.yaml `
  --dataset-dir runs/opd_datasets/stage8-opd-dataset `
  --output-dir runs/opd_training/stage8-opd `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1
```

训练使用共享模型的 frozen teacher view 和 trainable student view，在在线采样 response token 上计算 `KL(teacher || student)`。训练完成后从 `runs/opd_training/stage8-opd/training_manifest.json` 读取：

- `latest_checkpoint`：C1 相对 training 目录的路径。
- `adapter_revision`：C1 的最终 adapter revision，例如 `opd-step-00000054`。

这两个最终值必须同时用于 Cell C 和 D，不能继续填写 source revision
`zero-impact-init-v1`。

## 4. 四组 Test 评测

每个 cell 默认运行 40 个 test task 和一个 trial，即每组 40 个 episode。四组必须共享完全相同的 task order、seed `42`、Tau2 commit、split hash、用户模拟器、NL evaluator、生成参数和最大步数。用户模拟器与 NL evaluator 均固定为 `deepseek/deepseek-v4-pro`。

Cell A：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol no_memory `
  --run-id stage8-a-base-no-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION
```

Cell B 使用冻结 S1，但仍由基础模型服务：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol test_static `
  --run-id stage8-b-base-with-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --memory-snapshot "history/agents/retail/memory/snapshots/<S1>"
```

加载 OPD adapter C1 后执行 Cell C：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol test_static `
  --run-id stage8-c-opd-with-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "<C1_ADAPTER_REVISION>" `
  --checkpoint "runs/opd_training/stage8-opd/checkpoints/<step>" `
  --memory-snapshot "history/agents/retail/memory/snapshots/<S1>"
```

Cell D 使用与 C 完全相同的 checkpoint：

```powershell
python -m scripts.evaluate_retail `
  --config configs/default.yaml `
  --protocol no_memory `
  --run-id stage8-d-opd-no-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "<C1_ADAPTER_REVISION>" `
  --checkpoint "runs/opd_training/stage8-opd/checkpoints/<step>"
```

`evaluate_retail` 不负责给 vLLM 热加载 adapter。运行 A/B 时服务端必须是裸基础模型，运行 C/D 时必须实际加载 C1。

## 5. Memory 与 Test 隔离

B 和 C 只能通过 `test_static` 读取同一个 S1：

```text
history/agents/retail/memory/snapshots/<S1>/
```

只读 repository 允许 retrieve/select，但拒绝 write、delete、merge 和 maintenance。A/D 完全不加载 Memory 或 Embedding。`test_streaming` 保留为补充实验，不进入四组主实验；其 quarantine 仍位于 `history/evaluations/`，训练 loader 明确拒绝该路径。

## 6. 核心指标

### Test 指标

- `pass@1`：成功 episode 数除以总 episode 数；四次 trial 不折叠成“至少成功一次”的 `pass@4`。
- Agent 平均 Token：`prompt_tokens + completion_tokens` 的 episode 均值。
- 成功 episode 平均 Token、输入/输出 Token 总量。
- Tau2 mean reward 与完整官方 `reward_info`。
- 平均 step、parse error、truncation 和 failure category。
- 基于 task-level paired bootstrap 的 pass@1 delta 与 95% 置信区间。

### Memory 指标

- S1 总量及 `trajectory/tip/skill/tool` 分层数量。
- 每次 train pass 后的 Memory 增长曲线。
- Memory selection 总次数和唯一复用数量。
- train reuse coverage：至少拥有一次后续使用机会、且后来被实际选入 prompt 的 Memory 比例。
- B/C 对同一 S1 的 test reuse coverage。

仅进入 retrieval top-k 不算复用，只有 `MemorySelected` 中实际进入 prompt 的 ID 才计数。

### OPD 指标

- evidence episode、maintenance 和 Memory score 数量。
- `sel/act/write/maint` 样本量与 skip reason。
- forward KL 总体均值、按 kind 均值和训练曲线。
- optimizer step、completed example 和生成 response token 数。
- dataset、snapshot、checkpoint 和 adapter lineage。

## 7. 实验报告与可视化

四个 `evaluation_report.json`、三个 train run、OPD dataset、training 目录和 S1 通过统一入口汇总：

```powershell
python -m scripts.build_stage8_report `
  --experiment-id stage8-main `
  --base-no-memory-report runs/stage8-a-base-no-memory/evaluation_report.json `
  --base-with-memory-report runs/stage8-b-base-with-memory/evaluation_report.json `
  --opd-with-memory-report runs/stage8-c-opd-with-memory/evaluation_report.json `
  --opd-no-memory-report runs/stage8-d-opd-no-memory/evaluation_report.json `
  --train-run runs/stage8-train-pass-1 `
  --train-run runs/stage8-train-pass-2 `
  --train-run runs/stage8-train-pass-3 `
  --dataset-dir runs/opd_datasets/stage8-opd-dataset `
  --training-dir runs/opd_training/stage8-opd `
  --memory-snapshot "history/agents/retail/memory/snapshots/<S1>" `
  --output-dir runs/stage8-main-report
```

入口先校验完整 2×2 合同、四组各 `40×1` episode 与完整 Token
telemetry、三次 74-task pass、snapshot chain、dataset audit/source policy、
training manifest 以及 C/D 的 checkpoint 和 adapter revision，再生成：

```text
runs/stage8-main-report/
├── stage8_experiment_report.json
└── stage8_dashboard.html
```

HTML 为无外部依赖的自包含报告，包含：

1. 四组 pass@1。
2. 四组平均 Agent Token。
3. B/C Memory 复用覆盖率。
4. 三次 train pass 的 Memory 增长。
5. 四类 OPD 样本量。
6. forward-KL 训练曲线。
7. 五个 paired contrast 及其 95% 置信区间。

## 8. 验收边界

2026-07-23 至 2026-07-24 已完成的真实五任务 Memory build/audit、Qwen3.5-9B BF16 单步更新和 Stage 7 恢复验收属于训练侧前置 gate，不需要重复执行。

Stage 8 最终 gate 只有在以下条件全部满足后才通过：

- 三个真实 74-task train pass 完成并产生 S1。
- OPD 数据审计和一次真实 Slow Loop 训练完成。
- A/B/C/D 四组各 40 task × 1 trial 完成。
- 统一报告通过 lineage/control 校验并生成 JSON 与 HTML 图表。
- 不存在任何 test artifact 返回训练流程的路径。
