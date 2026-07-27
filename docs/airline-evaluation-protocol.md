# Tau2 Airline：Memory × OPD 四组实验协议

## 接入约束

Airline 在原有 Retail 代码路径上作为第二个 Tau2 domain 接入，不复制一套
Fast/Slow Loop。三个运行入口——`run_baseline`、`run_fast_loop` 和
`evaluate_airline`——都调用官方 `tau2.run.run_domain`，并通过
`rollout.max_concurrency` 控制并发数。

固定 Tau2 revision 中 Airline 的官方划分为：

- `train`：30 个任务；
- `test`：20 个任务；
- `base`：50 个任务（仅用于显式的官方 base reproduction）。

Airline 不按意图、工具或动作再次划分 task group。全部 Airline episode 的
归因组统一为 `airline-v2`，维护决策使用 `airline-v2:maintenance`。OPD
数据构建和审计会拒绝混入 Retail group 或其他 Airline 子组。

先执行无网络预检：

```powershell
python -m tools.preflight.check_tau2_domain `
  --config configs/airline.yaml `
  --split train `
  --task-id 0
```

## S1：三次 Airline Train Pass

使用未经过本轮 OPD 更新的同一个基础模型，连续执行三次完整的 30-task
train pass。Memory 跨 pass 累积，第三次结束后冻结为 S1：

```powershell
python -m scripts.run_fast_loop `
  --config configs/airline.yaml `
  --split train --all-train-tasks --seed 42 `
  --run-id airline-train-pass-1 `
  --completed-train-tasks-before 0 `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1

python -m scripts.run_fast_loop `
  --config configs/airline.yaml `
  --split train --all-train-tasks --seed 43 `
  --run-id airline-train-pass-2 `
  --completed-train-tasks-before 30 `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1

python -m scripts.run_fast_loop `
  --config configs/airline.yaml `
  --split train --all-train-tasks --seed 44 `
  --run-id airline-train-pass-3 `
  --completed-train-tasks-before 60 `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1
```

每个 pass 内部由 `run_domain` 并发执行。并发任务共享批次开始时的只读 Memory
视图；完成后按官方任务顺序串行提交 Memory 写入。到 30-task 维护边界时才暂停
批处理执行维护，因此不会发生并发写入竞争。

Airline 同步采用最新 Fast Loop Memory 约束：每个 episode 最多新增 2 条 Tip，
Skill、Tool 和 Trajectory 各最多 1 条；maintenance 按 tier 保存轮转游标，
优先检查高相似度候选。当活跃 Tip 超过 240 条时，本轮必须对展示的 Tip 给出
显式 review，并执行 merge 或 retire。

## OPD 数据与 C1

三个 source run 共同构建一次 OPD 数据集：

```powershell
python -m scripts.build_opd_dataset `
  --config configs/airline.yaml `
  --source-run runs/airline-train-pass-1 `
  --source-run runs/airline-train-pass-2 `
  --source-run runs/airline-train-pass-3 `
  --dataset-build-id airline-opd-dataset

python -m scripts.audit_opd_dataset `
  --dataset-dir runs/airline-opd-dataset/slow_loop `
  --project-root .

python -m scripts.train_opd_lora `
  --config configs/airline.yaml `
  --dataset-dir runs/airline-opd-dataset/slow_loop `
  --output-dir runs/opd_training/airline-opd `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision zero-impact-init-v1
```

训练产出的最终 checkpoint 和 adapter revision 记为 C1。Cell C 与 D 必须使用
完全相同的 C1；Cell B 与 C 必须读取完全相同、只读的 S1。

## 四组 Test 实验

四个 cell 均按同一官方顺序执行全部 20 个 `test` 任务，使用相同 seed、Tau2
revision、用户模拟器、NL evaluator、生成参数和最大步数：

| Cell | 模型 | Memory | 主要解释 |
|---|---|---|---|
| A `base_no_memory` | Base | 关闭 | 裸基础模型 |
| B `base_with_memory` | Base | 冻结 S1 | Memory 效果，`B-A` |
| C `opd_with_memory` | C1 | 冻结 S1 | 完整系统 |
| D `opd_no_memory` | C1 | 关闭 | OPD 内化效果，`D-A` |

```powershell
# A
python -m scripts.evaluate_airline `
  --config configs/airline.yaml --protocol no_memory `
  --run-id airline-a-base-no-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION

# B
python -m scripts.evaluate_airline `
  --config configs/airline.yaml --protocol test_static `
  --run-id airline-b-base-with-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --memory-snapshot "history/agents/airline/memory/snapshots/<S1>"

# C（服务端需实际加载 C1）
python -m scripts.evaluate_airline `
  --config configs/airline.yaml --protocol test_static `
  --run-id airline-c-opd-with-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "<C1_ADAPTER_REVISION>" `
  --checkpoint "runs/opd_training/airline-opd/checkpoints/<step>" `
  --memory-snapshot "history/agents/airline/memory/snapshots/<S1>"

# D（与 C 使用同一 C1）
python -m scripts.evaluate_airline `
  --config configs/airline.yaml --protocol no_memory `
  --run-id airline-d-opd-no-memory `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "<C1_ADAPTER_REVISION>" `
  --checkpoint "runs/opd_training/airline-opd/checkpoints/<step>"
```

统一报告入口会从四份评测报告的 provenance 自动识别 Airline，并校验三次
30-task train pass、四组各 20-task test、S1、C1 和控制变量：

```powershell
python -m scripts.build_stage8_report `
  --experiment-id airline-memory-opd `
  --base-no-memory-report runs/airline-a-base-no-memory/evaluation_report.json `
  --base-with-memory-report runs/airline-b-base-with-memory/evaluation_report.json `
  --opd-with-memory-report runs/airline-c-opd-with-memory/evaluation_report.json `
  --opd-no-memory-report runs/airline-d-opd-no-memory/evaluation_report.json `
  --train-run runs/airline-train-pass-1 `
  --train-run runs/airline-train-pass-2 `
  --train-run runs/airline-train-pass-3 `
  --dataset-dir runs/airline-opd-dataset/slow_loop `
  --training-dir runs/opd_training/airline-opd `
  --memory-snapshot "history/agents/airline/memory/snapshots/<S1>" `
  --output-dir runs/airline-memory-opd-report
```

重点查看 `B-A`（Memory 增益）、`D-A`（OPD 内化增益）、`C-B`（已有 Memory
时的 OPD 增益）、`C-D`（OPD 后的 Memory 增益）以及交互项
`(C-D)-(B-A)`。`test_streaming` 仍只作为补充协议，不进入这四组主实验。
