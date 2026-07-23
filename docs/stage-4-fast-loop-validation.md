# Stage 4 Fast Loop 与 Stage 5 数据验证

## 前置条件

- 使用项目支持的 Python 3.12 或 3.13 环境，并安装项目依赖。
- `configs/default.yaml` 指向完整的 Tau2 checkout。checkout 必须处于 `external/tau2-bench.commit` 固定的 commit，retail split 必须通过官方数量和 SHA-256 校验。
- Qwen 服务提供 `Qwen/Qwen3.5-9B`，并使用项目固定的 qwen3 reasoning parser、qwen3_coder tool parser 和生成参数。
- Stage 4 只采集 OPD on-policy 数据，不进行 self-distillation，也不训练 LoRA。LoRA 约束保持 PEFT `r=32`、`alpha=64`。

在当前 PowerShell 会话中设置端点和凭证：

```powershell
$env:QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
$env:QWEN_API_KEY = "EMPTY"
$env:OPENROUTER_API_KEY = Read-Host "OpenRouter API Key"
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API Key"
$env:QWEN_MODEL_REVISION = "Qwen/Qwen3.5-9B@<immutable-revision>"
```

`OPENROUTER_API_KEY` 供默认的 OpenRouter NL assertion evaluator 使用。默认用户模拟器 `deepseek/deepseek-v4-pro` 读取 `DEEPSEEK_API_KEY`。凭证不得写入 YAML、命令参数、manifest、事件或 summary。

## Fast Loop 运行

单任务 smoke：

```powershell
python -m scripts.run_fast_loop `
  --config configs/default.yaml `
  --split train `
  --task-id 0 `
  --run-id stage4-train-0 `
  --model-revision $env:QWEN_MODEL_REVISION `
  --completed-train-tasks-before 0
```

官方 train split 的前五个确定性任务是 `0`、`1`、`2`、`3`、`4`：

```powershell
python -m scripts.run_fast_loop `
  --config configs/default.yaml `
  --split train `
  --task-id 0 --task-id 1 --task-id 2 --task-id 3 --task-id 4 `
  --run-id stage4-train-0-4 `
  --model-revision $env:QWEN_MODEL_REVISION `
  --adapter-revision "adapter@immutable-revision" `
  --completed-train-tasks-before 0
```

每次运行必须使用新的 safe-slug `--run-id`。已存在的 run ID 和失败过的 run ID 都不能复用。`--model-revision` 与 `--adapter-revision` 必须描述实际服务中的不可变模型和 LoRA revision。

## Memory 开关与消融

`memory.enabled` 是 YAML 中的严格布尔总开关。默认 `true` 会执行 retrieve、select、write 和到期 maintenance。

无 Memory 消融使用单独的配置副本：

```yaml
memory:
  enabled: false
```

消融仍使用同一条 `python -m scripts.run_fast_loop` 命令、相同任务顺序、seed、Qwen 和用户模拟器。关闭后 manifest 与 summary 的 Memory snapshot 字段为 `null`，事件包含 `MemoryDisabled`，且不得创建或修改 `history/agents/<agent_id>/memory/`。Memory-disabled run 只能用于 baseline/ablation，Stage 5 会拒绝把它作为训练来源。

## 累计 Maintenance

`--completed-train-tasks-before` 必须显式给出同一 agent namespace 在本次运行前已经成功完成的 train task 总数。

- `0 + 5`：累计任务数到 5，不执行 maintenance。
- `25 + 5`：第五个任务后累计到 30，执行一次 maintenance round 1。

触发第 30 个任务的命令只需把五任务命令改为：

```powershell
  --run-id stage4-train-25-29 --completed-train-tasks-before 25
```

maintenance 只在 episode 完整成功后调度。episode 执行异常时保留已有事件，但不写 `fast_loop_summary.json`。

## Stage 4 产物

- `runs/<run_id>/manifest.json`：schema 2、模型和 adapter revision、输入 snapshot、Tau2 commit、split hash、任务和运行配置。
- `runs/<run_id>/rollouts/events.jsonl`：canonical rollout 与 maintenance 事件。
- `runs/<run_id>/fast_loop_summary.json`：reward、任务范围、输入和输出 snapshot、maintenance rounds。
- `history/agents/retail/memory/*_memory.json`：四层权威 Memory。
- `history/agents/retail/memory/snapshots/<snapshot_id>/`：不可变 Memory snapshot。

成功 episode 的事件顺序必须是：

1. `EpisodeStarted`
2. `MemoryCandidatesRetrieved`
3. `MemorySelected`
4. 一组或多组 `DecisionMade`、`EnvironmentStepped`
5. `EpisodeFinished`
6. `MemoryWriteProposed`
7. `MemoryWriteCommitted`

到达维护边界后，再出现 `MaintenanceStarted`、`MaintenanceProposed`、`MaintenanceCommitted`。

## Stage 5 构建

Stage 5 只接受相同 `iteration`、`model_revision` 和 `adapter_revision`，且任务范围与 Memory snapshot 连续的 schema-2 train runs。source run 必须显式列出，CLI 不会扫描整个 `runs/`。

```powershell
python -m scripts.build_opd_dataset `
  --config configs/default.yaml `
  --source-run runs/stage4-train-0-4 `
  --dataset-build-id opd-iter0-001 `
  --output-root runs `
  --project-root .
```

多个连续 run 重复传入 `--source-run`：

```powershell
python -m scripts.build_opd_dataset `
  --config configs/default.yaml `
  --source-run runs/stage4-train-0-14 `
  --source-run runs/stage4-train-15-29 `
  --dataset-build-id opd-iter0-030 `
  --output-root runs `
  --project-root .
```

正式产物位于 `runs/<dataset_build_id>/slow_loop/`：

- `evidence/episodes.jsonl`
- `attribution/memory_scores.jsonl`
- `datasets/sel.jsonl`
- `datasets/act.jsonl`
- `datasets/write.jsonl`
- `datasets/maint.jsonl`
- `dataset_manifest.json`
- `audit_report.json`

构建器先在临时兄弟目录中写入并审计，只有审计通过才发布正式目录。已有 `dataset_build_id` 不会被覆盖。

## 独立审计

```powershell
python -m scripts.audit_opd_dataset `
  --dataset-dir runs/opd-iter0-001/slow_loop `
  --project-root .
```

退出码 `0` 表示通过，退出码 `1` 表示失败。`--project-root` 可省略，此时 auditor 根据 manifest 中的 dataset 相对路径自动反推项目根目录。auditor 会从磁盘重新加载 source run、官方 Retail split 和冻结 Memory snapshot，重建 evidence、Eq.11 至 Eq.12 attribution 与四类样本，同时检查 canonical JSONL、SHA-256、public/privileged 边界和在线采样合同。

## 进入 Stage 6 的硬门槛

真实五任务 Memory-enabled schema-2 验证已于 2026-07-23 在 AutoDL 完成：

- 连续 source runs：`stage5-clean-iter0-task0-20260723a` 与 `stage5-clean-v2-iter0-task1-4-20260723a`。
- Memory snapshot chain：`ed80c101... -> 969e4cf5... -> 82032ba1...`。
- 数据集：`opd-iter0-5tasks-20260723g`。
- Build revision：`509715aa9375eb77a93c7f47438c38444a58599e`。
- 产出：5 条 episode evidence、11 条 Memory score、4 条 `sel`、1 条 `write`、0 条 `act`、0 条 `maint`。
- 独立审计：`passed=true`，检查 6 个 JSONL artifact，`errors=[]`。

本次同时修复了教师选择顺序与检索顺序不一致导致的 evidence 拒绝，以及 Tau2 evaluator/simulator 私有明细进入发布 evidence 的问题。原始 NL assertions 和模拟器明细仍保留在只读 source run，OPD evidence 只发布标量 outcome。

五任务 run 不保证四类数据都非空，因为 attribution 需要同 task group 的 selected/control 证据，maintenance 还要求累计到 `Q=30`。真实 evidence 构建与审计硬门槛已经通过，可以进入 Stage 6 最小训练验证；真实 30 任务四类样本覆盖仍作为扩大训练前的后续验证。四类 builder 非空与确定性重建已由离线两段连续 run 集成测试覆盖。
