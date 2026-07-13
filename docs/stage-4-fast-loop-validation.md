# Stage 4 Fast Loop 验证指南

## 前置条件

- 使用项目支持的 Python 3.12 或 3.13 环境，并安装项目依赖。
- `configs/default.yaml` 指向完整的 tau2 checkout；checkout 必须处于 `external/tau2-bench.commit` 固定的 commit，且 retail split 必须通过官方数量和 SHA-256 校验。
- Qwen 服务必须提供 `Qwen/Qwen3.5-9B`，使用 qwen3 reasoning parser、qwen3_coder tool parser、thinking 和项目固定的生成参数。
- Stage 4 只做 OPD on-policy 数据采集，不做 self-distillation，也不训练 LoRA。LoRA 约束仍是 PEFT `r=32`、`alpha=64`。

在当前 PowerShell 会话中设置凭证和端点：

```powershell
$env:QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
$env:QWEN_API_KEY = "EMPTY" # 可省略；省略时 CLI 精确使用 EMPTY
$env:OPENROUTER_API_KEY = Read-Host "OpenRouter API Key"
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API Key"
$env:QWEN_MODEL_REVISION = "Qwen/Qwen3.5-9B@<immutable-revision>"
```

`OPENROUTER_API_KEY` 仅供 OpenRouter NL assertions/evaluator 使用。默认 `tau2.user_llm=deepseek/deepseek-v4-pro` simulator 使用 `DEEPSEEK_API_KEY`；如果改用其他 simulator provider，必须另行配置该 provider 的凭证。凭证不得写入 YAML、命令参数、manifest、事件或 summary。

## 运行命令

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

pinned 官方 train split 的前五个确定性任务是 `0`、`1`、`2`、`3`、`4`：

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

每次运行必须使用新的 safe-slug `--run-id`。已有 `runs/<run_id>/` 或 manifest 的 run ID 会被拒绝，失败的 run ID 也不能复用。

`--model-revision` 是实际服务模型的不可变 provenance；运行时模型名称仍固定为 `Qwen/Qwen3.5-9B`。`--adapter-revision` 是可选的当前服务 LoRA/adapter provenance，不会在 Stage 4 触发训练或切换 endpoint。

## Memory 开关与消融运行

`memory.enabled` 是 YAML 配置级总开关，必须是严格 boolean；默认 `true` 保持完整的 Stage 4 Memory 生命周期。无 Memory 消融使用单独、受版本控制的配置副本，例如：

```yaml
memory:
  enabled: false
```

关闭开关**仍使用同一条** `python -m scripts.run_fast_loop` 命令和相同 CLI 参数、Qwen、任务顺序、seed 与用户模拟器；只将 `--config` 指向上述配置副本。不要增加 `run_no_memory` 或 CLI override，以免消融经过不同的 orchestration 路径。`--completed-train-tasks-before` 仍为必填参数，用于保持任务计数可比；关闭 Memory 时它不会触发 maintenance。

示例（将配置副本路径和 run ID 换成实际值）：

```powershell
python -m scripts.run_fast_loop `
  --config configs/memory-disabled.yaml `
  --split train `
  --task-id 0 `
  --run-id stage4-memory-disabled-0 `
  --model-revision $env:QWEN_MODEL_REVISION `
  --completed-train-tasks-before 0
```

启用 `memory.enabled: true` 时，使用本指南前述命令与既有检查：manifest/summary 记录实际 Memory snapshot provenance，事件包含 retrieve、select、write，达到边界时包含 maintenance。

关闭 `memory.enabled: false` 时，对 fake 无 Memory run 审计以下证据：

- manifest 的 `rollout_options.memory_enabled` 和 summary 顶层的 `memory_enabled` 都是 `false`。
- manifest 的 `memory_snapshot_id`、summary 的 `input_memory_snapshot_id` 与 `output_memory_snapshot_id` 都是 `null`；`maintenance_rounds_executed` 是空数组。
- 每个 episode 的 `EpisodeStarted` 后出现 `MemoryDisabled`，其 `reason` 为 `config`；不得出现 `MemoryCandidatesRetrieved`、`MemorySelected`、任何 Memory write 或 maintenance 事件。
- run 可以保留 trajectory、官方 terminal reward/result 以及 `FastLoopFailed` 失败证据；但项目根目录不得新建或修改 `history/agents/<agent_id>/memory/`，也不得产生 snapshot 或 maintenance state。

可用以下 PowerShell 检查 disabled 运行：

```powershell
$run = "stage4-memory-disabled-0"
$manifest = Get-Content "runs/$run/manifest.json" | ConvertFrom-Json
$summary = Get-Content "runs/$run/fast_loop_summary.json" | ConvertFrom-Json
$events = Get-Content "runs/$run/rollouts/events.jsonl" | ForEach-Object { $_ | ConvertFrom-Json }
$manifest.rollout_options.memory_enabled
$summary.memory_enabled, $summary.input_memory_snapshot_id, $summary.output_memory_snapshot_id, $summary.maintenance_rounds_executed
$events | Where-Object event_type -eq "MemoryDisabled" | Select-Object task_id,reason
$events | Where-Object event_type -match "^Memory(CandidatesRetrieved|Selected|Write)|^Maintenance" | Select-Object event_type,task_id
Test-Path "history/agents/retail/memory"
```

最后一条必须为 `False`，且前一个事件查询不返回任何记录。`memory_enabled=false` 的 run 只可用于无 Memory baseline/ablation 指标比较；Stage 5 attribution 和 OPD dataset builder 不得消费它作为 Memory 生命周期监督来源。

## 累计维护调度

`--completed-train-tasks-before` 是 required 参数，必须显式给出同一共享 agent namespace 在本次调用前已成功完成的 train 任务总数。遗漏会在 argparse 阶段失败，不会静默默认为 0。CLI 不从 run ID 或目录猜测该值。summary 同时记录 `completed_train_tasks_before` 和 `completed_train_tasks_after`。

- `0 + 5`：五任务运行依次检查累计值 1 到 5，不执行 maintenance。
- `25 + 5`：前四个任务检查 26 到 29；第五个任务成功后累计值为 30，执行且只执行 maintenance round 1。

对应的触发命令只需把五任务命令改为：

```powershell
  --run-id stage4-train-25-29 --completed-train-tasks-before 25
```

maintenance 只在 episode 成功后调度。episode 失败时保留已有 rollout evidence，原异常继续抛出，不写 `fast_loop_summary.json`。

Qwen HTTP 请求默认使用 120 秒有限超时。可选的五任务真实集成 smoke 对整个 CLI 子进程使用 1800 秒上限，超时会直接使测试失败，避免 CI 无限等待。

## 输出与检查

默认输出如下：

- `runs/<run_id>/manifest.json`：schema version 2；记录 model、adapter、输入 Memory snapshot、tau2 commit、官方 split hash、任务列表、模拟器配置和固定生成契约。
- `runs/<run_id>/rollouts/events.jsonl`：逐条 fsync 的 canonical rollout/maintenance evidence。
- `runs/<run_id>/fast_loop_summary.json`：run ID、episode 数、terminal reward 总和、成功任务 ID、实际执行的 maintenance rounds、累计任务数和输入/输出 snapshot ID。
- `history/agents/retail/memory/*_memory.json`：trajectory、tip、skill、tool 四层共享 JSON Memory。
- `history/agents/retail/memory/snapshots/<snapshot_id>/`：不可变输入或输出快照及其 `manifest.json`。
- `history/agents/retail/memory/maintenance_state.json`：至少执行过一次 maintenance 后出现，记录已完成 round。

每个成功 episode 的预期顺序是：

1. `EpisodeStarted`
2. `MemoryCandidatesRetrieved`
3. `MemorySelected`
4. 一组 `DecisionMade`、`EnvironmentStepped`
5. `EpisodeFinished`
6. `MemoryWriteProposed`
7. `MemoryWriteCommitted`

到达维护边界时，随后出现 `MaintenanceStarted`、`MaintenanceProposed`、`MaintenanceCommitted`。可以用 PowerShell 检查关键证据：

```powershell
$events = Get-Content runs/stage4-train-0-4/rollouts/events.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
$events | Select-Object event_type,task_id,final_reward,memory_snapshot_id
$events | Where-Object event_type -eq "MemoryCandidatesRetrieved" | Select-Object task_id,candidates
$events | Where-Object event_type -eq "MemorySelected" | Select-Object task_id,selected
$events | Where-Object event_type -eq "MemoryWriteProposed" | Select-Object task_id,proposals
Get-Content runs/stage4-train-0-4/manifest.json | ConvertFrom-Json | Select-Object adapter_revision,memory_snapshot_id
Get-Content runs/stage4-train-0-4/fast_loop_summary.json | ConvertFrom-Json
```

`EpisodeFinished.final_reward`、`terminal_evaluation` 和 `simulation_result` 来自官方 tau2 terminal fields。`MemoryWriteProposed.proposals[*].source_task_ids` 以及 Memory JSON 中的 `source_task_ids` 用于核对任务 provenance。若没有写入或维护变更，输入和输出 snapshot ID 可以相等；发生 canonical Memory 变更时二者必须不同。

Stage 4 **不计算 Outcome-calibrated attribution**。Stage 5 才会创建 `runs/<run_id>/attribution/scores.jsonl`。
