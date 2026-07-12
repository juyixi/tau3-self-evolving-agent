# 2026-07 Stage 1/2 交付归档

## 范围

本摘要记录 Tau3 Retail OPD-Evolver 项目阶段一、阶段二 2.1/2.2 的长期交付边界，以及阶段完成后的代码生命周期整理。详细实现演进由 Git 历史保留。

## 阶段一：Tau2 Retail 环境接入

提交范围：`31bc423` 至 `e9735e2`。

长期保留的交付包括：

- 严格类型化的项目配置和 train/test split 约束。
- 固定 Tau2 checkout commit、包版本、数据路径和导入来源检查。
- 官方 Retail 任务目录、74/40/114 split 兼容性及 SHA-256 校验。
- `Tau2RetailEnv` Gym 适配器、reset/step/close 生命周期和错误上下文。
- DeepSeek V4 Pro 非 solo 用户模拟器真实 reset/close 集成门禁。
- 固定外部依赖 revision 的 `external/tau2-bench.commit`。

归档处理：

- Tau2 smoke 命令从产品入口 `scripts/` 迁移到 `tools/preflight/`。
- 真实 Tau2 集成测试继续保留，作为长期环境契约门禁。
- 无行为的 `envs/factory.py` 及其唯一包装层测试被删除。

## 阶段二：无 Memory Qwen Baseline

提交范围：`58f8709` 至 `3eff24c`。

长期保留的交付包括：

- Qwen OpenAI-compatible policy 和严格 HTTP/structured tool-call 边界。
- Tau2 policy、工具和公开 observation 的 baseline prompt。
- thinking 清理、工具名验证和 Tau2 stop action 编解码。
- 无 memory、无 adapter 的逐 turn baseline runner。
- schema v1 rollout events、耐久 JSONL 和不可覆盖的原子 manifest。
- 模型 revision、Tau2 commit、split hash、seed 和用户模拟器配置溯源。
- API key、token、密码和凭证 URL 的 artifact 脱敏。

归档处理：

- 只供单元测试使用的 `ScriptedPolicy` 从生产模型包迁移到 `tests/support/`。
- `scripts/run_baseline.py` 继续保留，因为它仍是 Qwen baseline 和后续对照实验入口。

## OpenRouter 单任务验证

以下步骤验证一个真实的 `train` 任务。先按既有 Qwen vLLM 隧道操作保持服务运行，并在控制机当前 PowerShell 会话中复用其 `QWEN_BASE_URL`：

```powershell
$env:OPENROUTER_API_KEY = Read-Host "OpenRouter API Key"
$env:QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
python -m scripts.run_baseline `
  --split train `
  --task-id 0 `
  --run-id baseline-openrouter-20260712-01 `
  --qwen-base-url $env:QWEN_BASE_URL `
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

`OPENROUTER_API_KEY` 只需配置在运行 `python -m scripts.run_baseline` 的 Windows/控制机上；只通过 vLLM 提供 Qwen 服务的 AutoDL 机器不需要该变量。每次运行都必须使用新的不可变 `--run-id`；失败的 run ID 不支持重跑。

运行后检查 `runs/baseline-openrouter-20260712-01/manifest.json`：其中应有 `schema_version: 2` 和 `evaluation_config.nl_assertions.model: openrouter/openai/gpt-4.1`，但绝不能出现 key 值。进程 summary 成功返回，并且 `rollouts/events.jsonl` 含带 reward 字段的 `EpisodeFinished` 事件，才说明该管线完成了这次验证；本文不宣称真实 OpenRouter 运行已经通过。

## 生命周期整理

设计与实施提交：

- `3d1f072`：定义代码生命周期归档设计。
- `59d76ba`：提交详细归档实施计划。
- `1dd2329`：迁移 Tau2 preflight 工具。
- `5f1e882`：移出测试 policy 并删除无行为 factory。

本次整理遵循 `docs/development/code-lifecycle.md`。`.superpowers/` 中的 review diff、brief、report 和 progress 属于被忽略的本地工作区，没有复制到正式归档。

整理前默认门禁为 `94 passed, 2 skipped`。删除一个只验证 factory 转发存在的测试后，整理完成门禁固定为 `93 passed, 2 skipped`；真实 Tau2 集成门禁在配置凭证和 `RUN_TAU2_INTEGRATION=1` 时要求 `2 passed`。
