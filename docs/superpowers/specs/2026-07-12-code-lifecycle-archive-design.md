# 阶段代码生命周期与归档设计

## 目标

在不削弱 Tau2 Retail 环境兼容性和回归保护的前提下，清理阶段一、阶段二遗留的验证入口与测试辅助代码，建立后续开发必须遵守的代码生命周期规则。核心运行代码、产品级命令、运维诊断工具、持续测试和历史记录必须拥有清晰且互不混杂的目录边界。

## 核心原则

每个阶段结束前，新增文件必须被明确归入以下类别之一：

1. `src/`：后续 rollout、memory、OPD 训练或评测直接依赖的核心运行代码。禁止放置只供测试或阶段验收使用的实现。
2. `scripts/`：用户长期执行的产品级工作流入口，例如 baseline、fast loop、OPD 训练和正式评测。一次性 smoke、迁移和排障命令不得放在这里。
3. `tools/`：部署前检查、环境诊断、数据迁移和人工排障等长期运维工具。工具不得成为核心运行链的隐式依赖。
4. `tests/`：持续保护当前有效契约的自动化测试及其 fixtures、fakes、stubs 和测试支持代码。测试辅助实现不得由 `src/` 导出。
5. `docs/`：当前有效的设计、操作规范和压缩后的历史决策。原始 review diff、临时 brief、执行报告和重复日志不属于正式文档。

仅为阶段推进、一次性比较或临时 review 生成的文件，在完成目的后必须删除。Git 已跟踪内容由 Git 历史保留；被忽略的本地 scratch 内容不构成项目交付，也不得复制进正式文档。确有长期审计价值的结论应压缩成 `docs/archive/` 中的里程碑摘要。

## 阶段结束归档门禁

每个实施阶段的完成条件必须包含一次生命周期审计：

1. 搜索新增模块的生产引用和测试引用。
2. 仅被测试引用的实现移入 `tests/support/` 或对应测试模块。
3. 仅供人工诊断但仍有长期价值的命令移入 `tools/`。
4. 已被正式工作流取代且无独立运维价值的命令直接删除。
5. 核心代码、产品命令和测试不得依赖 `docs/archive/`、`.superpowers/` 或其他本地 scratch 目录。
6. 更新命令文档和测试引用后运行完整默认测试，并在具备凭证时运行相关真实集成门禁。
7. 在阶段提交中记录保留、迁移和删除的文件清单。

未通过该门禁的阶段不得标记为完成，也不得继续扩展下一个阶段的核心目录。

## 阶段一与阶段二盘点

### 保留在核心目录

以下实现会被后续 fast loop、slow loop 或正式评测直接复用，继续保留：

- `src/tau3_retail_evolver/config.py`
- `src/tau3_retail_evolver/envs/base.py`
- `src/tau3_retail_evolver/envs/runtime.py`
- `src/tau3_retail_evolver/envs/split_guard.py`
- `src/tau3_retail_evolver/envs/task_catalog.py`
- `src/tau3_retail_evolver/envs/tau2_retail.py`
- `src/tau3_retail_evolver/fast_loop/action_codec.py`
- `src/tau3_retail_evolver/fast_loop/baseline_prompt.py`
- `src/tau3_retail_evolver/fast_loop/baseline_runner.py`
- `src/tau3_retail_evolver/fast_loop/events.py`
- `src/tau3_retail_evolver/io/jsonl.py`
- `src/tau3_retail_evolver/models/openai_compatible.py`
- `src/tau3_retail_evolver/models/policy.py` 中的 `DecisionRequest`、`DecisionResponse` 和 `Policy`
- `src/tau3_retail_evolver/runs/manifest.py`
- `scripts/run_baseline.py`
- `external/tau2-bench.commit`

### 移入长期运维工具

`scripts/check_tau2_retail.py` 仍然具有新机器部署、Tau2 revision 升级和用户模拟器故障诊断价值，但不属于产品工作流。它迁移为：

```text
tools/preflight/check_tau2_retail.py
```

长期命令改为：

```bash
python -m tools.preflight.check_tau2_retail --split train --task-id 0
```

工具继续输出机器可读、凭证脱敏的 inspect/reset/close 摘要。核心训练和 rollout 代码不得导入该工具。

### 移入测试支持目录

`ScriptedPolicy` 只在单元测试中使用，不属于可部署 policy API。它从 `src/tau3_retail_evolver/models/policy.py` 及模型公开导出中移除，迁移到：

```text
tests/support/policy.py
```

所有使用方改为从测试支持模块导入。生产包不再包含 scripted response 队列或请求捕获逻辑。

### 删除无行为包装层

`src/tau3_retail_evolver/envs/factory.py` 仅被一个单元测试引用，函数只是直接构造 `Tau2RetailEnv`，没有提供生命周期、配置或依赖边界。该文件删除，测试直接构造 `Tau2RetailEnv`。

### 保留持续回归测试

以下内容不是阶段性垃圾，必须继续保留：

- `tests/integration/test_real_tau2_retail.py`：保护固定 checkout、官方 split、真实 reset/close 和用户模拟器契约。
- 环境、配置、Qwen policy、action、baseline、JSONL 和 manifest 单元测试。
- `tests/fixtures/tau2_retail/` 中用于稳定验证 split 解析的 fixture。

诊断工具的单元测试迁移到 `tests/unit/tools/preflight/`，集成测试更新为调用新的模块路径。

## 文档归档

新增 `docs/development/code-lifecycle.md` 作为长期核心规范，并在 OPD-Evolver 设计文档和分阶段实施计划中显式引用。分阶段计划的每个 stage gate 都必须继承该规范。

新增 `docs/archive/2026-07-stage-1-2-delivery.md`，只记录：

- 阶段一、阶段二已完成的长期交付边界。
- 被迁移或删除的验证资产。
- 对应 commit 范围和测试门禁结果。

`.superpowers/` 已由 `.gitignore` 排除，继续作为本地工作区，不进入交付清单。原始 review diff、brief、report 和本地 progress 不迁入 `docs/archive/`。

## 兼容性与命令迁移

旧命令：

```bash
python -m scripts.check_tau2_retail ...
```

在本次整理后不再作为兼容入口保留。项目当前没有外部发布版本或稳定 CLI 契约，保留转发模块只会延长错误目录边界。所有项目文档和测试一次性更新到新命令。

`scripts.run_baseline` 保持不变，因为它仍是阶段二真实 baseline 和后续对照实验的产品级入口。

## 测试策略

整理过程采用行为保持式重构：

1. 先修改测试，使其引用目标目录中的 preflight 工具和测试支持 policy，并验证因目标模块尚不存在而失败。
2. 迁移实现并删除旧导出，使聚焦测试恢复通过。
3. 删除无行为 factory 后更新适配器测试。
4. 运行完整默认测试。删除无行为 factory 及其唯一包装层测试后，预期结果为 `93 passed, 2 skipped`。
5. 设置 `RUN_TAU2_INTEGRATION=1` 且凭证可用时运行真实 Tau2 集成测试，预期 `2 passed`。
6. 使用 `rg` 确认旧模块路径、旧命令和生产侧 `ScriptedPolicy` 引用全部消失。

## 验收标准

- `scripts/` 只保留产品级工作流入口。
- Tau2 环境诊断命令位于 `tools/preflight/`，功能和脱敏行为不变。
- `src/` 不再导出或包含 `ScriptedPolicy`。
- 无行为的 `envs/factory.py` 被删除。
- 长期生命周期规范被现有设计和阶段计划引用。
- Stage 1/2 历史只保留压缩摘要，不把本地过程资产纳入版本控制。
- 默认测试通过；真实集成门禁的运行条件和结果被明确记录。
