# Tau3 Retail OPD-Evolver 分阶段实施计划

> **2026-07-28 修订：** `sel/act/write/maint` 现在分别训练独立 LoRA，禁止按最大
> 类别过采样补齐。单个能力内部仍由同一 Qwen3.5-9B 与该能力当前 LoRA 执行
> teacher/student OPD。现行契约见
> [四能力 LoRA OPSD 训练设计](../../four-lora-opd-training.md)。

> **供智能体实施者使用：** 必须使用子技能 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，逐任务实施本计划。各步骤使用 checkbox（`- [ ]`）跟踪状态。

**目标：** 分八个可独立测试的阶段构建并验证 tau3 retail 训练系统，覆盖官方 tau2-bench 环境接入、OPD-Evolver 快循环记忆、共享策略 LoRA 慢循环训练，以及留出 retail 任务评测。

**架构：** 使用严格的适配器和 split guard 封装官方 `sierra-research/tau2-bench` retail Gym 环境。仅由 train 同策略 rollout 驱动文件持久化的四层记忆快循环和 OPD-Evolver 归因/数据管线；一个 Qwen3.5-9B 和一个当前 LoRA 适配器同时执行学生 forward 与特权 stop-gradient 教师 forward。评测使用独立的只读或隔离管线，test 信息不能进入记忆归因、数据集、checkpoint 或 checkpoint 选择。

**技术栈：** Python 3.12、uv、pytest、Pydantic、PyYAML、Gymnasium/tau2、JSON memory 加 JSONL 日志/训练数据、PyTorch、由 uv 锁定的最新版 Qwen3.5-compatible Transformers revision、PEFT、Accelerate，以及用于 rollout 推理的 OpenAI-compatible vLLM 或 SGLang endpoint。

## 全局约束

- 算法参考资料：`C:\Users\huang\Downloads\2606.17628v1.pdf`、https://arxiv.org/abs/2606.17628v1 和 https://github.com/bingreeky/opd-evolver。
- 环境参考资料：https://github.com/sierra-research/tau2-bench 及其 `tau2.gym.gym_agent.AgentGymEnv`。
- 外部 tau2-bench checkout 默认位于 `external/tau2-bench/`，由 git 忽略，并在每个 run manifest 中记录固定 commit。
- `train` 是唯一学习 split。`test` 仅用于评测。`base` 仅用于可选的官方聚合复现，训练命令不得接受它。
- 默认不创建内部 dev split。
- 除非显式配置本地镜像，否则基础模型必须是 `Qwen/Qwen3.5-9B`。
- 学生和教师是同一个模型对象、同一个当前 LoRA 适配器上的两条条件输入路径。教师路径为 stop-gradient，而不是单独训练的教师 checkpoint。
- LoRA 默认值为 `use_peft=true`、`lora_r=32`、`lora_alpha=64` 和 `lora_dropout=0.05`。必须拒绝全参数微调。
- OPD 生成默认值为 `temperature=1.0`、`top_p=0.95` 和 `max_episode_steps=40`。
- Memory 默认值：层级为 `trajectory`、`tip`、`skill`、`tool`；检索 50 条；教师上限 20 条；分数阈值 0.01；维护周期 `Q=30`。
- 单元测试不依赖网络、API 凭证、Qwen 权重、GPU 或 tau2 checkout。
- 真实环境、模型和 GPU 测试使用显式 pytest marker，绝不进入默认单元测试套件。
- 每个阶段都以测试 gate 通过和一次 commit 结束。前一阶段 gate 为红色时，不得开始后一阶段。
- 每个阶段结束时必须执行 [`docs/development/code-lifecycle.md`](../../development/code-lifecycle.md) 定义的生命周期审计；验证工具、测试辅助实现和历史资产必须在阶段完成前归入正确目录或删除。

## 已确定决策

- 虽然项目面对用户的文档将任务称为 tau3 retail，但集成的是当前官方环境包 `tau2`。
- 固定 revision 下的官方 split 文件是任务 ID 的唯一来源。当前兼容性检查预期 74 个 train、40 个 test 和 114 个 base ID。
- 用于归因的训练任务分组使用特权离线签名，该签名由任务要求的非只读 evaluator action 名称生成，绝不暴露给学生 prompt。
- 与 OPD-Evolver 官方仓库一致，四层可变 memory 分别以普通 JSON 文件作为权威存储；运行日志、归因和 OPD 训练数据使用只追加 JSONL。版本化 JSON snapshot 用于检查、训练溯源和 checkpoint 打包。
- Memory JSON 的修改必须先完整校验内存状态，再通过同目录临时文件、flush/fsync 和 `os.replace` 原子替换。单层文件更新失败时旧文件必须保持可读；跨层写入不宣称数据库式 ACID，依靠稳定 ID、幂等重试和 JSONL 生命周期事件恢复。
- 真实检索使用可插拔 embedding retriever，并以 `Qwen/Qwen3-Embedding-0.6B` 作为与论文对齐的默认值。单元测试使用确定性的 fake embedding。
- 慢循环 batch 由当前学生 checkpoint 生成，并且只在当前 iteration 中消费。LoRA checkpoint 更新后不得重复使用该 batch。
- 禁止在 `public_input + privileged_input` 上执行普通 causal SFT。OPD loss 只能是对齐后的学生采样 response token 上的全词表 `KL(teacher || student)`。

## 阶段依赖关系

1. 环境接入
2. 无 memory baseline rollout
3. 四层 memory 基础
4. 完整快循环
5. 归因与 OPD 数据集
6. 共享模型慢循环 LoRA 训练
7. 快/慢循环迭代编排
8. 留出集评测与报告

各阶段按验收 gate 顺序执行。同一阶段内，写入文件范围不重叠的纯单元测试任务可以独立实施。

## 运行时产物约定

每个 `runs/<run_id>/` 包含：

- `manifest.json`：模型和适配器 revision、tau2 commit、split hash、任务 ID、seed、用户模拟器、环境选项、memory snapshot、parent checkpoint 和运行命令。
- `rollouts/events.jsonl`：只追加的决策与环境事件。
- `runs/<run_id>/` stores rollout JSONL, attribution JSONL, OPD examples, checkpoints, evaluation output and manifest only.
- `history/agents/<agent_id>/memory/` stores the continuously evolving four-tier Memory, embedding cache and immutable snapshots.
- `history/evaluations/<run_id>/<agent_id>/quarantine/` stores streaming evaluation Memory that training loaders must reject.
- `attribution/scores.jsonl`。
- `opd_examples/{sel,act,write,maint}.jsonl`。
- `checkpoints/iteration-<n>/adapter/`。
- `eval/<protocol>/episodes.jsonl` 和 `eval/<protocol>/summary.json`。

所有记录都包含 `run_id`、`iteration`、`split`、`task_id`、`task_group`、`model_revision`、`adapter_revision`、`memory_snapshot_id` 和 `seed`。

---

## 阶段 1：接入官方 Tau2 Retail 环境

**产出：** 能够通过项目适配器对真实 train 任务执行 reset、step、evaluate 和 close；错误使用 split 时，在创建环境前立即失败。

### 任务 1.1：最小项目与环境配置

**文件：**
- 创建：`pyproject.toml`
- 创建：`.gitignore`
- 创建：`configs/default.yaml`
- 创建：`src/tau3_evolver/config.py`
- 测试：`tests/unit/test_config.py`

**接口：**
- 产出：`load_config(path: Path, overrides: Sequence[str] = ()) -> ProjectConfig`
- 产出：`Tau2Config(repo_path, domain, train_split, eval_split, user_llm, user_llm_args, solo_mode)`
- 产出：`ModelConfig`、`LoraConfig`、`RolloutConfig`、`MemoryConfig` 和 `TrainingConfig`

- [ ] 编写失败测试，断言 Python 3.12、`domain="retail"`、`train_split="train"`、`eval_split="test"`、不存在 `dev` 字段、使用 Qwen3.5-9B，并且 LoRA 默认值完全正确。
- [ ] 运行 `pytest tests/unit/test_config.py -v`，确认因 import 或配置缺失而失败。
- [ ] 实现带类型的配置加载和验证。拒绝 `use_peft=false`，并拒绝 `train` 以外的训练 split。
- [ ] 将 `external/`、`runs/`、模型产物、cache 和本地 secret 加入 `.gitignore`。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `chore: bootstrap tau3 retail environment config`。

### 任务 1.2：Tau2 Runtime 探测、任务目录与 Split Guard

**文件：**
- 创建：`src/tau3_evolver/envs/runtime.py`
- 创建：`src/tau3_evolver/envs/task_catalog.py`
- 创建：`src/tau3_evolver/envs/split_guard.py`
- 测试：`tests/unit/envs/test_task_catalog.py`
- 测试 fixture：`tests/fixtures/tau2_retail/split_tasks.json`

**接口：**
- 产出：`Tau2Runtime.inspect(repo_path: Path) -> RuntimeFingerprint`
- 产出：`RetailTaskCatalog.from_files(tasks_path, split_path) -> RetailTaskCatalog`
- 产出：`task_ids(split: Literal["train", "test", "base"]) -> tuple[str, ...]`
- 产出：`require_learning_split(split: str) -> None`

- [ ] 编写失败测试，覆盖 train/test 不相交、`base == train union test`、74/40/114 兼容性数量、稳定的 split SHA-256，以及 `require_learning_split` 对 `test`/`base` 的拒绝行为。
- [ ] 运行 `pytest tests/unit/envs/test_task_catalog.py -v`，确认失败。
- [ ] 使用结构化方式解析官方 JSON；禁止在源码中复制任务 ID。
- [ ] 实现 runtime probe，在一个可操作的错误信息中报告仓库路径、Git commit、package version、数据路径和 Gym 可用性。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add tau2 retail task catalog and split guard`。

### 任务 1.3：Gymnasium 适配器

**文件：**
- 创建：`src/tau3_evolver/envs/base.py`
- 创建：`src/tau3_evolver/envs/tau2_retail.py`
- 创建：`src/tau3_evolver/envs/factory.py`
- 测试：`tests/unit/envs/test_tau2_retail_adapter.py`

**接口：**
- 产出：`Tau2RetailEnv(task_id, config, gym_factory=None)`
- 产出：`reset(seed: int) -> ResetResult`
- 产出：`step(action: str) -> StepResult`
- 产出：`close() -> None` 和 context-manager 支持
- 规范化：将 `terminated or truncated` 合并为 `done`，同时保留两个原始 flag

- [ ] 编写 fake Gym 环境及失败测试，覆盖构造参数、reset info（`task`、`tools`、`policy`）、五返回值 step 规范化、parse error 保留、官方 `reward_info` 保留、最大步数截断和幂等 close。
- [ ] 运行 `pytest tests/unit/envs/test_tau2_retail_adapter.py -v`，确认失败。
- [ ] 实现 lazy `tau2` import 和依赖注入；任何单元测试都不得导入真实 package。
- [ ] 确保每个异常都包含 split、task ID、episode step 和原始异常原因。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: integrate tau2 retail gym adapter`。

### 任务 1.4：真实环境 Smoke Gate

**文件：**
- 创建：`tools/preflight/check_tau2_retail.py`
- 创建：`tests/integration/test_real_tau2_retail.py`
- 修改：`pytest.ini` 或 `pyproject.toml` 中的 marker

- [ ] 添加 `tau2_integration` marker；仅当 `RUN_TAU2_INTEGRATION=1` 时运行。
- [ ] 让脚本输出 tau2 commit、split hash、选中的 train task ID、tool 数量、policy hash、初始 observation 长度和解析后的用户模拟器配置。
- [ ] 使用 `uv pip install -e "external/tau2-bench[gym]"` 安装固定版本的 checkout。
- [ ] 在用户模拟器凭证和配置有效时运行 `python -m tools.preflight.check_tau2_retail --split train --task-id 0`。
- [ ] 运行 `pytest -m tau2_integration tests/integration/test_real_tau2_retail.py -v`，确认真实 reset/close 流程通过。
- [ ] 将 pin 写入 `external/tau2-bench.commit`；只提交该文本 pin，绝不提交外部 checkout。
- [ ] 提交为 `test: verify real tau2 retail environment`。

**阶段 1 gate：** 默认单元测试通过；可选的真实 reset/close smoke 通过；训练命令在创建环境前拒绝 test/base。

---

## 阶段 2：无 Memory 的 Qwen Baseline 与规范化 Rollout 数据

**产出：** 当前 Qwen3.5-9B checkpoint 可以在不使用 memory 的情况下运行官方 train 任务，并且每个 turn 都具有可复现日志。

### 任务 2.1：Policy、Prompt 与 Action 边界

**文件：**
- 创建：`src/tau3_evolver/models/policy.py`
- 创建：`src/tau3_evolver/models/openai_compatible.py`
- 创建：`src/tau3_evolver/fast_loop/action_codec.py`
- 创建：`src/tau3_evolver/fast_loop/baseline_prompt.py`
- 测试：`tests/unit/models/test_policy.py`
- 测试：`tests/unit/fast_loop/test_action_codec.py`

**接口：**
- 产出：`Policy.generate(request: DecisionRequest) -> DecisionResponse`
- 产出：`Tau2ActionCodec.decode(model_output: str, tool_names: set[str]) -> str`
- 消费 reset info 中的官方 tau2 policy 和 tools

- [ ] 编写失败测试，覆盖普通用户消息、JSON tool call、函数式 tool call、未知 tool、格式错误的参数、stop action，以及不会丢弃 final answer 的 thinking block 移除。
- [ ] 运行聚焦测试并确认失败。
- [ ] 实现 fake policy 和 OpenAI-compatible Qwen endpoint client。记录原始输出、解析后的 action、采样参数和 latency。
- [ ] 在 serving 层使用官方 Qwen tool-call parser；项目 codec 仍需保持防御性，因为 tau2 同时接受 JSON 和函数式字符串。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add qwen policy and tau2 action boundary`。

### 任务 2.2：Episode Schema、Manifest 与 Baseline Runner

**文件：**
- 创建：`src/tau3_evolver/io/jsonl.py`
- 创建：`src/tau3_evolver/runs/manifest.py`
- 创建：`src/tau3_evolver/fast_loop/events.py`
- 创建：`src/tau3_evolver/fast_loop/baseline_runner.py`
- 创建：`scripts/run_baseline.py`
- 测试：`tests/unit/fast_loop/test_baseline_runner.py`

**接口：**
- 产出：`run_baseline(tasks, env_factory, policy, run_context) -> RolloutSummary`
- 产出只追加事件：`EpisodeStarted`、`DecisionMade`、`EnvironmentStepped` 和 `EpisodeFinished`

- [ ] 编写确定性的双任务失败测试，断言事件顺序、最终 reward、官方 evaluator 详情、seed、task group、model revision 和环境 close。
- [ ] 运行聚焦测试并确认失败。
- [ ] 实现原子化 manifest 创建和 schema version `1` 的只追加 JSONL 写入。
- [ ] 添加 `--split train` 并强制执行 `require_learning_split`。baseline 命令不得接受 `test` 或 `base`。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add canonical no-memory rollout pipeline`。

### 任务 2.3：真实三任务 Baseline

- [ ] 使用 language-only serving、Qwen reasoning parser 和 tool-call parser 启动 Qwen3.5-9B。
- [ ] 使用 `temperature=1.0`、`top_p=0.95` 和 `max_episode_steps=40` 运行三个固定 train ID。
- [ ] 验证每个 episode 都包含官方 reward 详情，不存在遗留环境线程，并且 manifest 完整。
- [ ] 将输出存放在 `runs/baseline-<timestamp>/`；不提交运行时数据。
- [ ] 对所有集成修复添加聚焦回归测试后再提交。

**阶段 2 gate：** 真实三任务 train baseline 完成，并可根据 manifest 和 events 审计重放过程。

---

## 阶段 3：四层 Memory 基础

**产出：** trajectory、tip、skill 和 tool memory 可以原子文件持久化、检索、版本化、维护和快照。

### 任务 3.1：Memory 类型与 JSON Repository

**文件：**
- 创建：`src/tau3_evolver/memory/types.py`
- 创建：`src/tau3_evolver/memory/repository.py`
- 创建：`src/tau3_evolver/memory/json_store.py`
- 测试：`tests/unit/memory/test_repository.py`

**接口：**
- 产出：`MemoryItem(id, tier, content, version, status, source_task_ids, created_round, updated_round, metadata)`
- 产出：`MemoryRepository.add/get/list/update_status/snapshot`
- 强制层级：`trajectory`、`tip`、`skill` 和 `tool`

- [ ] 编写失败测试，覆盖所有层级、稳定 ID、provenance、重复项拒绝、version 递增、active/retired 过滤、原子替换失败后旧文件仍可读取、重启恢复和确定性 snapshot 导出。
- [ ] 运行聚焦测试并确认失败。
- [ ] 实现四个层级 JSON store；每个文件包含 `schema_version`、`tier`、`count` 和 `items`，每条 item 保存 content、embedding、metadata、source task、version、status 和使用统计。写入使用同目录临时文件和 `os.replace`。
- [ ] 按稳定的 `tier,id` 顺序和规范化 JSON 编码导出四层 snapshot，并对 snapshot manifest 和文件内容计算 hash 作为 `memory_snapshot_id`。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add four-tier json memory repository`。

### 任务 3.2：检索与候选日志

**文件：**
- 创建：`src/tau3_evolver/memory/retrieval.py`
- 创建：`src/tau3_evolver/memory/embeddings.py`
- 测试：`tests/unit/memory/test_retrieval.py`

**接口：**
- 产出：`Retriever.retrieve(query, repository, top_k=50) -> list[MemoryCandidate]`
- 候选字段包括 rank、similarity、tier、memory version、retriever revision 和 query hash

- [ ] 使用 fake vector 编写失败测试，覆盖确定性排序、tie、跨层级多样性、排除 retired memory 和精确 top-k。
- [ ] 运行聚焦测试并确认失败。
- [ ] 实现 embedding 接口和与论文对齐的 Qwen3-Embedding 配置，单元测试不得加载真实模型。
- [ ] 记录所有检索候选，而不只是选中的 memory；归因依赖被检索但未被选择的对照项。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add memory retrieval and candidate provenance`。

### 任务 3.3：Lookup、Merge、Delete 与 Snapshot 隔离

**文件：**
- 创建：`src/tau3_evolver/memory/operations.py`
- 创建：`src/tau3_evolver/memory/read_only.py`
- 测试：`tests/unit/memory/test_operations.py`

- [ ] 编写失败测试，覆盖结构化 lookup、同层级 merge、拒绝跨层级 merge、soft delete、provenance 合并、无效批次不修改原文件，以及只读 snapshot 拒绝修改。
- [ ] 将 operation 实现为带类型的 command，绝不能直接把未经校验的模型输出写入 Memory JSON。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add memory lifecycle operations`。

**阶段 3 gate：** repository 重开后仍可用，能够导出稳定 snapshot、确定性检索 50 条候选，并通过只读评测视图阻止所有写入。

---

## 阶段 4：OPD-Evolver 快循环

**产出：** train 任务使用当前策略输出，执行完整的 retrieve/select/act/write/maintain 生命周期，并保留完整 provenance。

### 任务 4.1：带类型的生命周期决策与 Prompt

**文件：**
- 创建：`src/tau3_evolver/fast_loop/decisions.py`
- 创建：`src/tau3_evolver/fast_loop/prompts.py`
- 测试：`tests/unit/fast_loop/test_decisions.py`
- 测试：`tests/unit/fast_loop/test_prompts.py`

**接口：**
- 产出 Pydantic 输出：`SelectionDecision`、`ActionDecision`、`WriteDecision`、`MaintenanceDecision`
- 产出 `sel`、`act`、`write` 和 `maint` 的 prompt builder

- [ ] 编写失败测试，证明公开 prompt 包含 task、官方 policy/tools、observation/history 和允许使用的 memory 内容，但绝不包含 attribution score、evaluator criteria、test ID 或 privileged hindsight。
- [ ] 编写 parser 测试，覆盖有效 JSON、未知 ID、重复选择、无效 tier、不安全 maintenance operation 和 retry 耗尽。
- [ ] 实现一次 repair 尝试；仍失败后记录 no-op/failure，禁止静默强制转换无效生命周期输出。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add typed fast-loop lifecycle decisions`。

### 任务 4.2：Episode 快循环 Runner

**文件：**
- 创建：`src/tau3_evolver/fast_loop/runner.py`
- 修改：`src/tau3_evolver/fast_loop/events.py`
- 测试：`tests/unit/fast_loop/test_runner.py`

**接口：**
- 产出：`run_fast_loop_episode(task, env, policy, memory, config, context) -> EpisodeResult`
- 事件顺序：retrieve、select、action step、terminal result、write proposal、memory persistence

- [ ] 编写 fake-policy episode 失败测试，证明 selected set 是 candidate 的子集，每个学生 action 都是 on-policy，terminal reward 被保留，并且写入的 memory 引用源 episode。
- [ ] 添加失败测试，覆盖无效 action、环境异常、最大步数截断、policy timeout 和 Memory JSON 原子替换失败。
- [ ] 实现 learning runner，使得只有 split 为 `train` 的 `RunMode.LEARN` 能获得训练 memory 修改能力。阶段 8 仅可通过 evaluation-quarantine capability 复用生命周期逻辑。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add opd evolver fast-loop runner`。

### 任务 4.3：周期性维护

**文件：**
- 创建：`src/tau3_evolver/fast_loop/maintenance.py`
- 测试：`tests/unit/fast_loop/test_maintenance.py`

- [ ] 编写失败测试，证明每完成 30 个 train 任务运行一次 maintenance；其只接收有边界的 repository diagnostics，按层级原子应用 lookup/merge/delete，并记录完整 maintenance 轨迹。
- [ ] 根据已完成 round 数实现调度，保证 resume 不会重复运行同一个 maintenance round。
- [ ] 重新运行测试并确认 PASS。
- [ ] 提交为 `feat: add periodic memory maintenance`。

### 任务 4.4：真实 Train 快循环 Smoke

- [x] 初始化一个小型四层 repository。
- [x] 使用当前 Qwen LoRA checkpoint 运行五个固定 train 任务。
- [x] 验证 candidate/selected set、环境轨迹、memory 写入、官方 reward 和 snapshot 变化。
- [x] 使用 fake 环境运行 30 任务 scheduler smoke，在不产生外部 API 成本的情况下覆盖 maintenance。

**阶段 4 gate：** 已通过。真实五任务 train run 携带完整 schema-2 生命周期日志完成，形成连续 Memory snapshot chain；任何 test/base 路径都无法修改 train-derived memory。

---

## 阶段 5：结果归因与 OPD 数据集构造

**产出：** 将快循环日志转换为符合论文定义、可审计公开/特权信息隔离的 `sel`、`act`、`write` 和 `maint` 样本。

### 任务 5.1：Retail 任务分组

**文件：**
- 创建：`src/tau3_evolver/slow_loop/task_grouping.py`
- 测试：`tests/unit/slow_loop/test_task_grouping.py`

- [x] 编写失败测试，确保移除只读 lookup tool，规范化并排序所需 action 名称，为语义等价的 action set 生成相同 group，并且绝不包含 evaluator 参数或 expected value。
- [x] 在 rollout 后根据特权任务元数据实现分组。公开事件元数据中只存储 group signature。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: add retail attribution task grouping`。

### 任务 5.2：严格对应论文的 Memory Attribution

**文件：**
- 创建：`src/tau3_evolver/slow_loop/attribution.py`
- 测试：`tests/unit/slow_loop/test_attribution.py`

**接口：**
- 产出：`compute_memory_scores(events, tier_priors) -> dict[str, MemoryScore]`
- 实现论文公式 11-12：
  - `rho_g = n_selected_g / (n_selected_g + n_retrieved_not_selected_g)`
  - `A_hat = sum_g rho_g * (mean(R_selected_g) - mean(R_not_selected_g))`
  - `gamma = 1 - 1 / sqrt(1 + N_selected)`
  - `V = tier_prior * gamma * A_hat`

- [x] 编写合成失败测试，覆盖候选集受控集合、group weighting、空 group 忽略、confidence、tier prior、负值和 0.01 监督阈值。
- [x] 运行聚焦测试并确认失败。
- [x] 根据 retrieved 和 selected event ID 实现。绝不与未检索到该 memory 的任务进行比较。
- [x] 随每个 score 导出 count 和 group delta，确保可以审计。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: add outcome calibrated memory attribution`。

### 任务 5.3：四类 Hindsight Builder

**文件：**
- 创建：`src/tau3_evolver/slow_loop/examples.py`
- 创建：`src/tau3_evolver/slow_loop/leakage.py`
- 测试：`tests/unit/slow_loop/test_examples.py`

**接口：**
- 产出：`OPDExample(kind, public_input, privileged_hindsight, student_output, response_schema, provenance)`
- 对 `sel`、`act`、`write` 和 `maint` 实现公式 13

- [x] 为每个生命周期视图编写失败测试，并断言 privileged value 永远不会出现在 `public_input` 中。
- [x] 对 `act`，公开输入排除 memory；教师 hindsight 在可用时包含有正价值的 selected memory 和同组成功轨迹。
- [x] 对 `write`，只在写入的 memory 具有未来 retrieved/selected 证据后完成样本；不得使用创建它的 episode 为其赋值。
- [x] 对 `maint`，仅在教师 hindsight 中包含 score、confidence、usage 和 redundancy diagnostics。
- [x] 添加严格 leakage scan，拒绝 `split != train`、test task ID、evaluator criteria 和 `eval/` 下的路径。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: build four-view opd hindsight data`。

### 任务 5.4：数据集 CLI 与审计

**文件：**
- 创建：`scripts/build_opd_dataset.py`
- 创建：`scripts/audit_opd_dataset.py`
- 测试：`tests/unit/slow_loop/test_dataset_cli.py`

- [x] 让 builder 为每种决策类型写入独立 JSONL 文件，并生成包含 source run、checkpoint、memory snapshot、count 和 hash 的 dataset manifest。
- [x] 当出现重复 example ID、过期 checkpoint provenance、缺失 student output、privileged/public 重叠或非 train source event 时，让 audit 失败。
- [x] 运行单元测试和合成端到端 build。
- [x] 提交为 `feat: add auditable opd dataset pipeline`。

**阶段 5 gate：** 已通过。Attribution 与手工计算 fixture 一致，四类 builder 通过 leakage 和 provenance audit；真实五任务 build `opd-iter0-5tasks-20260723g` 的独立审计返回 `passed=true`。本次真实小样本产出 4 条 `sel` 和 1 条 `write`，真实 `act/maint` 非空覆盖仍待 30 任务验证。

---

## 阶段 6：使用 LoRA 的共享策略 OPD 慢循环

**产出：** 一个 Qwen3.5-9B 和一个 LoRA 适配器，在相同采样前缀上计算学生与特权教师分布，并且只更新 LoRA 参数。

### 任务 6.1：Qwen3.5 与 PEFT Loader

**文件：**
- 创建：`src/tau3_evolver/models/qwen35.py`
- 创建：`src/tau3_evolver/models/lora.py`
- 测试：`tests/unit/models/test_lora_config.py`
- 集成测试：`tests/integration/test_qwen35_loader.py`

- [x] 编写单元测试，断言 `use_peft=true`、rank 32、alpha 64、dropout 0.05、可训练基础模型参数数目为零，并拒绝全参数微调。
- [x] 使用 text-only 输入，通过当前官方 Transformers model/processor API 加载 Qwen3.5，并把显式 model revision 传给两个 loader。
- [x] 将 `training` extra 的 lower bounds 提升到已发布的 Qwen3.5-compatible Transformers/PEFT 版本；按 Task 6 要求不创建 lockfile。
- [ ] 已添加 opt-in Qwen GPU 集成契约；仍需在缓存真实权重的 BF16 CUDA 机器上执行，确认只有 LoRA 参数需要 gradient 且 artifact 仅含 adapter。
- [x] 提交为 `feat: load qwen35 shared policy with lora`。

### 任务 6.2：前缀对齐与 Token KL

**文件：**
- 创建：`src/tau3_evolver/slow_loop/alignment.py`
- 创建：`src/tau3_evolver/slow_loop/loss.py`
- 测试：`tests/unit/slow_loop/test_alignment.py`
- 测试：`tests/unit/slow_loop/test_loss.py`

**接口：**
- 产出：`build_aligned_batch(example, processor) -> AlignedOPDBatch`
- 产出：`token_kl(student_logits, teacher_logits, student_positions, teacher_positions) -> Tensor`

- [x] 编写 toy-tokenizer 失败测试，证明两条路径都以完全相同的学生采样 response 结尾、response position array 长度相等、prompt/padding token 被 mask，并且 truncation 不会只移除某一侧的 prefix。
- [x] 为全词表 `KL(teacher || student)`、teacher detach、相同 logits 时 loss 为零，以及只在 response token 上求平均编写手工计算测试。
- [x] 实现学生序列 `z + y` 和教师序列 `z + h + y`，并提供显式 response position map。
- [x] 重新运行聚焦测试并确认 PASS。
- [x] 提交为 `feat: add aligned on policy distillation loss`。

### 任务 6.3：共享模型 Teacher/Student Step

**文件：**
- 创建：`src/tau3_evolver/slow_loop/opd_step.py`
- 测试：`tests/unit/slow_loop/test_opd_step.py`

- [x] 构建一个可观测的小型 causal model，并编写失败测试，确认两次 forward 使用同一个 Python model object 和 parameter storage。
- [x] 断言 teacher forward 首先在 `torch.no_grad()` 和 eval mode 下执行，teacher logits 已 detach，student forward 保留 gradient，并且 optimizer 只看到 LoRA 参数。
- [x] 断言教师以 `h` 为条件，但只在学生的 `y_<n` 前缀上评分。
- [x] 实现 two-pass step，并验证 teacher、student 或 loss 异常时可靠恢复 model mode。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: add shared model opd training step`。

### 任务 6.4：Trainer、Checkpoint 与 GPU Smoke

**文件：**
- 创建：`src/tau3_evolver/slow_loop/trainer.py`
- 创建：`scripts/train_opd_lora.py`
- 测试：`tests/unit/slow_loop/test_trainer.py`
- 集成测试：`tests/integration/test_opd_gpu_smoke.py`

- [x] 编写 CPU toy-model 测试，覆盖 gradient accumulation、四类样本 sampling、checkpoint manifest、resume 和仅保存 adapter。
- [x] 实现 batch/epoch 控制，同时保留论文默认值：learning rate 1e-5、per-device batch 2、accumulation 4、训练三轮。
- [x] 拒绝 `source_adapter_revision` 与 trainer 起始 adapter 不一致的数据集。
- [x] opt-in GPU smoke 已在 RTX 4090 48GB 上使用真实 Qwen3.5-9B/BF16 执行一个 optimizer batch。
- [x] LoRA gradient/update、零基础模型 gradient、有限 KL、adapter-only artifact 与逐 tensor reload 断言均已通过。
- [x] 提交为 `feat: train qwen35 lora with shared policy opd`。

**阶段 6 gate：** 已通过。本地 toy、trainer、CLI/dry-run、resume 与 adapter-only 契约通过；真实 Stage 5 数据集审计与 dry-run 通过；Qwen3.5-9B BF16 GPU smoke 于 2026-07-23 完成一个 optimizer step，并验证 adapter tensor 可无损重载。验证中发现并修复了 PEFT 二次筛选生成空 adapter checkpoint 的问题。

---

## 阶段 7：迭代式快/慢循环协同演化

**产出：** 一个可恢复的 iteration 顺序执行 train rollout、memory attribution、OPD 数据、LoRA 更新和 artifact promotion，且不会混用 revision。

### 任务 7.1：Iteration 状态机

**文件：**
- 创建：`src/tau3_evolver/pipeline/state.py`
- 创建：`src/tau3_evolver/pipeline/iteration.py`
- 创建：`src/tau3_evolver/pipeline/executor.py`
- 创建：`scripts/run_iteration.py`
- 测试：`tests/unit/pipeline/test_iteration.py`

**接口：**
- 状态：`created -> rollout_complete -> attribution_complete -> dataset_complete -> training_complete -> promoted`
- 产出：`run_iteration(request, executor, stop_after=None) -> IterationResult`
- 接口：`open_training_memory(config.memory)`；iteration records the returned Repository's pre/post snapshot IDs and never resets it.

- [x] 编写端到端 fake 失败测试，覆盖精确阶段顺序、artifact hash、parent/child revision 和 promoted output。
- [x] 为每个已完成状态编写 resume 测试，并断言已完成阶段通过 hash 验证，而不是重新运行。
- [x] 编写失败测试，证明不完整的 adapter 或 memory 输出永远不会被 promote。
- [x] 实现原子状态转换，以及限定在单个 run ID 内的 lock file。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: orchestrate resumable opd iterations`。

### 任务 7.2：Curriculum 与 Sampling 控制

**文件：**
- 创建：`src/tau3_evolver/pipeline/sampling.py`
- 测试：`tests/unit/pipeline/test_sampling.py`

- [x] 根据已记录 seed 实现确定性的 train task 排序/打乱。
- [x] 防止 task ID 通过 override 或 resume manifest 跨入 test/base。
- [x] 平衡 `sel/act/write/maint` 样本，同时不得伪造缺失的 write/maintenance 监督。
- [x] 添加可配置的 iteration task count，生产默认值仍为完整官方 train set。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: add train-only opd iteration sampling`。

### 任务 7.3：多 Iteration Smoke

- [x] 运行两个完整 fake iteration，并验证 checkpoint N+1 只消费 checkpoint N 生成的数据。
- [ ] 在 train 任务上运行一个小型完整真实 iteration；真实数据 dry-run 与独立单 batch Qwen3.5-9B GPU update 已通过，端到端 iteration 仍在 Stage 8 验收实验中完成。
- [x] 验证从最终 manifest 到初始 checkpoint 的 memory 和 adapter lineage。

**阶段 7 gate：** 本地代码 gate 要求中断 run 可确定性恢复、promoted checkpoint 具有完整 lineage、所有学习 artifact 不存在 test ID；2026-07-24 已完成真实五任务 iteration 到 `dataset_complete`，并验证两次真实失败后的 Memory 回滚、同 ID 恢复、artifact 哈希、train-only guard 和独立 audit。该随机五任务数据集没有产生满足条件的 OPD 样本，因此没有发布空 checkpoint；独立 Qwen3.5-9B GPU update gate 已通过，非空真实 promotion 与多 iteration gate 保持 pending，并在 Stage 8 验收实验中完成。

---

## 阶段 8：留出 Retail 评测与报告

**产出：** 在官方 test 任务上测量指定 LoRA checkpoint，不进行参数更新，也不会意外复用 test artifact。

### 任务 8.1：评测隔离

**文件：**
- 创建：`src/tau3_evolver/eval/guard.py`
- 创建：`src/tau3_evolver/eval/runner.py`
- 测试：`tests/unit/eval/test_guard.py`
- 测试：`tests/unit/eval/test_evaluation_runner.py`

**协议：**
- `test_static`：以只读方式加载由 train 产生的冻结 memory snapshot；不允许 test memory 写入。
- `test_streaming`：从空的隔离 memory 开始，允许在 test stream 内进行快循环 memory 演化，以复现论文评测协议；绝不将该 memory 导出到训练。

- [x] 编写失败测试，确保两种协议都禁用 optimizer 创建、attribution、dataset 写入、checkpoint 保存和 train-memory 修改。
- [x] 断言 `test_static` 拒绝所有 memory operation。
- [x] 断言 `test_streaming` 只写入 `history/evaluations/<run_id>/<agent_id>/quarantine/` via `evaluation_quarantine_root(run_id, agent_id)`，并且训练 artifact loader 拒绝该路径。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: isolate static and streaming test evaluation`。

### 任务 8.2：官方指标与可复现 CLI

**文件：**
- 创建：`src/tau3_evolver/eval/metrics.py`
- 创建：`scripts/evaluate_retail.py`
- 测试：`tests/unit/eval/test_metrics.py`

- [x] 保留每个 episode 的 tau2 reward 和解析后的 `reward_info`，不得重新实现官方 evaluator。
- [x] 报告 task count、completed count、mean reward/success、逐任务结果、failure category、step count、parse-error rate 和配置的重复 trial。
- [x] 最终报告默认每个任务运行一个 seed 为 `42` 的 trial。
- [x] 记录 checkpoint、base model、tau2 commit、split hash、精确的 40 个 test ID、task order、用户模拟器、seed、protocol 和 memory snapshot。
- [x] 只添加 `--split test`。单独的 `--official-base-reproduction` flag 可以运行 `base`，但必须写入独立标记的报告。
- [x] 重新运行测试并确认 PASS。
- [x] 提交为 `feat: evaluate opd evolver on retail test split`。

### 任务 8.3：Baseline、Ablation 与最终报告

**文件：**
- 创建：`scripts/compare_evaluations.py`
- 创建：`src/tau3_evolver/eval/experiment.py`
- 创建：`src/tau3_evolver/eval/visualization.py`
- 创建：`scripts/build_stage8_report.py`
- 创建：`docs/evaluation_protocol.md`
- 修改：`README.md`
- 测试：`tests/unit/eval/test_experiment.py`
- 测试：`tests/unit/scripts/test_build_stage8_report.py`

- [x] 将主实验固定为 2×2：A 基础 Qwen 无 Memory、B 基础 Qwen 加冻结 S1、C 训练后 C1 加同一 S1、D 同一 C1 无 Memory；`test_streaming` 降为补充协议。
- [x] 支持基础模型在全部 74 个 train task 上连续运行三次，使用固定但不同的 task shuffle seed，并持续积累同一个 agent 的 Memory。
- [x] Stage 5 source loader、task grouping 和 dataset audit 支持跨 pass 重复 task ID，并以 `run_id:task_id` 保持 222 个 episode 的唯一身份。
- [x] 校验三个 train pass 的 74-task 完整性、Memory snapshot chain、OPD dataset source run lineage、C/D checkpoint 一致性和 B/C snapshot 一致性。
- [x] 汇总 `pass@1`、平均 Agent Token、Memory 总量与复用覆盖率、OPD 四类样本量、response token、optimizer step 和 forward-KL。
- [x] 计算主要对照差值、按 task 聚类的 paired bootstrap 95% CI，以及 Memory 与 OPD 的交互效应。
- [x] 生成无外部依赖的 HTML 仪表盘，展示四组结果、Token、Memory 增长/复用、OPD 样本构成、KL 曲线和 lineage。
- [x] 只在主流程稳定后添加 selection、writing 和 maintenance ablation。
- [x] 将 protocol、adapter、checkpoint 和 Memory snapshot 作为处理变量；只比较 tau2 commit、split hash、基础模型 revision、用户模拟器、NL evaluator、task order、seed set、trial 数和最大步数完全相同的 run。
- [x] 记录：在不使用 dev 的设计中，test 结果绝不用于超参数调整或 checkpoint 选择。
- [ ] 真实运行三个 74-task train pass、一次 OPD Slow Loop 和 A/B/C/D 各 `40 task × 1 trial`。
- [ ] 运行 `pytest -v`、所有已启用 integration test、dataset audit 和 manifest lineage audit。
- [x] 提交为 `docs: finalize tau3 retail opd evaluation protocol`。

**阶段 8 gate：** 三个真实 74-task train pass、一次真实 OPD 训练和 A/B/C/D 各 `40 task × 1 trial` 全部完成；统一报告通过 lineage/control 校验、生成 JSON 与 HTML 图表，并且不存在从 test artifact 返回训练流程的路径。

---

## 阶段级 Review Checkpoint

每个阶段结束后：

- 只 review 当前阶段的接口、测试和验收 gate。
- 运行 `pytest tests/unit -v` 以及当前阶段已启用的 integration marker。
- 运行 `git diff --check` 和 `git status --short`。
- 审计新增文件的生产引用和测试引用，完成核心代码、产品入口、运维工具、测试支持和历史摘要的归档分类。
- 开始下一阶段前完成 commit。
- 在当前阶段完成的同一个 commit 中更新本计划的 checkbox。

## 自查

- Spec 覆盖：环境、split 隔离、baseline、四层 memory、完整快循环、精确归因公式、四类 hindsight view、共享模型 OPD token KL、仅 LoRA 更新、iteration/resume 和留出集评测，都分别归入带 gate 的阶段。
- 占位符检查：计划不包含实现占位符。用户模拟器模型被有意设计为显式运行时配置；省略时会委托给固定 tau2 默认值，并记录解析后的实际配置。
- 类型一致性：`ProjectConfig` 和 `Tau2Config` 先于适配器定义；规范事件先于 memory attribution；`OPDExample` 先于 alignment/loss；adapter 和 memory revision 先于 iteration/evaluation manifest。
- 泄漏检查：只有 train 可以调用学习 API。Test streaming memory 被隔离，并由训练 loader 拒绝。Base 无法进入训练。
- 算法检查：公式 11-16 分别落实为候选集受控归因、特权同前缀教师分布、stop-gradient 全词表 KL，以及 response-token 平均。
