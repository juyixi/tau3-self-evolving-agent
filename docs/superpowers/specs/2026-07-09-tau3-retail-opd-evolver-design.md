# Tau3 Retail OPD-Evolver 训练项目设计

## 目标

构建一个面向 tau3-bench retail 任务的训练项目，使用 Qwen3.5-9B 同时作为特权教师（privileged teacher）和可部署学生（deployable student）。训练方法采用 OPD-Evolver 的同策略蒸馏（on-policy distillation），而非离线自蒸馏。学生模型使用 LoRA 适配器训练，不进行全参数微调。

项目需要针对 tau3 retail 明确实现 OPD-Evolver 的快循环（fast loop）和慢循环（slow loop）：

- 快循环：智能体与 retail 任务交互，检索并选择记忆，执行动作，写入新记忆，并周期性维护记忆库。
- 慢循环：在学生访问到的状态上训练同一个 Qwen3.5-9B 策略，对比学生分布与教师在相同学生前缀上的特权视角分布。

## 主要参考资料

实现必须同时参考以下两项 OPD-Evolver 资料：

- 论文："OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation"，arXiv 2606.17628v1。
  - 本地文件：`C:\Users\huang\Downloads\2606.17628v1.pdf`
  - arXiv：https://arxiv.org/abs/2606.17628v1
- 官方代码仓库：
  - GitHub：https://github.com/bingreeky/opd-evolver

retail 环境由当前官方 tau benchmark 仓库提供：

- Tau2-bench 仓库：https://github.com/sierra-research/tau2-bench
- Gym 适配器文档：
  https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/gym/README.md
- Retail 任务划分：
  https://github.com/sierra-research/tau2-bench/blob/main/data/tau2/domains/retail/split_tasks.json

论文是算法定义的权威来源；官方仓库是可执行 OPD-Evolver 工程模式的参考，尤其用于参考已发布的 executor OPD 示例和项目组织方式。本项目中所有 tau3 retail 快循环和慢循环的算法或结构决策，都必须能够追溯到这两项资料。

## 算法理解

本项目将 OPD-Evolver 实现为共享策略模型上的同策略蒸馏。不得将其描述或实现为离线自蒸馏：

- 学生：Qwen3.5-9B 加当前 LoRA 适配器，仅接收部署时可用的输入。
- 教师：与学生相同的 Qwen3.5-9B 实例和当前 LoRA 适配器，额外接收特权后见信息（privileged hindsight），在 `no_grad` 下对学生采样得到的相同前缀进行评估。
- 训练信号：学生访问前缀上的 token 级蒸馏损失。
- 范围：论文定义的四类生命周期决策：
  - `sel`：经验选择（experience selection）。
  - `act`：基于经验的执行（experience-grounded execution）。
  - `write`：经验写入（experience writing）。
  - `maint`：记忆库维护（repository maintenance）。

项目不得把 OPD 当成静态轨迹自蒸馏。rollout 必须来自当前学生策略，蒸馏目标必须基于这些同策略状态计算。

## Tau3 Retail 环境边界

当前官方 retail 环境来自 `sierra-research/tau2-bench`。项目面向任务的名称仍使用“tau3 retail”，而集成代码导入官方 `tau2` Python 包。Tau2-bench 是固定版本的外部依赖，不复制进本项目源码。建议将本地 checkout 放在 `external/tau2-bench/`，由 git 忽略，并以 editable 模式安装其 `gym` extra。

每次真实运行都必须在 run manifest 中记录 tau2-bench Git commit、任务 split 名称、精确任务 ID、split 文件哈希、用户模拟器配置、seed 和环境选项。单元测试使用 mock 或 fake Gym 对象，不依赖外部仓库。

项目适配器封装 `tau2.gym.gym_agent.AgentGymEnv`，并暴露：

- `reset(task_id | task_spec) -> observation`
- `step(action) -> observation, reward, done, info`
- `task_group(task_spec) -> str`
- `metadata(task_spec) -> dict`
- `success(info, reward, done) -> bool`

项目其余部分只依赖该适配器接口。适配器负责规范化 Gymnasium 返回的 `(observation, reward, terminated, truncated, info)`，保留 `info["reward_info"]` 中的官方评测结果，并在每个 episode 后关闭环境。单元测试和 dry run 使用 mock retail 适配器；真实 rollout 和评测命令必须使用 tau2-backed 适配器，且不需要修改 OPD 代码。

## 任务数据与划分策略

默认不创建内部 dev 划分。官方 retail split 的职责严格单向：

- `train`：快循环 rollout 采集、记忆写入与维护、结果校准的归因、教师特权后见信息以及 LoRA 更新的唯一数据来源。
- `test`：只用于最终评测或显式请求的 checkpoint 评测。test 任务不得生成训练样本、归因更新或模型更新，也不得修改由 train 学得的记忆。
- `base`：仅用于可选的官方全任务聚合复现。它同时包含 train 和 test 任务 ID，因此绝不能作为训练数据来源。

任务加载器必须在代码中强制执行该策略，不能只依赖 CLI 使用约定。训练入口拒绝 `test` 和 `base`。评测提供两个独立标记的协议：

- `test_static`：打开由 `train` 学得的只读记忆快照；test episode 不能修改该快照，也不能在后续 test episode 中共享新写入的记忆。
- `test_streaming`：从空的评测专用记忆库开始，允许按照论文协议在 test 任务流中运行快循环记忆演化。该记忆库隔离在评测目录下，所有训练 artifact loader 都必须拒绝加载它。

两种协议都不允许参数更新、归因数据集生成或基于 test 结果选择 checkpoint。

当前官方 split 文件包含 74 个 train 任务、40 个 test 任务和 114 个 base 任务。这些数量是针对固定 tau2-bench revision 的兼容性断言，不是硬编码的任务数据；依赖 revision 发生变化时，必须显式复核 split 并更新 manifest。

## 快循环设计

快循环遵循论文 Algorithm 1。

对于每个 retail 任务：

1. 根据任务、环境元数据、当前 observation 和可选的 retail 状态提示构造查询。
2. 从四层记忆库检索候选记忆：
   - `trajectory`：完整或压缩后的 retail 交互轨迹。
   - `tip`：局部警告、约束或启发式规则。
   - `skill`：可复用的 retail 任务流程。
   - `tool`：可执行或结构化的动作模板。
3. 使用当前学生策略选择一个紧凑的记忆子集。
4. 将选中的记忆格式化到执行 prompt 中。
5. 使用当前学生策略在 retail 环境中执行 rollout。
6. 记录 observation、action、reward、选中记忆、候选记忆、任务组、终止结果和解析错误。
7. 让当前策略根据任务、轨迹、结果和选中记忆生成记忆更新。
8. 每经过 `maintenance_period` 个任务，使用 `lookup`、`merge` 和 `delete` 操作执行记忆库维护。

论文默认值：

- 记忆层级：`trajectory`、`tip`、`skill`、`tool`。
- 教师侧检索候选数：50。
- 特权教师上下文中注入的记忆上限：20。
- 维护周期：`Q = 30`。
- 最大 episode 长度：40，除非 tau3 retail 需要不同上限。

## Memory 存储设计

Memory 持久化遵循 OPD-Evolver 官方仓库的文件式实现思路，不使用 SQLite 或其他数据库。四个层级分别使用普通 JSON 文件保存当前权威状态：

- `trajectory_memory.json`
- `tip_memory.json`
- `skill_memory.json`
- `tool_memory.json`

训练 Memory 与运行代码分离，统一存放在项目根目录下且由 Git 忽略的 `history/`。Memory 使用 Agent namespace 隔离，而不是使用训练轮次或 `run_id` 隔离。默认 `agent_id` 为 `retail`，其权威状态目录固定为：

```text
history/agents/retail/memory/
```

同一个 `agent_id` 的所有 fast loop、slow loop 和训练轮次持续读取并更新同一份权威 Memory，使经验能够跨轮次累积。`run_id` 仅标识日志、OPD 数据、checkpoint 和 manifest，不得参与训练 Memory 路径。未来扩展 airline 时使用 `history/agents/airline/memory/`，不同 Agent namespace 之间禁止隐式读取、合并或迁移 Memory。`agent_id` 和评测 `run_id` 只允许小写 ASCII 字母、数字、连字符和下划线，必须拒绝大写或其他非规范形式、空值、`.`、`..`、路径分隔符和其他路径穿越形式。

Memory 路径必须相对项目根目录解析，不能依赖进程启动时的当前工作目录。程序首次打开某个 Agent namespace 时自动创建目录；仓库不提交 `.gitkeep` 或任何运行时 Memory。已有其他路径下的 Memory 不自动迁移或合并，迁移必须由显式工具完成并记录来源 snapshot。

路径与生命周期测试必须证明：不同 `run_id` 但相同 `agent_id` 解析到同一训练 Memory；`retail` 与 `airline` 解析到不同目录；从项目外的当前工作目录启动仍得到项目根目录下的相同路径；非法 `agent_id` 在创建目录前失败；`history/` 下的文件不会进入 Git 跟踪。测试不得在真实 `history/` 写入数据，而应使用隔离的临时项目根目录。

每个层级文件包含 `schema_version`、`tier`、`count` 和 `items`。每条 MemoryItem 至少包含稳定 ID、content、embedding、metadata、source task IDs、created/updated round、version、active/retired status，以及 usage/success 统计。可选的 provider 级 embedding 结果另存为 `embedding_cache.json`，并按输入 hash 和 embedding model revision 隔离。

Repository 在进程内以字典和 embedding 数组维护活动状态。启动时完整加载四个 JSON 文件；新增、更新、merge 或 retire 时，先在内存副本中完成类型、层级、provenance、重复项和版本校验，再将完整层级状态写入同目录临时文件，执行 flush/fsync 后通过 `os.replace` 原子替换目标文件。单层写入失败时旧文件必须保持完整可读。

四层文件之间不宣称数据库式跨文件 ACID。一次 write decision 涉及多个层级时，各层使用稳定 Memory ID 幂等应用；只有全部目标层级完成后才追加成功事件。进程中断后根据 rollout JSONL 中的 write proposal/commit 状态检查并重试未完成层级，禁止生成重复 Memory。

模型不得直接修改 JSON 文件。Writer 和 Maintainer 只能输出经过 Pydantic 校验的 `create`、`update`、`merge`、`retire` 和 `lookup` 命令；Repository 负责应用命令和持久化。Merge 只能发生在同一层级，retire 是软删除，历史决策保留在 JSONL 日志中。

普通 JSON 仅保存当前 Memory 状态；以下追加式数据使用 JSONL：rollout 生命周期事件、候选/选择/使用证据、Memory 写入与维护操作、归因结果以及四类 OPD 训练样本。Checkpoint 或评测所使用的 Memory 通过四层规范化 JSON 副本和 snapshot manifest 固化，snapshot 内容 hash 作为 `memory_snapshot_id`。

每轮训练 manifest 必须记录 `agent_id`、输入 `memory_snapshot_id` 和输出 `memory_snapshot_id`。新 `run_id` 不得重置 Memory。`test_static` 只能通过只读 Repository 加载指定训练 snapshot；`test_streaming` 只能写入 `history/evaluations/<run_id>/<agent_id>/quarantine/`。训练 Memory loader 必须拒绝任何 `history/evaluations/` 路径，防止 test 经验回流训练。

## 结果校准的记忆归因

项目只在某条记忆实际被检索到的任务上估计其价值。对于记忆 `m` 和任务组 `g`，比较“被检索且被选择”的使用情况与“被检索但未被选择”的使用情况。这对应论文的候选集受控归因思想：

- `Omega_plus_g(m)`：任务组 `g` 中，`m` 被检索且被选择的任务。
- `Omega_g(m)`：任务组 `g` 中，`m` 被检索但未被选择的任务。
- 归因值比较这两组任务的平均 return。
- 置信度因子降低选择证据不足的记忆权重。
- 层级先验允许对 trajectory、tip、skill 和 tool 记忆采用不同权重。

最终得分 `V(m)` 用作选择、执行、写入和维护决策的特权后见信息。默认过滤记忆分数低于 `0.01` 的监督样本，与论文设置一致。

## 慢循环设计

慢循环为四类生命周期决策构造训练样本：

- 选择：
  - 学生输入 `z_sel`：retail 任务和检索到的候选记忆。
  - 教师后见信息 `h_sel`：每条候选记忆及其校准价值 `V(m)`。
- 执行：
  - 学生输入 `z_act`：retail 任务和公开环境历史，不包含特权记忆价值。
  - 教师后见信息 `h_act`：有价值的已选记忆，以及同一任务组中可用的成功轨迹。
- 写入：
  - 学生输入 `z_write`：任务、轨迹、return 和选中记忆。
  - 教师后见信息 `h_write`：生成的记忆候选及其未来校准价值。
- 维护：
  - 学生输入 `z_maint`：记忆库快照、日志历史和可用维护工具。
  - 教师后见信息 `h_maint`：记忆价值、置信度、使用统计和冗余诊断。

对每个样本，学生首先在部署条件下采样输出；随后教师在加入特权后见信息的条件下，对相同的学生生成前缀进行评估。训练损失是从 stop-gradient 教师分布到学生分布的 token 级 KL。

实现必须对齐较短的公开学生序列和较长的特权教师序列中的 response token 位置，并且只在学生采样得到的 response token 上计算 KL。将教师特权上下文拼进单个 causal language model 训练字符串并应用普通 next-token SFT loss，不属于 OPD 实现。

可部署 artifact 仅包含面向学生的 LoRA 适配器。推理时永远不需要特权后见上下文。

## 模型与训练默认值

基础模型：

- `Qwen/Qwen3.5-9B`，或本地可用的等价 Qwen3.5-9B checkpoint。

精度与上下文：

- `bf16`
- `max_prompt_length = 8192`

rollout 与蒸馏数据生成：

- `temperature = 1.0`
- `top_p = 0.95`
- `max_episode_steps = 40`

LoRA：

- `use_peft = true`
- `lora_r = 32`
- `lora_alpha = 64`
- `lora_dropout = 0.05`
- 本项目不支持全参数微调。

训练默认值：

- `learning_rate = 1e-5`
- `per_device_train_batch_size = 2`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 3`

这些默认值在适用位置与 OPD-Evolver 论文保持一致，也可以通过配置文件或 CLI flag 覆盖。

## 项目架构

仓库按照职责清晰的小模块组织：

- `configs/`
  - 模型、LoRA、rollout、memory、tau3 retail 和 OPD 训练配置。
- `src/tau3_retail_evolver/envs/`
  - Tau3 retail 适配器接口、mock 适配器和真实 tau2-backed 适配器。
- `src/tau3_retail_evolver/memory/`
  - 四层记忆存储、检索、格式化、评分和维护操作。
- `src/tau3_retail_evolver/fast_loop/`
  - Retail rollout、选择、执行、写入、维护编排和日志。
- `src/tau3_retail_evolver/slow_loop/`
  - 归因、后见信息构造、OPD 样本构建和 token 级蒸馏训练。
- `src/tau3_retail_evolver/models/`
  - Qwen 模型加载、LoRA 加载与保存、tokenizer 处理，以及教师和学生调用封装。
- `scripts/`
  - rollout、归因、OPD 训练、评测和端到端 iteration 的单命令入口。
- `tests/`
  - memory 评分、适配器行为、循环日志和样本构造的单元测试。

## 数据布局

运行时日志、训练样本和模型 artifact 存放在 `runs/` 下；持续进化的 Memory 单独存放在 `history/` 下。两个目录均由 Git 忽略：

- `runs/<run_id>/rollouts/*.jsonl`
- `runs/<run_id>/attribution/*.jsonl`
- `runs/<run_id>/opd_examples/*.jsonl`
- `runs/<run_id>/checkpoints/`
- `runs/<run_id>/eval/`
- `history/agents/<agent_id>/memory/{trajectory,tip,skill,tool}_memory.json`
- `history/agents/<agent_id>/memory/embedding_cache.json`
- `history/agents/<agent_id>/memory/snapshots/<snapshot_id>/*.json`
- `history/evaluations/<run_id>/<agent_id>/quarantine/`

每个 run 根目录还包含 `manifest.json`，其中记录模型 revision、LoRA revision、tau2-bench commit、split hash、任务 ID、seed、用户模拟器配置、`agent_id`、输入/输出 memory snapshot ID 和 parent checkpoint。Manifest 可以记录规范化的项目相对 Memory 路径，但训练数据加载必须以 snapshot ID 和 Agent namespace 为权威边界。

每条日志事件必须保留足够信息，以便重建：

- 检索候选 `C_t`
- 选中记忆 `S_t`
- episode 轨迹 `tau_t`
- return `R_t`
- 写入的记忆更新 `Delta_t`
- 维护轨迹 `eta_q`
- 任务组 `g(t)`

## 测试策略

首个实现版本必须能够在不下载 Qwen3.5-9B、不运行 tau3-bench 的情况下测试：

- Mock retail 任务验证适配器和快循环控制流。
- 确定性的 fake policy 验证选择、写入和维护日志。
- 合成 rollout 验证结果校准的归因。
- 小型 toy logits 验证 token 级 KL loss 链路。
- CLI smoke test 验证配置加载和 run 目录创建。

高 GPU 开销的模型测试作为独立 integration test，默认跳过；仅在所需环境变量和模型 checkpoint 可用时运行。

## 代码生命周期与归档

所有实施阶段必须遵守 [`docs/development/code-lifecycle.md`](../../development/code-lifecycle.md)。`src/` 只保留运行时核心代码，`scripts/` 只保留产品级工作流入口，部署和诊断工具归入 `tools/`，测试辅助实现归入 `tests/`。每个阶段结束前必须完成生命周期审计；一次性验证、review 和调试资产在目的完成后删除，由 Git 历史或压缩后的 `docs/archive/` 里程碑摘要承担归档。

## 待接入项

tau2-bench checkout 和 Qwen3.5-9B 权重是外部运行时依赖，不提交到本仓库。用户模拟器模型保持为显式配置，因为它会影响凭证、成本和可复现性；省略时使用固定 tau2-bench revision 的默认值，但解析后的实际配置仍必须记录到 run manifest。

官方 OPD-Evolver 仓库作为实现参考。如果实施阶段发现其中存在与本项目 license 和依赖约束兼容的可复用代码，实施计划必须选择以下方式之一：附带来源说明后 vendoring，或通过文档清晰的适配器进行封装。

## 验收标准

项目在满足以下条件时视为成功：

- 提供可复现的 Python 项目，包括 tau3 retail OPD-Evolver 训练所需的配置和脚本。
- 提供固定版本 `sierra-research/tau2-bench` retail Gym 环境的真实适配器，以及用于测试的离线 fake。
- 强制执行 split 隔离：`train` 用于学习，`test` 用于评测，`base` 仅用于可选复现，并且默认不创建 dev。
- 快循环能够记录 retail 同策略 rollout 和四层记忆生命周期事件。
- 慢循环能够从日志轨迹构建选择、执行、写入和维护四类 OPD 样本。
- 提供 LoRA 训练入口，默认值为 `use_peft=true`、`lora_r=32`、`lora_alpha=64`。
- 共享策略 OPD trainer 在学生采样前缀上计算 stop-gradient 教师 logits 和 response-token KL，而不是 SFT loss。
- 提供冻结 memory 的 `test_static` 评测器和与论文兼容、隔离运行的 `test_streaming` 评测器；两者均冻结 checkpoint，并保留官方 tau2 reward 详情和可复现性元数据。
- 文档清晰说明 tau3 retail 快循环和慢循环如何对应 OPD-Evolver 论文及官方仓库。
- 单元测试无需下载大模型即可通过。
