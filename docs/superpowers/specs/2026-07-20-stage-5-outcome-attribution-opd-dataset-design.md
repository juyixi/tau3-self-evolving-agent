# Stage 5 Outcome-Calibrated Attribution 与 OPD 数据集设计

## 状态

本文档记录已经确认的 Stage 5 设计。Stage 5 负责把同一轮当前策略产生的 Fast Loop 事件转换为可审计的 episode 证据，按照 OPD-Evolver 论文计算 Outcome-calibrated attribution，并构建 `sel`、`act`、`write`、`maint` 四类 Slow Loop 输入数据。

Stage 5 不执行 LoRA 优化，也不提前固化 Slow Loop 的学生 completion。学生 completion、同前缀教师前向和 token KL 属于 Stage 6。

## 参考资料与优先级

实现必须同时参考：

- 论文：*OPD-Evolver: Cultivating a Holistic Agent Evolver via On-Policy Distillation*，arXiv 2606.17628v1。
  - 本地附件：`C:\Users\huang\Downloads\2606.17628v1.pdf`
  - 在线地址：https://arxiv.org/abs/2606.17628v1
- 官方代码仓库：https://github.com/bingreeky/opd-evolver

算法冲突时以论文公式为准。官方仓库用于参考文件式 Memory、usage log、scoring、JSONL dataset、collator 和在线 generation 的工程组织。官方 README 明确说明公开训练路径是 executor OPD 演示，完整 selector、reflection 和 slow-fast 调度没有全部开源，因此本项目不能声称逐行复刻未公开实现。

官方 `opd_evolver/memory/scoring.py` 当前使用“选中均值减候选池整体均值”并对 group contribution 求平均；论文 Eq.11 使用“选中均值减未选中均值”并对 group contribution 求和。本项目严格实现论文 Eq.11。

## 目标

Stage 5 必须做到：

1. 只聚合同一当前策略产生的连续 train runs，保持 on-policy 数据边界。
2. 从不可变 run manifest、事件和冻结 Memory snapshot 重建规范化证据。
3. 通过匿名 retail task-group signature 控制任务类型混杂。
4. 精确实现论文 Eq.11-Eq.12，并保留逐组可审计中间量。
5. 构造论文 Eq.13 的四类 public/privileged hindsight view。
6. 对 split、lineage、snapshot、未来证据和特权泄漏执行 fail-closed 审计。
7. 产出确定性 JSONL 和带哈希的 dataset manifest。

## 非目标

Stage 5 不负责：

- 加载训练框架或执行 LoRA 更新。
- 计算 teacher/student logits 或 token KL。
- 从 `test`、`base` 或 evaluation quarantine 产生训练数据。
- 把 attribution、置信度或冗余诊断写回权威 Memory JSON。
- 使用 Tau2 golden action 参数、NL assertion 标准或 test 结果作为教师特权信息。
- 为证据不足的 Memory 伪造零价值标签。

## 术语和轮次边界

- `episode`：一个 Tau2 retail task 的完整交互。
- `run_id`：一次 Fast Loop 命令执行及其 run artifact 目录；一个 run 可以包含多个 episode。
- `iteration`：使用同一模型和 LoRA revision 采集一批 Fast Loop 数据、构建 OPD 数据并完成一次 Slow Loop LoRA 更新的外层自进化轮次。
- `dataset_build_id`：一次 Stage 5 派生数据构建的不可变标识。
- `maintenance_round`：每完成 `Q=30` 个 train tasks 触发的一次 Memory 维护，不等同于 iteration。

Memory 权威状态继续跨 iteration 累积在 `history/agents/<agent_id>/memory/`。OPD 训练数据不跨策略 revision 混合：一次 dataset build 只能接收相同 `iteration`、`model_revision` 和 `adapter_revision` 的 source runs。

## 总体架构

采用“规范化证据层”方案：

```text
one or more continuous train runs
                |
                v
        SourceRunLoader
                |
                v
      EpisodeEvidenceBuilder
                |
                v
        RetailTaskGrouper
                |
                v
        AttributionEngine
                |
                v
         OPDExampleBuilder
                |
                v
          DatasetAuditor
```

建议模块边界：

- `slow_loop/source_runs.py`：source run、manifest、summary、事件和 snapshot lineage 校验。
- `slow_loop/evidence.py`：事件状态机与 `EpisodeEvidence`/`MaintenanceEvidence`。
- `slow_loop/task_grouping.py`：Retail 特权任务元数据到匿名 signature。
- `slow_loop/attribution.py`：Eq.11-Eq.12 和审计明细。
- `slow_loop/examples.py`：四类 Eq.13 view。
- `slow_loop/leakage.py`：递归 public/privileged 边界检查。
- `slow_loop/dataset.py`：确定性写入、manifest 和原子发布。
- `scripts/build_opd_dataset.py`：构建 CLI。
- `scripts/audit_opd_dataset.py`：独立审计 CLI。

模块只通过显式 typed models 交互，dataset builder 不直接解析散落的事件字段。

## Artifact 布局

Source rollout 目录保持只读。Stage 5 创建独立 build：

```text
runs/<dataset_build_id>/slow_loop/
├── evidence/
│   └── episodes.jsonl
├── attribution/
│   └── memory_scores.jsonl
├── datasets/
│   ├── sel.jsonl
│   ├── act.jsonl
│   ├── write.jsonl
│   └── maint.jsonl
├── dataset_manifest.json
└── audit_report.json
```

四个 dataset 文件始终存在。某类样本没有足够证据时允许为空，但 manifest 必须记录 `count=0` 和分类跳过原因。构建在同级临时目录完成，所有写入和审计成功后通过原子 rename 发布。已存在的 `dataset_build_id` 不允许覆盖。

## Source Run 输入契约

所有 source runs 必须满足：

- `split == "train"`。
- 所有事件 `mode == "learn"`。
- `rollout_options.memory_enabled == true`。
- 相同 `iteration`、`model_revision`、`adapter_revision`。
- 相同 Tau2 commit、split hash、domain 和 Memory agent namespace。
- 每个 task ID 属于固定 Tau2 revision 的官方 train split。
- 每个 episode 只有一个开始、一个成功完成和完整生命周期事件。
- 不含 `EpisodeFailed`、`MemoryWriteFailed`、`MaintenanceFailed` 或 `MemoryDisabled`。
- source summaries 存在，且 task count 与事件一致。

多个 runs 按 `completed_train_tasks_before` 排序。相邻 runs 必须满足：

- 前一 run 的 `completed_train_tasks_after` 等于后一 run 的 `completed_train_tasks_before`。
- 前一 run 的 `output_memory_snapshot_id` 等于后一 run manifest 的输入 `memory_snapshot_id`。

不同 seed 可以聚合；每个 episode 的 seed 必须保存在 evidence provenance 中。重复 run ID、重复 episode identity、重复 task ID 或不连续 completed-task range 均 fail closed。

## Stage 4 事件契约升级

Stage 5 需要把 Fast Loop 事件 schema 从 `1` 升到 `2`：

1. `task_group` 不再硬编码为 `retail`，改为匿名 action signature。
2. run manifest 记录 Memory `agent_id`/namespace，使 snapshot 能被无歧义定位。
3. maintenance 的学生公开仓库视图只保留 ID、tier、content、version 和 status；移除 `usage_count`、`success_count`、`last_used`、attribution 和 redundancy。
4. usage、success 和 redundancy 在 Stage 5 从事件与 snapshot 派生，只进入教师特权字段。
5. 每个事件继续记录该决策发生前的 `memory_snapshot_id`。

Schema-1 runs 可以保留作人工检查，但不得进入正式 Stage 5 build。延期的真实五任务 Memory-enabled run 必须使用 schema 2。

## Retail Task Grouping

Task grouping 只在 train 数据构建时读取固定 Tau2 `tasks.json` 的特权 `evaluation_criteria.actions`。它不读取 action arguments、expected values、NL assertions 或 test tasks。

分组算法 `retail-actions-v1`：

1. 取 golden reference trajectory 中的 action name。
2. 删除只读动作。
3. 对剩余状态变更动作去重并按 ASCII 排序。
4. 将 `{domain, grouping_revision, action_names}` canonical JSON 化。
5. 输出 `retail-actions-v1:<sha256>`。

固定 revision 的只读动作集合：

- `calculate`
- `find_user_id_by_email`
- `find_user_id_by_name_zip`
- `get_item_details`
- `get_order_details`
- `get_product_details`
- `get_user_details`

固定 revision 的状态变更或终止动作集合：

- `cancel_pending_order`
- `exchange_delivered_order_items`
- `modify_pending_order_address`
- `modify_pending_order_items`
- `modify_pending_order_payment`
- `modify_user_address`
- `return_delivered_order_items`
- `transfer_to_human_agents`

出现未分类 action name 时 fail closed，要求显式更新 grouping revision。公开事件和 evidence 只保存 signature；原始 action-name set 只用于构建时校验，不写入学生输入。

## 规范化证据模型

`EpisodeEvidence` 每行对应一个成功完成的 train episode，至少包含：

- schema version、episode ID、run ID 和 source event range/hash。
- iteration、model/adapter revision、Tau2 revision、split hash、seed。
- task ID、task-group signature、Memory agent namespace。
- episode 开始前的 snapshot ID。
- 检索 query hash、retriever revision 和候选的 ID/tier/version/rank/similarity。
- 实际 selected Memory IDs。
- 公开 policy、tools、observation/action/reward trajectory。
- final reward、terminated/truncated 状态。
- write proposals、committed IDs 和 replayed IDs。

候选 Memory 内容不相信当前最新仓库，而从事件对应的冻结 snapshot 加载并验证 ID、tier 和 version。selected 必须是 candidates 的子集。写入证据只接受 `MemoryWriteCommitted`，replayed ID 不属于本轮新写入集合 `Delta_t`。

`MaintenanceEvidence` 对应一次 `MaintenanceCommitted`，包含 maintenance round、触发 task index、维护前 snapshot、公开 repository view、从前序 episode 重建的 `H_qQ`、proposed commands 和 committed result。

事件状态机必须拒绝缺失、乱序、重复、跨 run provenance 或 proposal/commit 不一致。

## Outcome-Calibrated Attribution

对每条 Memory `m`，只使用它确实进入候选集合的 episode。设 `g(t)` 为 task-group signature：

```text
Omega_plus_g(m)  = {t: g(t)=g and m in S_t}
Omega_minus_g(m) = {t: g(t)=g and m in C_t \ S_t}
```

一个 group 必须同时存在 selected 和 not-selected 样本才能形成对照；缺任一侧时省略该 group。

严格实现论文 Eq.11：

```text
rho_g(m)   = n_selected_g / (n_selected_g + n_not_selected_g)
delta_g(m) = mean(R_selected_g) - mean(R_not_selected_g)
A_hat(m)   = sum_g rho_g(m) * delta_g(m)
```

严格实现论文 Eq.12：

```text
N_plus(m) = sum_g n_selected_g
gamma(m)  = 1 - 1 / sqrt(1 + N_plus(m))
V(m)      = alpha_tier(m) * gamma(m) * A_hat(m)
```

默认 tier prior 参考官方 scoring 实现，并写入配置和 manifest：

| Tier | Prior |
|---|---:|
| `trajectory` | 0.9 |
| `tip` | 0.8 |
| `skill` | 1.0 |
| `tool` | 1.2 |

论文 Eq.12 没有 recency 项，因此不加入时间衰减。

如果所有 groups 都无法形成双侧对照：

- `status = "insufficient_evidence"`
- `value = null`
- 保留 retrieved/selected counts
- 不允许进入正式教师监督

数学正价值仍定义为 `V(m) > 0`。实验监督资格使用配置 `memory.score_threshold`，默认 `0.01`。Attribution 文件保留全部有限正负分数；样本 builder 使用阈值决定是否有足够教学信号。

新写入 Memory 的 creator episode 不参与它自身的价值估计。只有创建之后的 episode 才能提供 future evidence。当前 iteration 冻结后，不允许使用未来 iteration 的结果反向改写该 build。

`memory_scores.jsonl` 每条记录至少包含：

- Memory ID、tier、观测版本。
- 每组 selected/not-selected counts 和 reward means。
- `rho`、`delta`、group contribution。
- `N_plus`、`gamma`、tier prior、`A_hat`、`V`。
- source episode IDs。
- evidence status 和 `qualified_for_supervision`。
- build、iteration、model/adapter provenance。

记录按 Memory ID 确定性排序。

## Slow Loop 数据配置

Stage 5 新增独立的 `slow_loop` 配置块，不把派生数据参数硬编码进 attribution 实现，也不改变 Fast Loop Memory 权威状态：

```yaml
slow_loop:
  tier_priors:
    trajectory: 0.9
    tip: 0.8
    skill: 1.0
    tool: 1.2
  redundancy_threshold: 0.90
  max_redundancy_pairs: 50
```

`score_threshold` 和 `teacher_memory_cap` 继续只读取现有 `memory` 配置，避免双配置漂移。Dataset manifest 同时记录解析后的 `slow_loop` 配置以及实际使用的这两个 Memory 参数。

## OPD Example 与在线采样契约

论文 Eq.14 要求：学生先在 public input 上现场采样 `y_hat`，教师随后在完全相同的学生 token prefix 上计算分布。官方 OPSD trainer 也在训练时生成 student completion。

因此 Stage 5 不把 Fast Loop 的历史响应或重新序列化 JSON 当作精确 student prefix。`OPDExample` 定义为：

```text
OPDExample(
    example_id,
    kind,
    public_input,
    privileged_hindsight,
    response_schema,
    sampling_contract,
    provenance,
)
```

`sampling_contract` 固定声明：

- `mode = "online"`
- Stage 6 必须加载 manifest 指定的当前模型和 LoRA revision。
- student completion 由 Stage 6 现场生成。
- teacher 使用同一 completion token prefix。
- teacher forward stop-gradient。
- Fast Loop 历史输出只能作行为 provenance，不是 gold label，也不是固定前缀。

Stage 6 把实际 completion 和 generation provenance 写入独立 training-generation JSONL，不回写 Stage 5 数据集。

## Selection View

论文：

```text
z_sel = (x_t, C_t)
h_sel = {(m, V(m)) for m in C_t}
```

每个完整 episode 最多生成一条 `sel`：

- public：任务、公开环境状态、公开 tools、全部候选 Memory。
- privileged：候选 ID 对应的 `V`、`gamma` 和 evidence status。
- response schema：结构化 Memory ID selection。
- 至少一个候选满足 `abs(V) >= score_threshold` 才生成。
- score table 保留正值和负值；负值用于指导“不选择”。

Candidate ID 可以同时出现在 public 和 privileged 中作为 join key；privileged 不重复候选 Memory content。

## Action View

论文：

```text
z_act = x_t
h_act = (S_plus_t, tau_plus_t)
```

按每个 action turn 生成：

- public：任务、公开 policy/tools、截至当前 turn 的公开 observation/action history。
- public 明确移除全部 Memory ID、content 和 Memory placeholder。
- privileged：本轮 selected 且 `V >= score_threshold` 的 Memory，以及同 task group 的成功 trajectory。
- 合格 Memory 最多 `teacher_memory_cap=20` 条，按 `V` 降序、retrieval rank 升序和 Memory ID 排序。
- 没有合格 Memory 或同组成功 trajectory 时跳过。

Tau2 成功 trajectory 使用官方 episode reward 判定 `R_t == 1.0`。优先使用当前成功 episode；当前 episode 不成功时，从同组 episode 按 final reward 降序、step count 升序、episode ID 升序确定性选择。该 trajectory 必须来自本次 build 的 train evidence，不能来自 test 或 golden actions。

## Writing View

论文：

```text
z_write = (x_t, tau_t, R_t, S_t)
h_write = {(new_memory, V(new_memory)) for new_memory in Delta_t}
```

每个产生 committed new Memory 的 episode 最多生成一条 `write`：

- public：任务、公开 trajectory、final reward 和 selected Memory。
- privileged：本轮新写入 Memory 的 future `V` 和逐组证据。
- creator episode 不得为自己的写入提供价值证据。
- replayed/idempotent Memory 不属于新写入集合。
- 至少一条新 Memory 满足 `abs(V) >= score_threshold` 才生成。
- 正负价值写入同时保留，使教师既能指导“应写什么”，也能指导“不应写什么”。

## Maintenance View

论文：

```text
z_maint = (M_qQ, H_qQ, T)
h_maint = D_mem_q
```

每个 `MaintenanceCommitted` 最多生成一条 `maint`：

- public：维护前的仓库内容、此前 train interaction history、`lookup/merge/delete` 工具定义。
- public 不含 usage、success、last-used、value、confidence 或 redundancy。
- privileged：每条 Memory 的 `V`、`gamma`、retrieved/selected/success usage，以及 pairwise cosine redundancy `kappa`。
- Memory diagnostics 最多 20 条。
- 诊断采样覆盖高价值、低价值、高使用和高冗余端点；各桶去重后按 Memory ID 稳定补齐。
- Pairwise redundancy 仅在 embedding model revision 和维度一致时计算。
- 默认 `redundancy_threshold=0.90`，最多保留 50 个最高 redundancy pairs；两者写入 manifest。
- 失败、仅 proposed 或未到期 maintenance 不生成样本。

## Public/Privileged 隔离

允许 stable ID 在 public 与 privileged 中重复作为显式 join key，不允许特权诊断或隐藏内容跨边界复制。

递归 leakage guard 必须保证：

- public 禁止 attribution/value/score、`gamma`、usage、success count、last-used、redundancy 等特权字段及命名变体。
- `act.public_input` 禁止任何 Memory ID、content 或 Memory context。
- public 和 privileged 均禁止 Tau2 `evaluation_criteria`、golden action arguments、NL assertion rubric 和 test data path。
- 所有 artifacts 禁止 credential-bearing key/value 或带凭证 URL。
- privileged 数据只能来自同一 build 的 train evidence、冻结 snapshot 和已计算 attribution。
- evaluator terminal details 不进入 Eq.13 public input；只保留论文允许的 scalar reward 和公开 trajectory。

## CLI

构建命令：

```powershell
python -m scripts.build_opd_dataset `
  --config configs/default.yaml `
  --source-run runs/<run-1> `
  --source-run runs/<run-2> `
  --dataset-build-id opd-iter0-001 `
  --output-root runs `
  --project-root .
```

独立审计：

```powershell
python -m scripts.audit_opd_dataset `
  --dataset-dir runs/opd-iter0-001/slow_loop `
  --project-root .
```

CLI 不隐式扫描整个 `runs/`。source runs 必须显式列出，顺序由 lineage 校验后规范化。正式 dataset、source runs、Tau2 Retail 元数据和 Memory root 必须位于同一项目根目录内；manifest 只保存项目相对路径。审计命令的 `--project-root` 可省略，此时从 dataset 相对路径自动反推。构建器向 stdout 输出 canonical summary，不输出 prompt、Memory content 或凭证。

## Dataset Manifest

`dataset_manifest.json` 至少记录：

- dataset schema、build ID 和构建代码 revision。
- source run IDs 及 manifest/events/summary SHA256。
- source runs、Tau2 Retail tasks/split、Memory root 和 dataset 的项目相对路径。
- iteration、model/adapter revision。
- Tau2 commit、split hash、精确 train task IDs。
- Memory agent namespace 和完整 snapshot chain。
- grouping/attribution formula revision。
- tier priors、score threshold、teacher cap、redundancy 参数。
- evidence 和四类样本 counts、跳过原因。
- 每个输出 JSONL 的行数和 SHA256。

Manifest 使用 canonical JSON，创建后不可覆盖。

## 审计规则

独立 auditor 必须检查：

1. JSONL 每个非空行是合法 object，schema version 正确。
2. example ID、episode identity 和 maintenance identity 不重复。
3. 全部 source task 属于官方 train split。
4. run policy lineage、completed-task range 和 snapshot chain 连续。
5. candidate 存在于对应冻结 snapshot，tier/version 匹配。
6. selected 是 candidates 的子集。
7. committed write 与事件一致，future evidence chronology 合法。
8. maintenance 样本对应成功 commit。
9. public/privileged 隔离、credential policy 和 test leakage 检查通过。
10. 每个 example 声明 online sampling contract。
11. 所有 artifact 行数和 SHA256 与 manifest 一致。

## 失败策略

以下条件终止整个 build，且不发布半成品：

- source policy lineage 不一致。
- snapshot 缺失、manifest 无法定位 namespace 或哈希错误。
- 事件缺失、乱序、重复或含失败事件。
- source 来自 test/base/evaluation quarantine 或 Memory-disabled run。
- task-group signature 不匹配固定任务元数据。
- privileged 信息进入 public input。
- duplicate ID、manifest 冲突或输出目录已存在。

证据不足不是 build failure，必须计入结构化 skip reason：

- `insufficient_selected_control`
- `no_scored_candidate`
- `no_successful_same_group_trajectory`
- `no_qualified_selected_memory`
- `no_future_write_evidence`
- `no_committed_maintenance`

## 测试策略

单元测试覆盖：

- Retail action classification、canonicalization 和匿名 signature。
- source run lineage 和 snapshot chain。
- event-to-evidence 状态机。
- candidate/snapshot/version 校验。
- Eq.11-Eq.12 手算 fixture，包括多 group、负值和证据不足。
- `sel`、`act`、`write`、`maint` builder。
- creator episode future-evidence 排除。
- public/privileged 递归 leakage guard。
- malformed、duplicate、stale、cross-split 和 Memory-disabled 失败路径。
- manifest/hash 和确定性重建。

集成测试使用两个连续的合成 schema-2 train runs，产生四类非空数据，执行 build 和独立 audit，并断言同一输入的所有输出哈希一致。

## Stage 5 验收 Gate

Stage 5 完成必须满足：

- 论文 Eq.11-Eq.12 与手算 fixture 完全一致。
- 四类样本通过 provenance、chronology 和 leakage audit。
- Schema-2 Fast Loop 回归测试通过。
- 全部 Stage 4/Stage 5 自动化测试通过。
- 设计中没有从 test artifact 返回训练路径。

真实五任务 Memory-enabled run 已明确延期，不阻塞本阶段代码开始；Stage 5 完成后必须用 schema 2 补跑，并成功构建/审计 evidence，作为进入 Stage 6 的硬门槛。
