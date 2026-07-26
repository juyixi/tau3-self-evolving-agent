# Memory Tier 分类边界增强设计

## 状态

本文档定义 `trajectory`、`tip`、`skill`、`tool` 四类 Memory 的分类边界、结构化契约、写入校验、检索呈现、维护和兼容迁移方案。

当前状态：核心契约已实施，真实 Retail iteration 与分类质量评估待执行。

本次实施只约束新产生的 Memory，不迁移、重写或重新分类已有 Memory。V1 文件继续保持可读，新的 Fast Loop 写入统一使用 V2 typed payload。

## 背景与问题

当前项目已经具备四层 Memory 文件、Embedding 检索、Fast Loop 写入、周期维护、Outcome-calibrated attribution 和 OPSD `write` 数据构建能力：

```text
history/agents/<agent_id>/memory/
├── trajectory_memory.json
├── tip_memory.json
├── skill_memory.json
└── tool_memory.json
```

但当前 `MemoryWrite` 只要求模型输出：

```json
{
  "tier": "tip | skill | tool | trajectory",
  "content": "free-form text",
  "retrieval_text": "optional text",
  "metadata": {}
}
```

写入提示只列出合法 tier，没有给出可执行的分类标准。运行时也没有验证：

- `tip` 是否只表达一个原子规则；
- `skill` 是否真的包含多个有序步骤；
- `tool` 是否引用当前 Tau2 环境中真实存在的工具；
- `trajectory` 是否绑定一个可追溯的具体 episode。

因此，tier 在当前实现中主要是语义标签。真实 Stage 5 Task 0 产出的两条 `skill`：

```text
When processing exchanges, verify that new item IDs ...
Check product inventory details to confirm requested variants ...
```

本质上仍是单点约束，更接近 `tip`，并不是完整工作流。

## 目标

增强后的分类必须稳定表达：

```text
tip        = 一个点
skill      = 多个步骤组成的工作流
tool       = 某个 API/环境工具的调用知识
trajectory = 一个具体案例
```

具体目标：

1. tier 从自由文本标签升级为代码可验证的结构化契约。
2. 同一条 Memory 只承担一种语义职责；混合内容必须拆分。
3. `tool` 明确表示 tool-use knowledge，不保存或执行任意代码。
4. 保留 `content` 和 `retrieval_text`，兼容当前 Embedding、Prompt 和 JSON 存储。
5. 不破坏既有 Memory snapshot、event、evidence 和 checkpoint lineage。
6. 让四层分类真正影响写入、检索、维护、归因分析和消融实验。

## 非目标

本文档不负责：

- 在 Memory 中存储 Python 函数或动态执行代码；
- 修改 Tau2 官方工具实现；
- 改变 Outcome-calibrated attribution 公式；
- 改变 OPSD Forward KL 训练目标；
- 使用 test 数据或 evaluator 隐藏信息构造 Memory。

## 权威分类定义

### Tip：原子规则

`tip` 是一个可独立判断和应用的局部规则、警告、约束或启发式。

必须满足：

- 只表达一个主要判断点；
- 不依赖某个具体用户、订单或 episode 才成立；
- 不以多个有序步骤作为主体；
- 不以某个工具的参数合同作为主体。

推荐形式：

```text
当 <condition> 时，应当/不得 <guidance>，因为 <rationale>。
```

示例：

```text
处理订单操作前必须完成用户认证。
只有 delivered 状态的订单才能执行换货。
不要把不同产品类型的商品互相换货。
```

反例：

```text
先认证，再查订单，然后查库存，最后调用换货工具。
```

反例包含多个有序步骤，应归为 `skill`。

### Skill：可复用工作流

`skill` 是为完成一个目标而组织的多步骤策略，可以跨多个工具和对话动作。

必须满足：

- 有明确 `goal`；
- 至少包含两个有序步骤；
- 每个步骤描述目的或状态转换，而不是只罗列口号；
- 可以包含分支、失败恢复和结束条件；
- 不包含只能用于单个用户或订单的具体 ID。

示例：

```text
目标：处理已送达订单的换货。
1. 通过邮箱或姓名加邮编认证用户。
2. 查询订单并确认状态为 delivered。
3. 确认原商品和目标商品属于同一产品类型。
4. 检查目标规格和库存。
5. 调用换货工具并确认数据库状态更新。
```

`skill` 可以引用工具名，但重点必须是端到端工作流，而不是某个工具的参数说明。

### Tool：工具调用知识

`tool` 是对一个当前环境中真实存在的 API/工具的使用说明。它不是可执行函数，也不拥有工具实现。

必须满足：

- 只绑定一个 `tool_name`；
- `tool_name` 必须存在于当前 Tau2 reset 返回的工具 schema；
- 描述调用时机或前置条件；
- 描述关键参数的含义、约束或映射关系；
- 可以包含一个合法调用示例；
- 不允许嵌入代码、shell 命令或未知工具。

示例：

```text
tool_name: exchange_delivered_order_items
调用条件：用户已认证，订单状态为 delivered。
参数规则：
- item_ids 与 new_item_ids 必须按位置一一对应；
- 新旧商品必须属于同一产品类型；
- 价格差通过 payment_method_id 处理。
```

真正执行时，Agent 仍然根据 Tau2 官方工具 schema 生成 tool call：

```json
{
  "name": "exchange_delivered_order_items",
  "arguments": {
    "order_id": "#W2378156",
    "item_ids": ["1151293680"],
    "new_item_ids": ["7706410293"],
    "payment_method_id": "credit_card_9513926"
  }
}
```

### Trajectory：具体案例

`trajectory` 是一个真实 episode 的具体执行案例或其压缩版本。

必须满足：

- 绑定 `source_episode_id`、`task_group` 和 `final_reward`；
- 包含具体初始问题或状态摘要；
- 包含按顺序排列的实际 action 和 outcome；
- 明确成功、失败或部分成功；
- 保留案例特征，不伪装成普适规则；
- 不包含 evaluator 隐藏答案、test 数据或用户模拟器私有推理。

示例：

```text
案例：用户要求更换已送达订单中的键盘和恒温器。
结果：reward=1.0。
轨迹：认证用户 → 查询订单 → 查询两个目标规格 →
调用 exchange_delivered_order_items → done。
```

## 分类决策树

模型和运行时按照以下顺序判断：

```text
是否描述一个具体 episode，并带有来源和结果？
├── 是 → trajectory
└── 否
    是否以一个已注册工具的调用合同为主体？
    ├── 是 → tool
    └── 否
        是否包含为同一目标服务的两个及以上有序步骤？
        ├── 是 → skill
        └── 否 → tip
```

补充规则：

1. 同一段内容同时满足多个类型时必须拆分，不能选择一个“最接近”的 tier 后整体写入。
2. 提到工具名不自动成为 `tool`。例如“认证前不要调用换货工具”仍可属于 `tip`。
3. `skill` 可以调用多个工具；`tool` 只解释一个工具。
4. 从具体成功案例中提炼出的普适工作流写入 `skill`，原案例本身写入 `trajectory`。
5. 无法稳定分类时不写入，记录 `MemoryWriteRejected`，而不是猜测 tier。

## V2 结构化契约

### 公共字段

现有 `MemoryItem` 增加：

```json
{
  "tier_schema_version": 2,
  "tier": "tip",
  "payload": {},
  "content": "由 payload 确定性渲染的文本",
  "retrieval_text": "用于 Embedding 的短文本",
  "metadata": {
    "classification_rule": "tip-atomic-v1"
  }
}
```

约束：

- `payload` 是 tier 语义的权威来源；
- `content` 由 renderer 从 payload 确定性生成；
- 模型不得同时自由生成 `payload` 和不一致的 `content`；
- stable Memory ID 继续使用 `tier + canonical_content`，保证现有去重逻辑可延续；
- event 和 snapshot 必须记录 `tier_schema_version`。

### TipPayload

```json
{
  "condition": "订单相关操作即将执行",
  "guidance": "先完成用户认证",
  "rationale": "订单操作需要绑定已认证用户",
  "scope": ["retail", "order"]
}
```

校验：

- `guidance` 必填；
- 禁止 `steps`、`tool_name`、`source_episode_id`；
- `condition` 和 `guidance` 不得包含具体订单号、邮箱或用户 ID；
- renderer 输出一个条件加动作的原子句。

### SkillPayload

```json
{
  "goal": "完成已送达订单换货",
  "preconditions": ["用户提出换货需求"],
  "steps": [
    {"order": 1, "instruction": "认证用户"},
    {"order": 2, "instruction": "查询并验证订单状态"},
    {"order": 3, "instruction": "核对目标规格和库存"},
    {"order": 4, "instruction": "执行换货并确认结果"}
  ],
  "success_condition": "换货操作成功且数据库状态正确",
  "recovery": ["目标规格无库存时请求用户选择其他规格"]
}
```

校验：

- `goal`、`steps`、`success_condition` 必填；
- `steps` 至少两步，`order` 从1连续递增；
- 禁止具体订单号和用户身份信息；
- 只包含一步时拒绝并建议改写为 `tip` 或 `tool`。

### ToolPayload

```json
{
  "tool_name": "exchange_delivered_order_items",
  "purpose": "交换已送达订单中的商品",
  "preconditions": [
    "用户已认证",
    "订单状态为 delivered"
  ],
  "argument_rules": {
    "order_id": "必须是当前已验证订单",
    "item_ids": "原商品 ID 列表",
    "new_item_ids": "与 item_ids 按位置一一对应",
    "payment_method_id": "用于处理价格差"
  },
  "expected_effect": "更新订单商品并处理价格差",
  "example": {
    "name": "exchange_delivered_order_items",
    "arguments": {
      "order_id": "<verified_order_id>",
      "item_ids": ["<old_item_id>"],
      "new_item_ids": ["<new_item_id>"],
      "payment_method_id": "<verified_payment_method_id>"
    }
  }
}
```

校验：

- `tool_name`、`purpose`、`preconditions`、`argument_rules` 必填；
- `tool_name` 必须存在于本 episode 的官方工具 schema；
- `argument_rules` 只能引用该工具声明的参数；
- `example.name` 必须等于 `tool_name`；
- 示例参数必须通过对应 JSON Schema；
- 示例只能使用占位符或训练 episode 中已公开的信息；
- 禁止保存函数代码或可执行命令。

### TrajectoryPayload

```json
{
  "source_episode_id": "run-id:task-id",
  "task_group": "retail-actions-v1:<sha256>",
  "initial_state": "用户请求更换已送达订单中的两个商品",
  "steps": [
    {
      "order": 1,
      "action": "find_user_id_by_name_zip",
      "outcome": "认证成功"
    },
    {
      "order": 2,
      "action": "get_order_details",
      "outcome": "确认订单已送达"
    }
  ],
  "final_reward": 1.0,
  "result": "success",
  "lesson": "认证和状态检查完成后再执行换货"
}
```

校验：

- episode provenance、步骤和结果必填；
- `final_reward` 必须与 `EpisodeFinished.final_reward` 一致；
- action 必须来自该 episode 的真实 trajectory；
- `result` 由 reward 规则确定，不能由模型自由声称；
- 禁止复制 `simulation_result`、golden action、NL assertion 答案或 evaluator 私有字段。

## 写入流程

增强后的 Fast Loop 写入流程：

```text
EpisodeFinished
      |
      v
Write Prompt 注入四层定义和 discriminated JSON schema
      |
      v
模型输出一个或多个 typed payload
      |
      v
TierBoundaryValidator
├── schema 校验
├── 分类决策树校验
├── Tau2 tool registry 校验
├── provenance 校验
├── 隐私/泄漏校验
└── 内容原子性与步骤数校验
      |
      v
确定性渲染 content/retrieval_text
      |
      v
MemoryWriteProposed
      |
      v
MemoryWriteCommitted 或 MemoryWriteRejected
```

写入模型可以从一个 episode 同时提炼多种 Memory。例如：

```text
1条 trajectory + 2条 tip + 1条 skill + 1条 tool
```

这些记录共享 source episode provenance，但各自只有一个明确职责。

## Prompt 增强

`_WRITE_SYSTEM` 必须加入简短、互斥的定义：

```text
TIP: one atomic reusable condition, warning, or rule.
SKILL: one reusable goal with at least two ordered steps.
TOOL: usage knowledge for exactly one tool present in the supplied tool schemas.
TRAJECTORY: one concrete episode case with source provenance and observed outcome.

Split mixed lessons into separate memories. If none passes a definition, return [].
```

用户 payload 继续提供 public policy、tools、trajectory 和 terminal reward。模型输出必须使用按 `tier` 判别的 `oneOf` JSON Schema，不能用一个通用自由文本对象覆盖四类。

Prompt 示例只用于帮助格式稳定，运行时 validator 才是最终权威。

## 检索与 Prompt 呈现

### 第一阶段：保持统一向量检索

为了降低改动范围，V2 首先保留当前全局 cosine Top-K，但改变展示格式：

```text
[TIP]
Condition: ...
Guidance: ...

[SKILL]
Goal: ...
Steps:
1. ...
2. ...

[TOOL: exchange_delivered_order_items]
Preconditions: ...
Arguments: ...

[TRAJECTORY: reward=1.0]
Initial state: ...
Observed steps: ...
```

这样即使候选来自同一个 Top-K，模型也能稳定区分其用途。

### 第二阶段：可选的 tier-aware retrieval

在验证分类质量后，再增加：

- 每个 tier 独立检索 Top-K；
- 全局 rerank；
- 每层最大配额；
- 不强制填满无关 tier；
- 记录 per-tier retrieved/selected/used 指标。

不在分类边界尚不稳定时直接增加 tier quota，否则只会放大错误分类。

## Attribution 与 OPSD 数据

Outcome-calibrated attribution 继续按 `memory_id` 计算，tier prior 暂不调整：

```yaml
trajectory: 0.9
tip: 0.8
skill: 1.0
tool: 1.2
```

V2 `write.jsonl` 的 `response_schema` 改为 discriminated union，并将教师特权信息与以下信息关联：

- 提议 tier；
- validator 结果；
- future attribution；
- creator episode；
- 后续 evidence episodes。

学生 public input 不包含 attribution、未来 reward 或 evaluator 信息。教师可以利用分类诊断和 future value 指导同前缀分布，但最终写入仍须通过运行时 validator。

新增按 tier 的数据指标：

```text
proposed_count
accepted_count
rejected_count
retiered_count
retrieved_count
selected_count
qualified_for_supervision_count
mean_attribution
```

## Maintenance 规则

1. 只允许同 tier 合并。
2. `tip` 合并后仍必须保持单点；两个不同规则不能拼成一条长文本。
3. `skill` 合并必须重新生成连续步骤并通过工作流校验。
4. `tool` 只能与相同 `tool_name` 的记录合并。
5. `trajectory` 默认不合并；可以压缩，但必须保留所有 source episode IDs。
6. 发现错误 tier 时执行 `retier`：
   - 创建符合目标 tier schema 的新 Memory；
   - 原 Memory 标记 retired；
   - 新记录保存 `retiered_from`；
   - 禁止原地修改 tier，因为 stable ID 包含 tier。

## 旧数据兼容与迁移

### 读取兼容

- `tier_schema_version` 缺失时视为 V1；
- V1 继续可检索和使用；
- V1 不允许与 V2 直接合并；
- Prompt 中为 V1 标记 `[LEGACY_UNVERIFIED_TIER]`。

### 迁移流程

提供独立、可 dry-run 的迁移工具：

```text
tools/memory/migrate_tiers.py
```

流程：

```text
读取一个冻结 Memory snapshot
→ 规则分类
→ 必要时调用当前 Qwen 生成结构化 payload
→ 本地 validator 校验
→ 输出 migration_report.jsonl
→ 创建新的不可变 snapshot
→ 人工确认后更新权威 Memory
```

迁移不能覆盖旧 snapshot。每条结果记录：

```json
{
  "source_memory_id": "...",
  "source_tier": "skill",
  "target_tier": "tip",
  "status": "migrated | unchanged | rejected | needs_review",
  "reason_codes": ["single_atomic_rule"],
  "target_memory_id": "..."
}
```

对当前 Stage 5 Task 0 的建议迁移：

| 原内容 | 原 tier | 建议 tier |
| --- | --- | --- |
| 订单操作前认证用户 | `tip` | `tip` |
| 操作前验证订单状态 | `tip` | `tip` |
| 新旧商品必须是同一产品类型 | `skill` | `tip` |
| 换货前确认规格和库存 | `skill` | `tip` |
| `exchange_delivered_order_items` 参数说明 | `tool` | `tool` |

迁移后应另外从成功 trajectory 提炼一条真正的换货 `skill` 和一条具体 `trajectory`，不能把两条 tip 简单拼接成 skill。

## 实施阶段

当前实施范围：

| 阶段 | 状态 | 已完成内容 |
| --- | --- | --- |
| Phase A | 已完成 | 四类 payload、`WriteDecision`、写入 Prompt、确定性 renderer 和核心正反例 |
| Phase B | 核心已完成 | Tau2 tool schema 校验、trajectory 真实来源注入、V2 event/snapshot 写入和 fail-closed 校验 |
| Phase C | 核心已完成 | Evidence typed payload、OPSD write schema、payload/content/tool/provenance 审计 |
| Phase D | 按本次决策跳过迁移 | 保留 V1 读取兼容，不处理已有 Memory |
| Phase E | 部分完成 | V2 payload 可进入检索 Prompt；tier-aware retrieval 与消融实验待执行 |

当前未实现 `MemoryWriteRejected` 独立事件。模型输出在一次修复后仍不满足契约时，episode 按现有 fail-closed 路径失败并且不写入 Memory。V2 Memory 暂不允许使用旧自由文本维护命令进行 merge，后续需要单独设计 typed maintenance payload。

### Phase A：分类契约与 Prompt

- 新增四类 payload Pydantic 模型；
- 将 `WriteDecision` 改为 discriminated union；
- 增强 `_WRITE_SYSTEM`；
- 实现确定性 content renderer；
- 增加分类正反例 fixture。

### Phase B：运行时校验与事件

- 实现 `TierBoundaryValidator`；
- 从 Tau2 reset tools 建立只读 tool registry；
- 增加 `MemoryWriteRejected` 事件和 reason code；
- snapshot 和 event 记录 `tier_schema_version`；
- 保持 Memory 写入原子性。

### Phase C：Evidence 与 OPSD

- `EpisodeEvidence` 保存 typed payload 和 validator provenance；
- 更新 `write.jsonl` response schema；
- 更新数据审计器；
- 增加 per-tier attribution/样本统计；
- 保证 teacher privileged 与 student public 边界不变。

### Phase D：兼容迁移

- 实现 V1 只读兼容；
- 实现 dry-run migration；
- 对已有 `retail` Memory 生成迁移报告；
- 人工确认后发布新 snapshot。

### Phase E：检索增强与消融

- 增加结构化 Prompt rendering；
- 评估统一 Top-K 与 tier-aware retrieval；
- 分别执行 no-tip、no-skill、no-tool、no-trajectory 消融；
- 根据真实选择率和 attribution 再决定是否调整 tier prior。

## 验收标准

### Schema

- `tip` 无法携带 `steps`、`tool_name` 或 episode provenance。
- `skill` 少于两步必须拒绝。
- `tool` 引用不存在的工具或参数必须拒绝。
- `trajectory` 缺少 episode provenance 或 reward 必须拒绝。
- 混合职责内容必须拆分或拒绝。

### 运行时

- 合法 Memory 可写入、快照、恢复和检索。
- 非法 Memory 只产生脱敏 `MemoryWriteRejected`，不改变权威状态。
- 旧 V1 snapshot 可以只读加载。
- retier 不原地修改 Memory ID，lineage 可追溯。

### 数据管线

- event、evidence、attribution 和 `write.jsonl` 中 tier/payload 完全一致。
- 数据审计可以检测 payload/content 不一致和无效 tool reference。
- teacher privileged 信息不会进入 student public input。
- 相同输入生成确定性的 JSON、JSONL 和 artifact hash。

### 分类质量

准备至少40条人工标注 fixture，每层不少于10条，并满足：

- 四层 schema 校验通过率100%；
- 明确样本分类准确率不低于95%；
- `tool_name` 和参数引用合法率100%；
- `skill` 全部包含至少两个可执行步骤；
- `tip` 全部保持单点；
- `trajectory` 全部具有真实 episode lineage。

### 真实 Retail 验证

至少完成一个 Memory-enabled iteration，并报告：

- 各 tier proposed/accepted/rejected 数量；
- 错误分类和 retier 数量；
- 各 tier 检索、选择和 attribution 分布；
- 与 V1 的平均 reward、成功率和 prompt token 开销；
- 四个单 tier 消融结果。

## 设计结论

四层 Memory 的价值不应来自四个文件名，而应来自四套不同且可验证的知识契约：

```text
tip        约束一次判断
skill      编排一段流程
tool       指导一次真实工具调用
trajectory 保存一个可追溯案例
```

模型负责提出和结构化经验，代码负责分类边界、工具合法性、来源和泄漏校验。只有通过契约的内容才能进入权威 Memory、后续 attribution 和 OPSD 训练数据。
