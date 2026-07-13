# Fast Loop Memory 消融开关设计

## 背景与目标

当前 `scripts.run_fast_loop` 会无条件打开训练 Memory、加载 embedding provider，并执行 retrieve、select、act、write 和 periodic maintenance。为了在相同 Qwen、任务顺序、seed、用户模拟器和执行入口下进行无 Memory 消融，需要增加一个配置级总开关。

本设计只增加总开关，不在这一轮引入 selection、writing 或 maintenance 的独立细粒度开关。

## 方案选择

采用 `memory.enabled` 总开关，默认值为 `true`：

```yaml
memory:
  enabled: true
```

考虑过但不采用以下方案：

1. 单独增加 `run_no_memory` 入口。它会让有 Memory 和无 Memory 实验经过不同 orchestration 路径，降低消融可比性。
2. 立即增加 retrieve、select、write、maintain 四组独立开关。它更灵活，但会扩大状态组合、日志契约和测试矩阵；应在总开关稳定后按具体消融需求增量实现。

`enabled` 必须是严格 YAML boolean。字符串 `"false"`、整数 `0` 等隐式值必须被配置校验拒绝。

## 开关语义

### `memory.enabled: true`

保持现有 Stage 4 行为不变：

1. 打开 `history/agents/<agent_id>/memory/` 下的共享训练 Memory。
2. 加载 embedding provider 并检索候选 Memory。
3. 由当前策略选择候选并将选中内容注入 action prompt。
4. Episode 完成后由当前策略生成并持久化 Memory。
5. 按累计 train task 数执行周期性 maintenance。
6. manifest、summary 和事件保留输入与输出 Memory snapshot provenance。

### `memory.enabled: false`

关闭完整 Memory 生命周期：

1. 不调用 `open_training_memory`。
2. 不加载 embedding provider，不执行检索或选择。
3. action prompt 中完全不包含 `memories` 字段或其他 Memory 占位内容。
4. Episode 完成后不生成 write prompt，不写入 Memory。
5. 不调用 maintenance scheduler。
6. 不创建或修改 `history/agents/<agent_id>/memory/`。

任务环境、Qwen action generation、trajectory、官方 terminal reward/result 和失败证据仍按原流程执行。每个 episode 在 `EpisodeStarted` 后记录一个 `MemoryDisabled` 事件，字段固定为 `reason="config"`，用于证明本次跳过来自显式配置，而不是检索失败。

关闭开关时 `EpisodeResult.selected_memory_ids` 和 `written_memory_ids` 都为空。Memory 相关事件 `MemoryCandidatesRetrieved`、`MemorySelected`、`MemoryWriteProposed`、`MemoryWriteCommitted`、`MemoryWriteFailed` 以及全部 maintenance 事件不得出现。

## 配置与运行接口

开关只由 YAML 配置决定，不增加 CLI override。消融运行应使用单独、受版本控制的配置文件或配置副本，避免命令行覆盖值没有进入配置审计。

`--completed-train-tasks-before` 继续作为必填参数并记录累计完成的 train task 数，以保持有 Memory 和无 Memory run 的任务计数口径一致；关闭开关时该值不会触发 maintenance。

`FastLoopConfig` 携带 `memory_enabled`。Episode runner 在启用时要求 mutable repository 和 retriever 均存在，在关闭时要求二者均不存在，拒绝一半启用的模糊状态。

## 运行产物

manifest 的 `rollout_options` 增加 `memory_enabled`。summary 顶层增加同名字段。

启用时：

- `memory_snapshot_id`、`input_memory_snapshot_id` 和 `output_memory_snapshot_id` 保持现有字符串值。

关闭时：

- manifest 的 `memory_snapshot_id` 为 `null`。
- summary 的 `input_memory_snapshot_id` 和 `output_memory_snapshot_id` 均为 `null`。
- `maintenance_rounds_executed` 固定为空数组。
- 不产生任何 Memory snapshot 或 maintenance state。

Stage 5 的 attribution 和 OPD dataset builder 必须拒绝将 `memory_enabled=false` 的 run 作为 Memory 生命周期监督来源；这类 run 只用于无 Memory baseline/ablation 指标比较。

## 错误处理

- `memory_enabled=true` 但 repository 或 retriever 缺失时，在环境 reset 前失败。
- `memory_enabled=false` 但仍传入 repository 或 retriever 时，在环境 reset 前失败，避免调用方误以为 Memory 未被访问。
- 无 Memory 模式下的 action、环境或 close 异常仍使用现有 `FastLoopFailed` 证据和异常传播语义。
- 配置关闭不能降级为“检索到零条 Memory”；显式 `MemoryDisabled` 与正常空检索必须可区分。

## 测试与验收

单元测试必须证明：

1. 默认配置继续启用 Memory，现有 Stage 4 测试不回归。
2. 配置只接受严格 boolean。
3. 关闭时 policy 只收到 action prompt，且 action payload 不包含 `memories`。
4. 关闭时 repository、retriever、embedding factory 和 maintenance scheduler 均未被调用。
5. 关闭时只出现 `MemoryDisabled`，不出现任何 retrieve/select/write/maintenance 事件。
6. 关闭时 manifest 和 summary 显式记录 `memory_enabled=false`，全部 Memory snapshot 字段为 `null`。
7. 关闭时项目根目录下不会创建 `history/`。
8. 启用时原有 snapshot、写入和 maintenance 行为保持不变。

最终 gate 为：完整测试套件通过；一个 fake 无 Memory run 能正常完成任务并通过 artifact 审计；有 Memory 路径的 Stage 4 lifecycle 测试保持绿色。
