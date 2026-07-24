# Stage 7 验收记录

## 验收结论

Stage 7 的 iteration 编排、失败恢复、train-only 隔离、artifact 哈希校验和单 GPU
分段运行 gate 已通过。

本阶段不把“随机五任务必须生成非空 OPD 样本并完成真实 promotion”作为通过条件。
该条件需要扩大真实 train 任务覆盖，并按既定计划在 Stage 8 完成。

## 验收环境

- 日期：2026-07-24
- 基础模型：`Qwen/Qwen3.5-9B`
- 模型 revision：`c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Tau2 revision：`1901a301961cbbe3fd11f3e84a2a376530c759e3`
- LoRA：`r=32`、`alpha=64`、`dropout=0.05`
- OPD loss：forward KL
- 用户模拟器与 NL evaluator：`deepseek/deepseek-v4-pro`
- iteration：`stage7-acceptance-iter0`
- train task：`0, 1, 2, 3, 4`

## 真实运行结果

- 五个 Tau2 retail train episode 全部完成，共写入 150 条审计事件。
- terminal reward 总和为 `3.0`。
- Memory 从输入快照
  `ed80c10158fb3b31a4b28d4254ebb1195a7d97e462f403231913e603216bb230`
  演进到输出快照
  `70e5a932c62aa14f1e2c5a0392e6d1b37bf0cf9d830e5e4e3dcf5b1513fb626e`。
- iteration 成功推进到 `dataset_complete`。
- Outcome-Calibrated Attribution 生成 12 条 Memory score。
- OPD dataset 独立 audit 通过，错误列表为空。
- 所有本轮学习 artifact 均通过官方 train split guard。
- 再次执行同一命令直接返回 `dataset_complete`，已完成 rollout、dataset build
  和 audit 均未重跑，artifact 哈希验证通过。

本轮四类 OPD 数据均为 0 条。主要跳过原因为没有足够的 selected/control
对照、没有可用的未来 write evidence，以及同组成功轨迹覆盖不足。Trainer
会对空数据集 fail closed，因此本轮没有创建无意义 checkpoint，也没有发布
promotion manifest。

## 故障恢复验证

真实 rollout 首次暴露 Qwen lifecycle 输出不是合法 JSON，第二次暴露合法 JSON
被额外包装为 `SelectionDecision`。修复后：

- selection、write 和 maintenance 请求使用 vLLM `json_schema` response format；
- action 仍使用官方 Tau2 tools，不受 lifecycle schema 约束；
- rollout 失败时把四个 Memory tier 恢复到 iteration 输入快照；
- 同一 iteration 重试时优先使用持久化 identity 中的输入快照；
- 失败后的当前 Memory 文件与输入快照逐文件一致；
- 同一 iteration ID 在两次真实失败后均可恢复，并最终发布五任务 rollout。

## 训练门禁

Stage 6 已保存的真实 GPU smoke 继续作为 Stage 7 的训练侧门禁：

- Qwen3.5-9B BF16 完成 1 个 optimizer step；
- teacher 和 student 共享模型，teacher 路径冻结；
- forward KL；
- LoRA `r=32`、`alpha=64`；
- 生成 `adapter_model.safetensors`、`adapter_config.json`、
  `checkpoint_manifest.json` 和 `training_manifest.json`；
- manifest 状态为 `complete`，adapter revision 为
  `opd-step-00000001`。

## 自动化测试

远程验收工作树完整测试结果：

```text
670 passed, 5 skipped in 7.64s
```

五个 skip 均为显式环境变量控制的真实 Tau2、Qwen loader 或 GPU integration
测试。本次验收已手工覆盖真实 Tau2 Fast Loop；真实 Stage 6 GPU smoke 由已校验
备份产物覆盖。

## Stage 8 边界

Stage 8 仍需扩大到足以覆盖 `sel`、`act`、`write`、`maint` 四类非空样本的真实
train 任务集合，并完成至少一个真实 `promoted` iteration 及下一 iteration 的
checkpoint、adapter revision、Memory snapshot 和累计任务数继承。
