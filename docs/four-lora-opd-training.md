# 四能力 LoRA OPSD 训练设计

## 设计结论

Slow Loop 将四类生命周期决策视为四个独立能力模块：

| OPD kind | 运行时职责 | LoRA |
| --- | --- | --- |
| `sel` | 从候选 Memory 中选择经验 | `sel` adapter |
| `act` | 基于公开上下文执行 Tau2 action | `act` adapter |
| `write` | 从 episode 结果提取并写入经验 | `write` adapter |
| `maint` | 合并、保留或淘汰 Memory | `maint` adapter |

四个 adapter 共用同一个 Qwen3.5-9B 基础模型 revision，但参数互不继承、优化器
互不共享、checkpoint 和恢复边界互相隔离。Iteration 0 的四套 adapter 都从
`zero-impact-init-v1` 独立初始化。

“Teacher 和 Student 共享模型”只在单个能力训练内部成立：Teacher 和 Student
加载该能力的同一份当前 LoRA；Teacher 使用 `eval()`、stop-gradient 和特权
hindsight，Student 使用公开输入进行 on-policy sampling，并且只有该能力的
Student LoRA 接收 forward KL 梯度。

## 自然样本调度

禁止用最大类别数量补齐其他类别，也禁止通过 round-robin 重复少数类样本。
每个 epoch 对某一 kind 的全部独立样本做一次确定性 shuffle，每条样本恰好出现
一次。三个 epoch 仅表示对同一 kind 的自然数据遍历三次。

Stage 8 Iteration 0 数据量为：

| kind | 每 epoch 独立样本 | 3 epochs 训练曝光 |
| --- | ---: | ---: |
| `sel` | 198 | 594 |
| `act` | 1586 | 4758 |
| `write` | 50 | 150 |
| `maint` | 7 | 21 |
| 合计 | 1841 | 5523 |

`memory_scores.jsonl` 是 attribution 证据，不作为第五类训练样本，也不计入上述
训练曝光数。

## 产物和恢复

四 LoRA 套件输出目录为：

```text
<output>/
  suite_manifest.json
  sel/
    training_manifest.json
    checkpoints/
  act/
    training_manifest.json
    checkpoints/
  write/
    training_manifest.json
    checkpoints/
  maint/
    training_manifest.json
    checkpoints/
```

每个子目录独立记录 `opd_kind`、自然调度 fingerprint、generation、forward KL、
optimizer 和 RNG。正在训练的能力保留最近两个完整 checkpoint 供中断恢复；该能力
训练完成后只保留最终 checkpoint，避免四组 optimizer 状态耗尽 AutoDL 训练盘。
恢复只能从同一 kind 的最新完整 checkpoint 继续，不能使用旧的四类混合 checkpoint。

标准入口为：

```bash
python -m scripts.train_opd_lora_suite \
  --config configs/default.yaml \
  --dataset-dir <audited-dataset>/slow_loop \
  --output-dir <four-lora-output> \
  --model-revision <qwen-revision> \
  --adapter-revision zero-impact-init-v1
```

套件按样本量从小到大顺序执行，优先完成 `maint` 和 `write` 以尽早验证完整
checkpoint 契约；该执行顺序不改变四套 LoRA 的初始化或数据。

## 推理路由

部署时由同一个 Qwen3.5-9B 服务加载四套 LoRA，并按照 Fast Loop prompt kind
显式路由。`selection` 只能调用 `sel` adapter，`action` 只能调用 `act` adapter，
`write` 只能调用 `write` adapter，`maintenance` 只能调用 `maint` adapter。
Cell C 必须同时使用同一 bundle 中的四套 checkpoint；Cell D 只关闭 Memory
生命周期调用，但 action 仍使用同一 bundle 的 `act` adapter。
