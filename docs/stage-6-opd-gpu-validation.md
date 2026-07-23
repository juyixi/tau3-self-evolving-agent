# Stage 6 OPD GPU 验证记录

## 验证结论

Stage 6 真实 GPU gate 已于 2026-07-23 通过。固定 revision 的 Qwen3.5-9B 在 RTX 4090 48GB 上以 BF16 完成一个 forward-KL optimizer step；LoRA 梯度、参数更新、基础模型冻结、adapter-only 保存和逐 tensor 重载均满足约束。

该 gate 验证的是 Stage 6 最小训练闭环。真实 30 任务四类数据覆盖和完整多 iteration 不属于本次结论。

## 固定输入

- 训练环境：`/root/autodl-tmp/conda-envs/tau3-opd`
- GPU：NVIDIA GeForce RTX 4090，48GB
- PyTorch：`2.13.0+cu130`
- Transformers：`5.14.1`
- PEFT：`0.19.1`
- 模型：`Qwen/Qwen3.5-9B`
- 模型 revision：`c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Stage 5 数据集：`opd-iter0-5tasks-20260723g`
- 输入 adapter revision：`zero-impact-init-v1`
- loss：full-vocabulary forward KL
- LoRA：`r=32`、`alpha=64`、`dropout=0.05`、`target_modules=all-linear`

## 执行结果

正式 Stage 5 数据集首先通过 `scripts.train_opd_lora --dry-run`，确认数据审计、模型/adapter lineage、BF16 和训练参数一致。随后 opt-in GPU smoke 使用一个合成 `sel` 样本执行：

- optimizer steps：1
- forward KL：`1.0236316919326782`
- LoRA gradient norm：非零
- LoRA 参数：至少一个 tensor 在 optimizer step 后发生更新
- 基础模型 gradient：全部为 `None`
- checkpoint：`step-00000001`
- adapter 文件：`adapter_model.safetensors`
- adapter 文件大小：346,294,736 bytes
- adapter tensor 数：496
- 基础模型权重文件：无
- 重载验证：保存前后 tensor 名称集合一致，所有 tensor 数值逐项相等

加强后的 GPU 测试结果为 `1 passed, 1 warning in 24.04s`；Stage 6 相关 LoRA、Trainer 和 CLI 回归为 `121 passed in 2.96s`，完整默认测试套件为 `668 passed, 5 skipped in 8.30s`。剩余 warning 仅说明 PEFT 未从模型 ID 找到可选配置文件并假定词表未修改，不影响 adapter 保存或重载。

## 本次修复

首次 GPU smoke 虽然 pytest 返回通过，但 PEFT 在重载时报告全部 adapter key 缺失。检查产物后发现 `adapter_model.safetensors` 只有 40 bytes 且不含 tensor。

根因是保存逻辑先调用 `get_peft_model_state_dict()` 得到已经过滤并规范化的 adapter state，再把该 state 传入 `save_pretrained(selected_adapters=...)`。PEFT 对它进行第二次 adapter-name 筛选，最终写出空文件。

修复后不再向 `save_pretrained` 传入已过滤 state，由 PEFT 从模型原始 state 自行提取指定 adapter。GPU 测试同时新增保存前后 adapter tensor 的精确比较，避免“配置可加载但权重为空”的假阳性。

## 后续边界

- 真实 30 任务应继续验证 `sel/act/write/maint` 四类样本覆盖。
- Stage 7/8 应运行带真实 Stage 5 数据、promotion 和下一轮继承的完整 iteration。
- 单 GPU 执行训练前仍需停止 vLLM；训练完成后可按 rollout 需要重新启动服务。
