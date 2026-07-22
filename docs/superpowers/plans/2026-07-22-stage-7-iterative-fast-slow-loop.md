# Stage 7 迭代式 Fast/Slow Loop 实施计划

**目标：** 将既有 Fast Loop、Outcome-Calibrated Attribution、OPD 数据审计和共享策略 LoRA 训练组合为可恢复、train-only、具备完整父子 lineage 的 iteration。

**真实实验边界：** 本阶段完成代码、CLI 和合成双 iteration 验证；真实五任务 Memory-enabled build/audit 与 Qwen3.5-9B GPU update 延期到 Stage 8 验收实验，gate 保持 pending。

## 任务 1：Iteration 状态与恢复

- [x] 实现 `created -> rollout_complete -> attribution_complete -> dataset_complete -> training_complete -> promoted` 六态状态机。
- [x] 使用单 iteration 进程锁和原子 JSON 状态替换。
- [x] 每次状态推进记录 artifact 相对路径与 SHA-256。
- [x] 恢复前重算所有已完成产物哈希，拒绝修改或缺失的 artifact。
- [x] promotion 前验证 training manifest、latest checkpoint、adapter config 和唯一 adapter weight。

## 任务 2：父子 Lineage 与 Train-only Guard

- [x] 下一 iteration 强制继承父 promotion 的 checkpoint、adapter revision、Memory snapshot 和累计任务数。
- [x] 拒绝不连续 iteration、过期 adapter、错误 checkpoint 和 Memory snapshot 分叉。
- [x] 对 iteration 下 JSON/JSONL artifact 递归检查 task ID，拒绝 test/base 进入学习产物。

## 任务 3：Curriculum 与 OPD Sampling

- [x] 根据 seed 与 iteration 生成确定性 train task 顺序。
- [x] 支持 `--task-id` 精确 smoke 任务和 `--task-count` 数量控制。
- [x] 默认每轮使用完整 74 个官方 train tasks。
- [x] 对 `sel/act/write/maint` 进行已有类别均衡采样，不生成不存在的监督。
- [x] Stage 6 Trainer 复用 Stage 7 的统一 sampling 实现。

## 任务 4：生产 CLI 与单 GPU 分段运行

- [x] 新增 `python -m scripts.run_iteration`。
- [x] 复用 `run_fast_loop`、`build_opd_dataset`、`audit_opd_dataset` 和 `train_opd_lora` 入口。
- [x] 支持 `--stop-after dataset_complete`，允许单 GPU 机器先完成 rollout，再停止 vLLM 后恢复 LoRA 训练。
- [x] 已发布 rollout、dataset 和 checkpoint 可被恢复逻辑复用，不重复执行已提交阶段。

## 任务 5：验证与收尾

- [x] 完成各状态恢复、artifact 篡改、不完整 adapter、非 train ID 和双 iteration 合成测试。
- [x] 运行完整默认测试套件：`661 passed, 5 skipped`。
- [x] 执行代码生命周期审计。
- [x] 提交 Stage 7 功能分支。
- [ ] Stage 8 执行真实五任务、dataset audit、单 batch GPU update 和真实多 iteration 验收。
