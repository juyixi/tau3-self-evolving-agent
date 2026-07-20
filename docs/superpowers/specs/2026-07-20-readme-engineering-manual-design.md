# README 工程使用手册设计

## 状态

已确认，待实施。

## 定位

根目录 `README.md` 面向需要部署、运行和理解本项目的工程使用者。文档以当前代码接口为准，解释系统边界、模块职责、关键算法约束、实验控制变量和端到端命令。

README 不设置以下独立章节：

- 项目目标与当前完成度。
- 产物目录与复现说明。
- 已完成阶段和待验证事项。

## 信息架构

README 按以下顺序组织：

1. **系统架构**：使用 Mermaid 展示 Tau2 Retail、Fast Loop、四层 Memory、Outcome-Calibrated Attribution、OPD 数据集和共享策略 Slow Loop 的数据流。
2. **代码组织**：用表格说明 `envs/`、`evaluation/`、`memory/`、`fast_loop/`、`slow_loop/`、`models/`、`scripts/`、`tools/` 和 `tests/` 的职责。
3. **技术设计**：覆盖 Tau2 Retail 接入与 split guard、Memory、Fast Loop、Stage 5 归因与数据构建、共享 Qwen3.5-9B LoRA OPD。
4. **实验设计**：覆盖 train/test/base 边界、无 dev 决策、实验矩阵、控制变量、官方指标和 test 泄漏隔离。
5. **使用指南**：覆盖安装、环境变量、预检、Baseline、Fast Loop、OPD 数据构建与审计、LoRA 训练与恢复、GPU smoke 和自动化测试。
6. **参考资料**：列出 OPD-Evolver 论文/代码仓库、Tau2-bench 仓库和固定 revision。

## 内容约束

- 使用中文说明，命令、配置键、路径和代码标识保留英文。
- 不把 OPD 描述为自蒸馏或离线蒸馏；必须说明学生先在线采样，教师在相同响应 token 前缀上提供特权分布。
- 明确教师和学生共享同一个 Qwen3.5-9B 模型与当前 LoRA；教师加载 LoRA 但不参与梯度更新。
- 明确损失为响应位置上的全词表 `KL(teacher || student)`，基础模型冻结，只更新 LoRA。
- 明确 Memory 权威状态使用 JSON，事件、归因和训练样本使用 JSONL。
- 明确训练只使用官方 `train`，最终评测只使用官方 `test`，`base` 不得进入训练，并且不额外划分 dev。
- 使用指南中的参数名和命令必须来自当前 argparse/config 实现，不写尚不存在的统一入口。
- 实验设计可以描述 `test_static` 与 `test_streaming` 协议，但不得伪造尚未提供的 CLI 命令。
- 保留现有 AutoDL 训练、断点恢复和离线 GPU smoke 的有效说明。

## 验证

- 检查 README 中的模块路径、脚本名和参数名均存在。
- 运行所有脚本的 `--help` 或对应解析测试，确认命令示例没有漂移。
- 扫描 README，确保没有把 test/base 数据写入训练路径，也没有把 Fast Loop 历史输出描述为固定监督标签。
- 运行完整测试套件和 `git diff --check` 后提交并推送到 `origin/master`。
