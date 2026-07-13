# Task 4：Memory 开关文档、审计与完整回归报告

工作目录：`C:\Users\huang\source\aicoding\tau3-retail-evolver\.worktrees\memory-ablation-switch`

## 文档交付

- `docs/stage-4-fast-loop-validation.md`：增加 `memory.enabled: true|false` 的运行说明、专用 disabled YAML 示例、同一 `python -m scripts.run_fast_loop` 入口，以及 enabled/disabled 的产物和事件审计。
- `docs/superpowers/specs/2026-07-09-tau3-retail-opd-evolver-design.md`：在快循环设计中记录总开关、fake 无 Memory run 的 artifact/事件/无 `history/` 验收，以及 Stage 5 不得将 disabled run 用作 Memory 生命周期监督来源。
- 未修改 `docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md`。

## 完整验证

| 命令 | 结果 |
| --- | --- |
| `python -m pytest -q --basetemp=.pytest-tmp/memory-switch-final` | 通过：391 passed，3 skipped，7.34s。 |
| `python -m compileall -q src scripts tests` | 通过，exit code 0。 |
| `git diff --check` | 通过，exit code 0。 |

默认 suite 的 3 个 skip 为外部集成相关的预期跳过；无失败。

## 生命周期审计

- 产品文档修改范围仅为两份文档；另新增本审计报告，且没有新增、修改或删除 `src/`、`scripts/`、`tools/` 或 `tests/` 文件。
- `scripts/` 的已跟踪产品入口仍为 `scripts/run_baseline.py` 和 `scripts/run_fast_loop.py`（及包初始化文件）；没有一次性验证脚本进入该目录。
- `git ls-files history runs` 无输出，确认运行数据未被 Git 跟踪。
- `.pytest-tmp/` 由 `.gitignore` 忽略，完整测试的 workspace-local basetemp 未进入交付。
- 预先存在的未跟踪 `.pytest-tmp-review/` 未读取、修改、暂存或提交。
- 文档明确保留 Stage 5 attribution 公式和 `test_static`/`test_streaming` 隔离协议；仅新增 disabled run 不得作为 Memory 生命周期监督来源的约束。

## 结论

Task 4 文档、完整回归、静态检查和生命周期审计均已完成。文档提交为 `906c5dd`（`docs: document memory ablation workflow`）；本报告随后独立提交。
