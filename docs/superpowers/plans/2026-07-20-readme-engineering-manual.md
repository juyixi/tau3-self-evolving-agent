# README Engineering Manual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the root README with a Chinese engineering manual that accurately summarizes the repository architecture, technical design, experiment design, and executable workflows.

**Architecture:** Keep the README as one navigable document organized from system architecture to module responsibilities, technical mechanisms, experiment controls, and commands. Treat the current Python modules, argparse definitions, `configs/default.yaml`, and pinned Tau2 revision as the source of truth; describe future evaluation protocols only as experiment design and do not invent commands.

**Tech Stack:** Markdown, Mermaid, Python CLI, Tau2 Retail Gym, JSON/JSONL, Qwen3.5-9B, Transformers, PEFT LoRA, PyTorch.

## Global Constraints

- Write explanatory prose in Chinese while preserving commands, paths, configuration keys, and identifiers in English.
- Do not add standalone sections for project goals/status, artifact layout/reproducibility, or completed/pending stages.
- Describe OPD as online policy distillation, not self-distillation or offline distillation.
- State that teacher and student share one Qwen3.5-9B model and current LoRA storage; teacher runs with no gradients.
- State that the loss is full-vocabulary `KL(teacher || student)` on aligned student response positions.
- State that official `train` is the only learning source, `test` is evaluation-only, `base` cannot train, and no dev split is created.
- Every command-line option in the README must exist in the current implementation.
- Preserve the valid AutoDL, resume, and opt-in GPU smoke guidance from the existing README.

---

### Task 1: Rewrite README As An Engineering Manual

**Files:**
- Modify: `README.md`
- Reference: `configs/default.yaml`
- Reference: `external/tau2-bench.commit`
- Reference: `docs/superpowers/specs/2026-07-20-readme-engineering-manual-design.md`

**Interfaces:**
- Consumes: current package layout and the CLIs in `scripts/` and `tools/preflight/`.
- Produces: one Chinese root README with architecture, module, technical, experiment, usage, and reference sections.

- [x] **Step 1: Replace the document structure**

Use exactly these top-level sections after the title and one-sentence repository description:

```markdown
## 系统架构
## 代码组织
## 技术设计
## 实验设计
## 使用指南
## 参考资料
```

- [x] **Step 2: Add architecture and module boundaries**

Add a Mermaid flowchart connecting Tau2 Retail, Baseline/Fast Loop, four-tier Memory, Stage 5 evidence/attribution/dataset building, and shared-policy OPD training. Add a module table covering `envs`, `evaluation`, `memory`, `fast_loop`, `slow_loop`, `models`, `scripts`, `tools`, and `tests`.

- [x] **Step 3: Add the technical design**

Document these exact mechanisms:

```text
Tau2 commit: 1901a301961cbbe3fd11f3e84a2a376530c759e3
Memory tiers: trajectory, tip, skill, tool
Memory state: JSON
Events/attribution/OPD examples: JSONL
Maintenance period: Q=30 completed train tasks
Attribution: candidate-controlled Eq.11/Eq.12 value V(m)
OPD kinds: sel, act, write, maint
Base model: Qwen/Qwen3.5-9B
LoRA: r=32, alpha=64, dropout=0.05, all-linear, zero-impact initialization
Loss: full-vocabulary forward KL(teacher || student)
```

- [x] **Step 4: Add the experiment design**

Include split isolation, no-dev policy, the primary experiment matrix, fixed comparison variables, official Tau2 metrics, four seeded trials for final evaluation, and test artifact quarantine. Present `test_static` and `test_streaming` as evaluation protocols without a nonexistent CLI example.

- [x] **Step 5: Add the executable guide**

Include install/configuration and examples for:

```text
python -m tools.preflight.check_tau2_retail
python -m scripts.run_baseline
python -m scripts.run_fast_loop
python -m scripts.build_opd_dataset
python -m scripts.audit_opd_dataset
python -m scripts.train_opd_lora --dry-run
python -m scripts.train_opd_lora
python -m pytest -q
```

Keep the existing AutoDL setup, adapter resume, and `RUN_OPD_GPU_SMOKE=1` instructions, adapting them to the new section hierarchy.

---

### Task 2: Validate, Commit, And Publish

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-20-readme-engineering-manual.md`

**Interfaces:**
- Consumes: the rewritten README and current command parsers.
- Produces: a validated documentation commit on `master`, pushed to `origin/master`.

- [x] **Step 1: Validate CLI examples**

Run each documented Python module with `--help` in the `tau3-bench` environment and expect exit code 0. Compare all required arguments with the README examples.

- [x] **Step 2: Run documentation consistency scans**

Run:

```powershell
rg -n "采用自蒸馏|使用自蒸馏|采用离线蒸馏|使用离线蒸馏|--split dev" README.md
rg -n "run_iteration|evaluate_retail|compare_evaluations" README.md
git diff --check
```

Expected: the first and second scans return no matches; `git diff --check` exits 0.

- [x] **Step 3: Run the complete test suite**

Run:

```powershell
conda run -n tau3-bench python -m pytest -q -p no:cacheprovider --basetemp=.pytest-tmp/readme-final
```

Expected: all local tests pass; opt-in integration tests skip unless their environment variables are enabled.

- [x] **Step 4: Mark this plan complete and commit**

Stage only `README.md` and this plan, then commit:

```bash
git commit -m "docs: expand engineering readme"
```

- [x] **Step 5: Push**

Run:

```bash
git push origin master
```

Expected: `origin/master` resolves to the new documentation commit.

## Self-Review

- Spec coverage: every approved README section and exclusion has a matching implementation step.
- Placeholder scan: the plan contains no implementation placeholders.
- Interface consistency: every documented executable module exists in the repository; planned Stage 7/8 commands are explicitly excluded.
