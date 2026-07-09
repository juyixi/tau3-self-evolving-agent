# Tau3 Retail OPD-Evolver Training Project Design

## Purpose

Build a training project for tau3-bench retail tasks using Qwen3.5-9B as both
the privileged teacher and the deployable student. The training method is
on-policy distillation, following OPD-Evolver rather than offline
self-distillation. The student is trained with LoRA adapters, not full
fine-tuning.

The project should make the OPD-Evolver fast loop and slow loop explicit for
tau3 retail:

- Fast loop: the agent interacts with retail tasks, retrieves and selects
  memories, acts, writes new memories, and periodically maintains the memory
  repository.
- Slow loop: the same Qwen3.5-9B policy is trained on student-visited states by
  comparing student distributions against a privileged teacher view on the same
  student prefixes.

## Primary References

The implementation must reference both OPD-Evolver sources:

- Paper: "OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy
  Distillation", arXiv 2606.17628v1.
  - Local copy: `C:\Users\huang\Downloads\2606.17628v1.pdf`
  - arXiv: https://arxiv.org/abs/2606.17628v1
- Official code repository:
  - GitHub: https://github.com/bingreeky/opd-evolver

The paper is the authoritative algorithm source. The repository is the
engineering reference for executable OPD-Evolver patterns, especially the
published executor OPD demonstration and project organization. Any tau3 retail
fast-loop and slow-loop design in this project must cite back to these two
sources when making algorithmic or structural decisions.

## Algorithm Interpretation

This project implements OPD-Evolver as an on-policy self-distillation system
with a shared backbone:

- Student: Qwen3.5-9B plus the current LoRA adapter, running only deployment
  inputs.
- Teacher: Qwen3.5-9B with privileged hindsight context and stop-gradient
  parameters, evaluated on the same prefixes sampled by the student.
- Training signal: token-level distillation loss on student-visited prefixes.
- Scope: four lifecycle decisions from the paper:
  - `sel`: experience selection.
  - `act`: experience-grounded execution.
  - `write`: experience writing.
  - `maint`: repository maintenance.

The project must not treat OPD as static trajectory self-distillation. Rollouts
come from the current student policy, and distillation targets are computed
against those on-policy states.

## Tau3 Retail Environment Boundary

Tau3 retail is represented by an environment adapter so the training framework
can work before and after the exact tau3-bench package or local harness is
available.

The adapter exposes:

- `reset(task_id | task_spec) -> observation`
- `step(action) -> observation, reward, done, info`
- `task_group(task_spec) -> str`
- `metadata(task_spec) -> dict`
- `success(info, reward, done) -> bool`

The rest of the project depends only on this adapter interface. A mock retail
adapter is acceptable for unit tests and dry runs, but real training scripts
must be able to swap in a tau3-bench retail adapter without changing OPD code.

## Fast Loop Design

The fast loop follows Algorithm 1 in the paper.

For each retail task:

1. Form a query from the task, environment metadata, current observation, and
   optional retail state hints.
2. Retrieve candidate memories from a four-tier repository:
   - `trajectory`: full or compressed retail interaction traces.
   - `tip`: local warnings, constraints, or heuristics.
   - `skill`: reusable retail task procedures.
   - `tool`: executable or structured action templates.
3. Select a compact subset of memories with the current student policy.
4. Format the selected memories into the execution prompt.
5. Roll out the current student policy in the retail environment.
6. Log observations, actions, rewards, selected memories, candidate memories,
   task group, terminal result, and parse errors.
7. Ask the current policy to write memory updates from the task, trajectory,
   result, and selected memories.
8. Every `maintenance_period` tasks, run repository maintenance using
   `lookup`, `merge`, and `delete` operations.

Default values from the paper:

- Memory tiers: `trajectory`, `tip`, `skill`, `tool`.
- Teacher-side retrieval candidates: 50.
- Privileged teacher memory injection cap: 20.
- Maintenance period: `Q = 30`.
- Maximum episode length: 40 unless tau3 retail requires a different cap.

## Outcome-Calibrated Attribution

The project computes memory value only from tasks where a memory was retrieved.
For a memory `m` and task group `g`, selected uses are compared against
retrieved-but-not-selected uses. This follows the paper's candidate-controlled
attribution idea:

- `Omega_plus_g(m)`: tasks in group `g` where `m` was retrieved and selected.
- `Omega_g(m)`: tasks in group `g` where `m` was retrieved but not selected.
- Attribution compares average returns between those two sets.
- A confidence factor downweights memories with little selected evidence.
- A tier prior allows different weighting for trajectory, tip, skill, and tool
  memories.

The resulting score `V(m)` is used as privileged hindsight for selection,
execution, writing, and maintenance. Supervision with memory score below
`0.01` is filtered by default, matching the paper.

## Slow Loop Design

The slow loop constructs training examples for the four lifecycle decisions:

- Selection:
  - Student input `z_sel`: retail task plus retrieved candidate memories.
  - Teacher hindsight `h_sel`: each candidate memory with calibrated value
    `V(m)`.
- Execution:
  - Student input `z_act`: retail task and public environment history, without
    privileged memory values.
  - Teacher hindsight `h_act`: valuable selected memories and a successful
    trajectory from the same task group when available.
- Writing:
  - Student input `z_write`: task, trajectory, return, and selected memories.
  - Teacher hindsight `h_write`: generated memory candidates with future
    calibrated value.
- Maintenance:
  - Student input `z_maint`: repository snapshot, logged history, and available
    maintenance tools.
  - Teacher hindsight `h_maint`: memory value, confidence, usage statistics,
    and redundancy diagnostics.

For each example, the student first samples an output under deployment
conditions. The teacher is then evaluated on the same student-generated prefixes
with the privileged hindsight appended. The training loss is token-level KL from
the stop-gradient teacher distribution to the student distribution.

The deployable artifact is only the student-facing LoRA adapter. Privileged
hindsight contexts are never required at inference time.

## Model And Training Defaults

Base model:

- `Qwen/Qwen3.5-9B` or the locally available equivalent Qwen3.5-9B checkpoint.

Precision and context:

- `bf16`
- `max_prompt_length = 8192`

Rollout and distillation data generation:

- `temperature = 1.0`
- `top_p = 0.95`
- `max_episode_steps = 40`

LoRA:

- `use_peft = true`
- `lora_r = 32`
- `lora_alpha = 64`
- `lora_dropout = 0.05`
- Full-parameter fine-tuning is out of scope for this project.

Training defaults:

- `learning_rate = 1e-5`
- `per_device_train_batch_size = 2`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 3`

These defaults intentionally match the OPD-Evolver paper where applicable and
can be overridden from config files or CLI flags.

## Project Architecture

The repository will be organized around small modules with clear boundaries:

- `configs/`
  - Model, LoRA, rollout, memory, tau3 retail, and OPD training settings.
- `src/tau3_retail_evolver/envs/`
  - Tau3 retail adapter interface, mock adapter, and real adapter scaffold.
- `src/tau3_retail_evolver/memory/`
  - Four-tier memory store, retrieval, formatting, scoring, and maintenance
    operations.
- `src/tau3_retail_evolver/fast_loop/`
  - Retail rollout, selection, execution, writing, maintenance orchestration,
    and logging.
- `src/tau3_retail_evolver/slow_loop/`
  - Attribution, hindsight construction, OPD example building, and token-level
    distillation training.
- `src/tau3_retail_evolver/models/`
  - Qwen model loading, LoRA loading/saving, tokenizer handling, and teacher
    plus student wrappers.
- `scripts/`
  - One-command entry points for rollout, attribution, OPD training, evaluation,
    and an end-to-end iteration.
- `tests/`
  - Unit tests for memory scoring, adapter behavior, loop logging, and example
    construction.

## Data Layout

Runtime artifacts are stored under `runs/` and ignored by git:

- `runs/<run_id>/rollouts/*.jsonl`
- `runs/<run_id>/memory/*.jsonl`
- `runs/<run_id>/attribution/*.jsonl`
- `runs/<run_id>/opd_examples/*.jsonl`
- `runs/<run_id>/checkpoints/`
- `runs/<run_id>/eval/`

Each logged event should preserve enough information to reconstruct:

- Retrieved candidates `C_t`
- Selected memories `S_t`
- Episode trajectory `tau_t`
- Return `R_t`
- Written memory updates `Delta_t`
- Maintenance trajectory `eta_q`
- Task group `g(t)`

## Testing Strategy

The first implementation should be testable without downloading Qwen3.5-9B or
running tau3-bench:

- Mock retail tasks validate the adapter and fast-loop control flow.
- Deterministic fake policies validate selection, writing, and maintenance
  logs.
- Synthetic rollouts validate outcome-calibrated attribution.
- Small toy logits validate token-level KL loss plumbing.
- CLI smoke tests validate config loading and run-directory creation.

GPU-heavy model tests are separate integration tests and should be skipped by
default unless the required environment variables and model checkpoints are
available.

## Open Integration Points

The exact tau3-bench retail package and task runner are not assumed to be
installed in the empty project folder. The real adapter must be implemented
behind the environment boundary once the local tau3 retail harness is available.

The official OPD-Evolver repository is treated as an implementation reference.
If implementation-time inspection shows reusable code that is compatible with
this project's license and dependency constraints, the implementation plan
should either vendor that code with attribution or wrap it through a clearly
documented adapter.

## Acceptance Criteria

The project is successful when it provides:

- A reproducible Python project with configs and scripts for tau3 retail
  OPD-Evolver training.
- A fast loop that logs retail on-policy rollouts and four-tier memory lifecycle
  events.
- A slow loop that builds selection, execution, writing, and maintenance OPD
  examples from logged trajectories.
- A LoRA training entry point with defaults:
  `use_peft=true`, `lora_r=32`, `lora_alpha=64`.
- Documentation that explains how the tau3 retail fast loop and slow loop map
  to the OPD-Evolver paper and official repository.
- Unit tests that pass without requiring large model downloads.
