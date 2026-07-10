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

The retail environment is provided by the current official tau benchmark
repository:

- Tau2-bench repository: https://github.com/sierra-research/tau2-bench
- Gym adapter documentation:
  https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/gym/README.md
- Retail task splits:
  https://github.com/sierra-research/tau2-bench/blob/main/data/tau2/domains/retail/split_tasks.json

The paper is the authoritative algorithm source. The repository is the
engineering reference for executable OPD-Evolver patterns, especially the
published executor OPD demonstration and project organization. Any tau3 retail
fast-loop and slow-loop design in this project must cite back to these two
sources when making algorithmic or structural decisions.

## Algorithm Interpretation

This project implements OPD-Evolver as on-policy distillation with a shared
policy model. It must not be described or implemented as offline
self-distillation:

- Student: Qwen3.5-9B plus the current LoRA adapter, running only deployment
  inputs.
- Teacher: the same Qwen3.5-9B instance with the same current LoRA adapter,
  given privileged hindsight context and evaluated under `no_grad` on the same
  prefixes sampled by the student.
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

The current official retail environment is consumed from
`sierra-research/tau2-bench`. The project continues to use "tau3 retail" as its
task-facing name, while integration code imports the official `tau2` Python
package. Tau2-bench is an external pinned dependency, not vendored application
code. The recommended local layout is `external/tau2-bench/`, ignored by git,
installed with its `gym` extra in editable mode.

Every real run records the tau2-bench Git commit, task split name, exact task
IDs, split file hash, user-simulator settings, seed, and environment options in
its run manifest. Unit tests use a mock or fake Gym object and do not require
the external repository.

The project adapter wraps `tau2.gym.gym_agent.AgentGymEnv` and exposes:

- `reset(task_id | task_spec) -> observation`
- `step(action) -> observation, reward, done, info`
- `task_group(task_spec) -> str`
- `metadata(task_spec) -> dict`
- `success(info, reward, done) -> bool`

The rest of the project depends only on this adapter interface. The adapter
normalizes Gymnasium's `(observation, reward, terminated, truncated, info)`
result, preserves official evaluator output from `info["reward_info"]`, and
closes each environment after an episode. A mock retail adapter is used for
unit tests and dry runs; real rollout and evaluation commands must use the
tau2-backed adapter without changing OPD code.

## Task Data And Split Policy

No internal dev split is created by default. The official retail splits have
one-way responsibilities:

- `train`: the only source for fast-loop rollout collection, memory writing and
  maintenance, outcome-calibrated attribution, privileged teacher hindsight,
  and LoRA updates.
- `test`: final or explicitly requested checkpoint evaluation only. Test tasks
  must never produce training examples, attribution updates, model updates, or
  mutations to train-derived memory.
- `base`: optional reproduction of the official all-task aggregate only. It is
  never a training source because it contains both train and test task IDs.

The task loader must enforce this policy in code rather than relying on CLI
convention. Training entry points reject `test` and `base`. Evaluation supports
two separately labeled protocols:

- `test_static`: opens a read-only snapshot of memory learned from `train`;
  test episodes cannot mutate it or share newly written memories.
- `test_streaming`: starts from an empty, evaluation-only memory repository and
  allows the paper's fast-loop memory evolution across the test stream. That
  repository is quarantined under the evaluation directory and all training
  artifact loaders reject it.

Neither protocol permits parameter updates, attribution dataset production, or
checkpoint selection from test outcomes.

The current official split file contains 74 train tasks, 40 test tasks, and 114
base tasks. These counts are compatibility assertions for the pinned
tau2-bench revision, not hard-coded task data; a revision change requires an
explicit split review and manifest update.

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

The implementation must align response-token positions between the public
student sequence and the longer privileged teacher sequence, and apply KL only
to the sampled response tokens. Concatenating privileged teacher context into a
single causal-language-model training string and applying ordinary next-token
SFT loss is explicitly not an OPD implementation.

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

Each run root also contains `manifest.json`, including the model revision,
LoRA revision, tau2-bench commit, split hash, task IDs, seeds, user-simulator
configuration, memory snapshot ID, and parent checkpoint.

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

The tau2-bench checkout and Qwen3.5-9B weights are external runtime
dependencies and are not committed to this repository. The user-simulator
model remains an explicit configuration choice because it affects credentials,
cost, and reproducibility; omitting it delegates to the default of the pinned
tau2-bench revision, and the resolved configuration is still recorded in the
run manifest.

The official OPD-Evolver repository is treated as an implementation reference.
If implementation-time inspection shows reusable code that is compatible with
this project's license and dependency constraints, the implementation plan
should either vendor that code with attribution or wrap it through a clearly
documented adapter.

## Acceptance Criteria

The project is successful when it provides:

- A reproducible Python project with configs and scripts for tau3 retail
  OPD-Evolver training.
- A real adapter for the pinned `sierra-research/tau2-bench` retail Gym
  environment, plus an offline fake for tests.
- Enforced split isolation: `train` for learning, `test` for evaluation,
  `base` for optional reproduction, and no default dev split.
- A fast loop that logs retail on-policy rollouts and four-tier memory lifecycle
  events.
- A slow loop that builds selection, execution, writing, and maintenance OPD
  examples from logged trajectories.
- A LoRA training entry point with defaults:
  `use_peft=true`, `lora_r=32`, `lora_alpha=64`.
- A shared-policy OPD trainer that computes stop-gradient teacher logits on
  student-sampled prefixes and response-token KL, rather than SFT loss.
- Test evaluators for frozen-memory `test_static` and quarantined
  paper-compatible `test_streaming`, both with a frozen checkpoint and official
  tau2 reward details plus reproducibility metadata.
- Documentation that explains how the tau3 retail fast loop and slow loop map
  to the OPD-Evolver paper and official repository.
- Unit tests that pass without requiring large model downloads.
