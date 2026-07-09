# Tau3 Retail OPD-Evolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable Python training project for tau3-bench retail OPD-Evolver with Qwen3.5-9B as shared teacher/student and LoRA student updates.

**Architecture:** The project is split into an environment adapter, four-tier memory, fast-loop rollout logging, slow-loop attribution/example construction, and OPD LoRA training. Unit tests use mock retail tasks and toy logits so the project is verifiable without downloading Qwen3.5-9B or installing tau3-bench.

**Tech Stack:** Python 3.12 target, pytest, PyYAML, PyTorch, Transformers, PEFT, Accelerate, optional vLLM, JSONL runtime artifacts.

## Global Constraints

- Primary algorithm source: `C:\Users\huang\Downloads\2606.17628v1.pdf` and https://arxiv.org/abs/2606.17628v1.
- Official engineering reference: https://github.com/bingreeky/opd-evolver.
- OPD-Evolver paper is authoritative for fast-loop, attribution, and slow-loop objectives.
- Official repository is a reference for Python 3.12/uv structure, rollout/memory scoring/data construction, and executor OPD demo.
- Base model default is `qwen/qwen3.5-9b`.
- Student and teacher share the Qwen3.5-9B backbone; the student carries the current LoRA adapter.
- Full-parameter fine-tuning is out of scope.
- LoRA defaults: `use_peft=true`, `lora_r=32`, `lora_alpha=64`, `lora_dropout=0.05`.
- Precision and context defaults: `bf16`, `max_prompt_length=8192`.
- Distillation generation defaults: `temperature=1.0`, `top_p=0.95`, `max_episode_steps=40`.
- OPD memory defaults: tiers are `trajectory`, `tip`, `skill`, `tool`; retrieve 50 candidates; cap privileged teacher memory injection at 20; score threshold is `0.01`; maintenance period is `Q=30`.
- Unit tests must not download models, call external APIs, require GPUs, or require tau3-bench.
- Runtime outputs go under `runs/` and are ignored by git.

---

## File Structure

- `pyproject.toml`: package metadata, dependencies, and pytest config.
- `.gitignore`: excludes caches, local envs, checkpoints, and `runs/`.
- `README.md`: user-facing setup and OPD-Evolver mapping notes.
- `configs/default.yaml`: defaults copied from the spec.
- `src/tau3_retail_evolver/config.py`: typed configuration loader with CLI override support.
- `src/tau3_retail_evolver/envs/base.py`: tau3 retail adapter protocol and shared task/result types.
- `src/tau3_retail_evolver/envs/mock_retail.py`: deterministic mock retail environment for tests.
- `src/tau3_retail_evolver/envs/tau3_retail.py`: import-backed real tau3 adapter with a helpful dependency error.
- `src/tau3_retail_evolver/memory/types.py`: memory item, candidate, selection, and tier types.
- `src/tau3_retail_evolver/memory/store.py`: in-memory four-tier memory store used by tests and rollout orchestration.
- `src/tau3_retail_evolver/memory/retrieval.py`: deterministic lexical retriever with an embedding hook boundary.
- `src/tau3_retail_evolver/memory/formatting.py`: memory context formatter for prompts.
- `src/tau3_retail_evolver/models/policy.py`: policy protocol, generation request/response, fake policy.
- `src/tau3_retail_evolver/models/qwen.py`: Qwen + LoRA loader wrapper isolated from unit tests.
- `src/tau3_retail_evolver/fast_loop/events.py`: JSON-serializable rollout, write, and maintenance events.
- `src/tau3_retail_evolver/fast_loop/prompts.py`: selection, action, writing, and maintenance prompt builders.
- `src/tau3_retail_evolver/fast_loop/runner.py`: Algorithm 1 fast-loop orchestration.
- `src/tau3_retail_evolver/slow_loop/attribution.py`: outcome-calibrated memory value computation.
- `src/tau3_retail_evolver/slow_loop/examples.py`: `z_k`/`h_k` OPD example construction.
- `src/tau3_retail_evolver/slow_loop/loss.py`: token-level KL distillation utilities.
- `src/tau3_retail_evolver/slow_loop/train.py`: LoRA OPD trainer entry point with dry-run mode.
- `src/tau3_retail_evolver/io/jsonl.py`: JSONL read/write helpers.
- `scripts/run_rollout.py`: fast-loop CLI.
- `scripts/build_attribution.py`: attribution CLI.
- `scripts/build_opd_dataset.py`: slow-loop dataset CLI.
- `scripts/train_lora.py`: LoRA OPD training CLI.
- `scripts/run_iteration.py`: one iteration CLI chaining rollout, attribution, dataset, train.
- `tests/`: focused tests for config, envs, memory, fast loop, attribution, examples, loss, and CLI smoke checks.

---

### Task 1: Project Foundation And Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `configs/default.yaml`
- Create: `src/tau3_retail_evolver/__init__.py`
- Create: `src/tau3_retail_evolver/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path = "configs/default.yaml", overrides: Sequence[str] = ()) -> ProjectConfig`
- Produces: dataclasses `ModelConfig`, `LoraConfig`, `RolloutConfig`, `MemoryConfig`, `TrainingConfig`, `Tau3Config`, `ProjectConfig`
- Consumes: none

- [ ] **Step 1: Write the failing config test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from tau3_retail_evolver.config import load_config


def test_default_config_matches_opd_evolver_defaults():
    cfg = load_config(Path("configs/default.yaml"))

    assert cfg.model.name_or_path == "qwen/qwen3.5-9b"
    assert cfg.model.torch_dtype == "bf16"
    assert cfg.model.max_prompt_length == 8192
    assert cfg.lora.use_peft is True
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 64
    assert cfg.lora.dropout == 0.05
    assert cfg.rollout.temperature == 1.0
    assert cfg.rollout.top_p == 0.95
    assert cfg.rollout.max_episode_steps == 40
    assert cfg.memory.tiers == ("trajectory", "tip", "skill", "tool")
    assert cfg.memory.retrieve_top_k == 50
    assert cfg.memory.teacher_memory_cap == 20
    assert cfg.memory.score_threshold == 0.01
    assert cfg.memory.maintenance_period == 30


def test_override_updates_nested_value():
    cfg = load_config(Path("configs/default.yaml"), overrides=("training.learning_rate=2e-5",))

    assert cfg.training.learning_rate == 2e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'tau3_retail_evolver'` or missing config file.

- [ ] **Step 3: Add package metadata and defaults**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tau3-retail-evolver"
version = "0.1.0"
description = "Tau3 retail OPD-Evolver training project"
requires-python = ">=3.12"
dependencies = [
  "accelerate>=0.33",
  "peft>=0.12",
  "pyyaml>=6.0.2",
  "torch>=2.3",
  "transformers>=4.44",
  "trl>=0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.2"]
vllm = ["vllm>=0.5"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

Create `.gitignore`:

```gitignore
__pycache__/
.pytest_cache/
.ruff_cache/
.venv/
*.egg-info/
runs/
outputs/
checkpoints/
*.pt
*.safetensors
```

Create `configs/default.yaml`:

```yaml
model:
  name_or_path: qwen/qwen3.5-9b
  torch_dtype: bf16
  max_prompt_length: 8192
lora:
  use_peft: true
  r: 32
  alpha: 64
  dropout: 0.05
rollout:
  temperature: 1.0
  top_p: 0.95
  max_episode_steps: 40
  run_dir: runs/dev
memory:
  tiers: [trajectory, tip, skill, tool]
  retrieve_top_k: 50
  teacher_memory_cap: 20
  score_threshold: 0.01
  maintenance_period: 30
  tier_priors:
    trajectory: 1.0
    tip: 1.0
    skill: 1.0
    tool: 1.0
training:
  learning_rate: 1.0e-5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  output_dir: runs/dev/checkpoints
tau3:
  env_name: retail
  adapter: mock
```

Create `src/tau3_retail_evolver/__init__.py`:

```python
__all__ = ["__version__"]

__version__ = "0.1.0"
```

Create the first `README.md`:

```markdown
# Tau3 Retail OPD-Evolver

This project trains a tau3-bench retail agent with OPD-Evolver style on-policy
distillation. Qwen3.5-9B is used as both the privileged teacher and the
student backbone; the deployable student is updated with LoRA adapters.

Primary references:

- Paper: OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation, arXiv 2606.17628v1
- Official repository: https://github.com/bingreeky/opd-evolver

The paper defines the algorithm. The official repository is used as an
engineering reference for rollout, memory scoring, dataset construction, and
executor OPD training patterns.
```

- [ ] **Step 4: Implement typed config loading**

Create `src/tau3_retail_evolver/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    torch_dtype: str
    max_prompt_length: int


@dataclass(frozen=True)
class LoraConfig:
    use_peft: bool
    r: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class RolloutConfig:
    temperature: float
    top_p: float
    max_episode_steps: int
    run_dir: str


@dataclass(frozen=True)
class MemoryConfig:
    tiers: tuple[str, ...]
    retrieve_top_k: int
    teacher_memory_cap: int
    score_threshold: float
    maintenance_period: int
    tier_priors: dict[str, float]


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    output_dir: str


@dataclass(frozen=True)
class Tau3Config:
    env_name: str
    adapter: str


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    lora: LoraConfig
    rollout: RolloutConfig
    memory: MemoryConfig
    training: TrainingConfig
    tau3: Tau3Config


def load_config(path: str | Path = "configs/default.yaml", overrides: Sequence[str] = ()) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} must be a mapping")
    merged = dict(raw)
    for override in overrides:
        _apply_override(merged, override)
    return _to_project_config(merged)


def _apply_override(config: dict[str, Any], override: str) -> None:
    key, sep, value = override.partition("=")
    if sep != "=" or not key:
        raise ValueError(f"override must use dotted.path=value form: {override}")
    cursor: dict[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"override path {key} crosses non-mapping value")
        cursor = next_value
    cursor[parts[-1]] = yaml.safe_load(value)


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing config key: {key}")
    return mapping[key]


def _to_project_config(raw: Mapping[str, Any]) -> ProjectConfig:
    model = _required(raw, "model")
    lora = _required(raw, "lora")
    rollout = _required(raw, "rollout")
    memory = _required(raw, "memory")
    training = _required(raw, "training")
    tau3 = _required(raw, "tau3")
    return ProjectConfig(
        model=ModelConfig(**model),
        lora=LoraConfig(**lora),
        rollout=RolloutConfig(**rollout),
        memory=MemoryConfig(
            tiers=tuple(memory["tiers"]),
            retrieve_top_k=int(memory["retrieve_top_k"]),
            teacher_memory_cap=int(memory["teacher_memory_cap"]),
            score_threshold=float(memory["score_threshold"]),
            maintenance_period=int(memory["maintenance_period"]),
            tier_priors={str(k): float(v) for k, v in memory["tier_priors"].items()},
        ),
        training=TrainingConfig(**training),
        tau3=Tau3Config(**tau3),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md configs/default.yaml src/tau3_retail_evolver/__init__.py src/tau3_retail_evolver/config.py tests/test_config.py
git commit -m "feat: add project config foundation"
```

---

### Task 2: Tau3 Retail Adapter Boundary

**Files:**
- Create: `src/tau3_retail_evolver/envs/__init__.py`
- Create: `src/tau3_retail_evolver/envs/base.py`
- Create: `src/tau3_retail_evolver/envs/mock_retail.py`
- Create: `src/tau3_retail_evolver/envs/tau3_retail.py`
- Create: `tests/test_envs.py`

**Interfaces:**
- Consumes: `RolloutConfig.max_episode_steps`
- Produces: `RetailTask`, `StepResult`, `RetailEnv`
- Produces: `MockRetailEnv(tasks: Sequence[RetailTask])`
- Produces: `make_tau3_retail_env(**kwargs) -> RetailEnv`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/test_envs.py`:

```python
from tau3_retail_evolver.envs.base import RetailTask
from tau3_retail_evolver.envs.mock_retail import MockRetailEnv
from tau3_retail_evolver.envs.tau3_retail import make_tau3_retail_env


def test_mock_retail_success_path():
    task = RetailTask(
        task_id="retail-001",
        instruction="Refund order A100",
        group="refund",
        metadata={"customer": "C1"},
        success_action="refund:A100",
    )
    env = MockRetailEnv([task])

    observation = env.reset(task)
    result = env.step("refund:A100")

    assert "Refund order" in observation
    assert result.done is True
    assert result.reward == 1.0
    assert env.success(result.info, result.reward, result.done) is True
    assert env.task_group(task) == "refund"
    assert env.metadata(task) == {"customer": "C1"}


def test_tau3_adapter_reports_missing_dependency():
    try:
        make_tau3_retail_env()
    except RuntimeError as exc:
        assert "tau3" in str(exc).lower()
        assert "retail" in str(exc).lower()
    else:
        assert False, "real tau3 adapter should require an installed tau3 retail harness"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_envs.py -v`

Expected: FAIL with missing `tau3_retail_evolver.envs`.

- [ ] **Step 3: Implement adapter protocol and mock environment**

Create `src/tau3_retail_evolver/envs/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RetailTask:
    task_id: str
    instruction: str
    group: str
    metadata: dict[str, Any] = field(default_factory=dict)
    success_action: str | None = None


@dataclass(frozen=True)
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class RetailEnv(Protocol):
    def reset(self, task: RetailTask) -> str:
        ...

    def step(self, action: str) -> StepResult:
        ...

    def task_group(self, task: RetailTask) -> str:
        ...

    def metadata(self, task: RetailTask) -> dict[str, Any]:
        ...

    def success(self, info: dict[str, Any], reward: float, done: bool) -> bool:
        ...
```

Create `src/tau3_retail_evolver/envs/mock_retail.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

from tau3_retail_evolver.envs.base import RetailTask, StepResult


class MockRetailEnv:
    def __init__(self, tasks: Sequence[RetailTask]) -> None:
        self._tasks = {task.task_id: task for task in tasks}
        self._current: RetailTask | None = None
        self._steps = 0

    def reset(self, task: RetailTask) -> str:
        self._current = task
        self._steps = 0
        return f"Task {task.task_id}: {task.instruction}"

    def step(self, action: str) -> StepResult:
        if self._current is None:
            raise RuntimeError("reset must be called before step")
        self._steps += 1
        matched = action.strip() == (self._current.success_action or "").strip()
        done = matched or self._steps >= 1
        reward = 1.0 if matched else 0.0
        status = "success" if matched else "failed"
        return StepResult(
            observation=f"{status}: received action {action}",
            reward=reward,
            done=done,
            info={"success": matched, "steps": self._steps, "task_id": self._current.task_id},
        )

    def task_group(self, task: RetailTask) -> str:
        return task.group

    def metadata(self, task: RetailTask) -> dict[str, object]:
        return dict(task.metadata)

    def success(self, info: dict[str, object], reward: float, done: bool) -> bool:
        return bool(done and reward > 0 and info.get("success") is True)
```

Create `src/tau3_retail_evolver/envs/tau3_retail.py`:

```python
from __future__ import annotations

from typing import Any


def make_tau3_retail_env(**kwargs: Any):
    try:
        import tau3_bench  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The real tau3 retail adapter requires the local tau3-bench retail "
            "harness. Install or expose tau3_bench, then wire its retail env here."
        ) from exc
    raise RuntimeError(
        "tau3_bench was imported, but this project needs the local retail runner "
        "constructor signature before the adapter can instantiate it."
    )
```

Create `src/tau3_retail_evolver/envs/__init__.py`:

```python
from tau3_retail_evolver.envs.base import RetailEnv, RetailTask, StepResult

__all__ = ["RetailEnv", "RetailTask", "StepResult"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_envs.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tau3_retail_evolver/envs tests/test_envs.py
git commit -m "feat: add tau3 retail environment boundary"
```

---

### Task 3: Four-Tier Memory Store And Retrieval

**Files:**
- Create: `src/tau3_retail_evolver/memory/__init__.py`
- Create: `src/tau3_retail_evolver/memory/types.py`
- Create: `src/tau3_retail_evolver/memory/store.py`
- Create: `src/tau3_retail_evolver/memory/retrieval.py`
- Create: `src/tau3_retail_evolver/memory/formatting.py`
- Create: `tests/test_memory.py`

**Interfaces:**
- Consumes: `MemoryConfig.tiers`, `retrieve_top_k`
- Produces: `MemoryItem`, `MemoryCandidate`, `MemorySelection`
- Produces: `InMemoryMemoryStore.add(item)`, `.by_tier(tier)`, `.delete(memory_id)`, `.merge(new_item, old_ids)`
- Produces: `retrieve_candidates(query, store, tiers, top_k) -> dict[str, list[MemoryCandidate]]`
- Produces: `format_selected_memories(selections) -> str`

- [ ] **Step 1: Write failing memory tests**

Create `tests/test_memory.py`:

```python
from tau3_retail_evolver.memory.formatting import format_selected_memories
from tau3_retail_evolver.memory.retrieval import retrieve_candidates
from tau3_retail_evolver.memory.store import InMemoryMemoryStore
from tau3_retail_evolver.memory.types import MemoryItem, MemorySelection


def test_retrieval_groups_candidates_by_tier_and_scores_text_overlap():
    store = InMemoryMemoryStore()
    store.add(MemoryItem(memory_id="tip-1", tier="tip", text="Refunds need order id"))
    store.add(MemoryItem(memory_id="skill-1", tier="skill", text="Escalate damaged item replacement"))

    candidates = retrieve_candidates("refund order request", store, ("tip", "skill"), top_k=1)

    assert candidates["tip"][0].memory_id == "tip-1"
    assert candidates["tip"][0].score > 0
    assert candidates["skill"][0].memory_id == "skill-1"


def test_format_selected_memories_is_stable():
    selection = MemorySelection(
        task_id="retail-001",
        selected=[
            MemoryItem(memory_id="tip-1", tier="tip", text="Ask for order id"),
            MemoryItem(memory_id="tool-1", tier="tool", text="refund:<order_id>"),
        ],
    )

    rendered = format_selected_memories(selection.selected)

    assert "[tip] tip-1: Ask for order id" in rendered
    assert "[tool] tool-1: refund:<order_id>" in rendered


def test_merge_replaces_old_items_with_new_item():
    store = InMemoryMemoryStore()
    store.add(MemoryItem(memory_id="tip-1", tier="tip", text="Old A"))
    store.add(MemoryItem(memory_id="tip-2", tier="tip", text="Old B"))

    store.merge(MemoryItem(memory_id="tip-merged", tier="tip", text="Merged"), ("tip-1", "tip-2"))

    assert [item.memory_id for item in store.by_tier("tip")] == ["tip-merged"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory.py -v`

Expected: FAIL with missing `tau3_retail_evolver.memory`.

- [ ] **Step 3: Implement memory types and store**

Create `src/tau3_retail_evolver/memory/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MemoryTier = str


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    tier: MemoryTier
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    tier: MemoryTier
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_item(self) -> MemoryItem:
        return MemoryItem(self.memory_id, self.tier, self.text, dict(self.metadata))


@dataclass(frozen=True)
class MemorySelection:
    task_id: str
    selected: list[MemoryItem]
```

Create `src/tau3_retail_evolver/memory/store.py`:

```python
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from tau3_retail_evolver.memory.types import MemoryItem


class InMemoryMemoryStore:
    def __init__(self, items: Iterable[MemoryItem] = ()) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._tiers: dict[str, list[str]] = defaultdict(list)
        for item in items:
            self.add(item)

    def add(self, item: MemoryItem) -> None:
        if item.memory_id in self._items:
            self.delete(item.memory_id)
        self._items[item.memory_id] = item
        self._tiers[item.tier].append(item.memory_id)

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def by_tier(self, tier: str) -> list[MemoryItem]:
        return [self._items[memory_id] for memory_id in self._tiers.get(tier, []) if memory_id in self._items]

    def delete(self, memory_id: str) -> None:
        item = self._items.pop(memory_id, None)
        if item is None:
            return
        self._tiers[item.tier] = [existing for existing in self._tiers[item.tier] if existing != memory_id]

    def merge(self, new_item: MemoryItem, old_ids: Iterable[str]) -> None:
        for memory_id in old_ids:
            self.delete(memory_id)
        self.add(new_item)
```

- [ ] **Step 4: Implement retrieval and formatting**

Create `src/tau3_retail_evolver/memory/retrieval.py`:

```python
from __future__ import annotations

import re
from collections.abc import Sequence

from tau3_retail_evolver.memory.store import InMemoryMemoryStore
from tau3_retail_evolver.memory.types import MemoryCandidate

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def retrieve_candidates(
    query: str,
    store: InMemoryMemoryStore,
    tiers: Sequence[str],
    top_k: int,
) -> dict[str, list[MemoryCandidate]]:
    return {tier: _top_for_tier(query, store, tier, top_k) for tier in tiers}


def _top_for_tier(query: str, store: InMemoryMemoryStore, tier: str, top_k: int) -> list[MemoryCandidate]:
    scored = [
        MemoryCandidate(item.memory_id, item.tier, item.text, _lexical_score(query, item.text), dict(item.metadata))
        for item in store.by_tier(tier)
    ]
    scored.sort(key=lambda candidate: (-candidate.score, candidate.memory_id))
    return scored[:top_k]


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(_TOKEN_RE.findall(query.lower()))
    text_tokens = set(_TOKEN_RE.findall(text.lower()))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens | text_tokens)
```

Create `src/tau3_retail_evolver/memory/formatting.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

from tau3_retail_evolver.memory.types import MemoryItem


def format_selected_memories(items: Sequence[MemoryItem]) -> str:
    if not items:
        return "No selected memories."
    return "\n".join(f"[{item.tier}] {item.memory_id}: {item.text}" for item in items)
```

Create `src/tau3_retail_evolver/memory/__init__.py`:

```python
from tau3_retail_evolver.memory.types import MemoryCandidate, MemoryItem, MemorySelection

__all__ = ["MemoryCandidate", "MemoryItem", "MemorySelection"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_memory.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tau3_retail_evolver/memory tests/test_memory.py
git commit -m "feat: add four-tier memory primitives"
```

---

### Task 4: Policy Boundary And Prompt Builders

**Files:**
- Create: `src/tau3_retail_evolver/models/__init__.py`
- Create: `src/tau3_retail_evolver/models/policy.py`
- Create: `src/tau3_retail_evolver/models/qwen.py`
- Create: `src/tau3_retail_evolver/fast_loop/prompts.py`
- Create: `tests/test_policy_prompts.py`

**Interfaces:**
- Consumes: `ModelConfig`, `LoraConfig`
- Produces: `GenerationRequest`, `GenerationResponse`, `Policy`
- Produces: `FakePolicy(scripted: Mapping[str, str])`
- Produces: `build_selection_prompt(...)`, `build_action_prompt(...)`, `build_writing_prompt(...)`, `build_maintenance_prompt(...)`
- Produces: `load_qwen_policy(model_config, lora_config, adapter_path=None) -> Policy`

- [ ] **Step 1: Write failing policy and prompt tests**

Create `tests/test_policy_prompts.py`:

```python
from tau3_retail_evolver.fast_loop.prompts import build_action_prompt, build_selection_prompt
from tau3_retail_evolver.memory.types import MemoryCandidate, MemoryItem
from tau3_retail_evolver.models.policy import FakePolicy, GenerationRequest


def test_fake_policy_returns_scripted_response_by_purpose():
    policy = FakePolicy({"act": "refund:A100"})

    response = policy.generate(GenerationRequest(prompt="Task", purpose="act"))

    assert response.text == "refund:A100"
    assert response.metadata["policy"] == "fake"


def test_selection_prompt_contains_candidates():
    candidates = {
        "tip": [MemoryCandidate("tip-1", "tip", "Ask for order id", 0.7)],
        "skill": [MemoryCandidate("skill-1", "skill", "Refund workflow", 0.4)],
    }

    prompt = build_selection_prompt("Refund order A100", candidates)

    assert "Return memory ids" in prompt
    assert "tip-1" in prompt
    assert "skill-1" in prompt


def test_action_prompt_contains_public_memory_context():
    prompt = build_action_prompt(
        task_instruction="Refund order A100",
        observation="Customer asks for refund",
        selected_memories=[MemoryItem("tip-1", "tip", "Ask for order id")],
        history=["observe customer", "ask order"],
    )

    assert "Refund order A100" in prompt
    assert "[tip] tip-1: Ask for order id" in prompt
    assert "ask order" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_policy_prompts.py -v`

Expected: FAIL with missing model or prompt modules.

- [ ] **Step 3: Implement policy boundary**

Create `src/tau3_retail_evolver/models/policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    purpose: str
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 512
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        ...


class FakePolicy:
    def __init__(self, scripted: Mapping[str, str]) -> None:
        self._scripted = dict(scripted)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        text = self._scripted.get(request.purpose, self._scripted.get("*", ""))
        return GenerationResponse(text=text, metadata={"policy": "fake", "purpose": request.purpose})
```

Create `src/tau3_retail_evolver/models/qwen.py`:

```python
from __future__ import annotations

from pathlib import Path

from tau3_retail_evolver.config import LoraConfig, ModelConfig
from tau3_retail_evolver.models.policy import GenerationRequest, GenerationResponse


class QwenPolicy:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
        )
        decoded = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return GenerationResponse(text=decoded, metadata={"policy": "qwen", "purpose": request.purpose})


def load_qwen_policy(
    model_config: ModelConfig,
    lora_config: LoraConfig,
    adapter_path: str | Path | None = None,
) -> QwenPolicy:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("QwenPolicy requires transformers and peft") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_config.name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_config.use_peft and adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    return QwenPolicy(model=model, tokenizer=tokenizer)
```

Create `src/tau3_retail_evolver/models/__init__.py`:

```python
from tau3_retail_evolver.models.policy import GenerationRequest, GenerationResponse, Policy

__all__ = ["GenerationRequest", "GenerationResponse", "Policy"]
```

- [ ] **Step 4: Implement prompt builders**

Create `src/tau3_retail_evolver/fast_loop/prompts.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence

from tau3_retail_evolver.memory.formatting import format_selected_memories
from tau3_retail_evolver.memory.types import MemoryCandidate, MemoryItem


def build_selection_prompt(task_instruction: str, candidates_by_tier: Mapping[str, Sequence[MemoryCandidate]]) -> str:
    lines = [
        "You are selecting useful experience for a tau3 retail task.",
        f"Task: {task_instruction}",
        "Return memory ids grouped by usefulness. Return only ids separated by commas.",
    ]
    for tier, candidates in candidates_by_tier.items():
        lines.append(f"Tier: {tier}")
        for candidate in candidates:
            lines.append(f"- {candidate.memory_id} score={candidate.score:.4f}: {candidate.text}")
    return "\n".join(lines)


def build_action_prompt(
    task_instruction: str,
    observation: str,
    selected_memories: Sequence[MemoryItem],
    history: Sequence[str],
) -> str:
    return "\n".join(
        [
            "You are acting in a tau3 retail environment.",
            f"Task: {task_instruction}",
            "Selected memories:",
            format_selected_memories(selected_memories),
            "History:",
            "\n".join(history) if history else "No previous actions.",
            f"Observation: {observation}",
            "Return the next retail action.",
        ]
    )


def build_writing_prompt(task_instruction: str, trajectory: Sequence[dict[str, object]], return_value: float) -> str:
    return "\n".join(
        [
            "Write reusable tau3 retail memories from this episode.",
            f"Task: {task_instruction}",
            f"Return: {return_value}",
            f"Trajectory: {list(trajectory)}",
            "Return JSON objects with tier, text, and optional metadata.",
        ]
    )


def build_maintenance_prompt(repository_summary: str, history_summary: str) -> str:
    return "\n".join(
        [
            "Maintain the tau3 retail memory repository.",
            "Available tools: lookup(query), merge(memory_ids), delete(memory_id).",
            f"Repository: {repository_summary}",
            f"History: {history_summary}",
            "Return tool calls as one command per line.",
        ]
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_policy_prompts.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tau3_retail_evolver/models src/tau3_retail_evolver/fast_loop/prompts.py tests/test_policy_prompts.py
git commit -m "feat: add policy and prompt boundaries"
```

---

### Task 5: Fast Loop Rollout Logging

**Files:**
- Create: `src/tau3_retail_evolver/fast_loop/__init__.py`
- Create: `src/tau3_retail_evolver/fast_loop/events.py`
- Create: `src/tau3_retail_evolver/fast_loop/runner.py`
- Create: `src/tau3_retail_evolver/io/__init__.py`
- Create: `src/tau3_retail_evolver/io/jsonl.py`
- Create: `tests/test_fast_loop.py`

**Interfaces:**
- Consumes: `RetailEnv`, `RetailTask`, `Policy`, `InMemoryMemoryStore`, `MemoryConfig`, `RolloutConfig`
- Produces: `RolloutEvent`, `RunResult`
- Produces: `run_fast_loop(tasks, env, policy, store, memory_config, rollout_config, output_dir) -> RunResult`
- Produces runtime JSONL at `<output_dir>/rollouts/events.jsonl`

- [ ] **Step 1: Write failing fast-loop test**

Create `tests/test_fast_loop.py`:

```python
from pathlib import Path

from tau3_retail_evolver.config import MemoryConfig, RolloutConfig
from tau3_retail_evolver.envs.base import RetailTask
from tau3_retail_evolver.envs.mock_retail import MockRetailEnv
from tau3_retail_evolver.fast_loop.runner import run_fast_loop
from tau3_retail_evolver.io.jsonl import read_jsonl
from tau3_retail_evolver.memory.store import InMemoryMemoryStore
from tau3_retail_evolver.memory.types import MemoryItem
from tau3_retail_evolver.models.policy import FakePolicy


def test_fast_loop_logs_candidates_selection_and_return(tmp_path: Path):
    task = RetailTask("retail-001", "Refund order A100", "refund", success_action="refund:A100")
    env = MockRetailEnv([task])
    store = InMemoryMemoryStore([MemoryItem("tip-1", "tip", "Refunds need order id")])
    policy = FakePolicy({"sel": "tip-1", "act": "refund:A100", "write": "tip|Refund success requires exact order id"})
    memory_config = MemoryConfig(
        tiers=("trajectory", "tip", "skill", "tool"),
        retrieve_top_k=50,
        teacher_memory_cap=20,
        score_threshold=0.01,
        maintenance_period=30,
        tier_priors={"trajectory": 1.0, "tip": 1.0, "skill": 1.0, "tool": 1.0},
    )
    rollout_config = RolloutConfig(temperature=1.0, top_p=0.95, max_episode_steps=40, run_dir=str(tmp_path))

    result = run_fast_loop([task], env, policy, store, memory_config, rollout_config, tmp_path)
    events = read_jsonl(tmp_path / "rollouts" / "events.jsonl")

    assert result.num_tasks == 1
    assert result.num_successes == 1
    assert events[0]["task_id"] == "retail-001"
    assert events[0]["selected_memory_ids"] == ["tip-1"]
    assert events[0]["return_value"] == 1.0
    assert events[0]["trajectory"][0]["action"] == "refund:A100"
    assert any(item.text == "Refund success requires exact order id" for item in store.by_tier("tip"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fast_loop.py -v`

Expected: FAIL with missing fast-loop runner or JSONL helpers.

- [ ] **Step 3: Implement JSONL helpers and event dataclasses**

Create `src/tau3_retail_evolver/io/jsonl.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
```

Create `src/tau3_retail_evolver/io/__init__.py`:

```python
from tau3_retail_evolver.io.jsonl import append_jsonl, read_jsonl, write_jsonl

__all__ = ["append_jsonl", "read_jsonl", "write_jsonl"]
```

Create `src/tau3_retail_evolver/fast_loop/events.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RolloutEvent:
    task_id: str
    task_group: str
    instruction: str
    candidates_by_tier: dict[str, list[dict[str, Any]]]
    selected_memory_ids: list[str]
    trajectory: list[dict[str, Any]]
    return_value: float
    success: bool
    written_memory_ids: list[str] = field(default_factory=list)
    maintenance_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    num_tasks: int
    num_successes: int
    output_path: str
```

- [ ] **Step 4: Implement fast-loop runner**

Create `src/tau3_retail_evolver/fast_loop/runner.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tau3_retail_evolver.config import MemoryConfig, RolloutConfig
from tau3_retail_evolver.envs.base import RetailEnv, RetailTask
from tau3_retail_evolver.fast_loop.events import RolloutEvent, RunResult
from tau3_retail_evolver.fast_loop.prompts import build_action_prompt, build_selection_prompt, build_writing_prompt
from tau3_retail_evolver.io.jsonl import append_jsonl
from tau3_retail_evolver.memory.retrieval import retrieve_candidates
from tau3_retail_evolver.memory.store import InMemoryMemoryStore
from tau3_retail_evolver.memory.types import MemoryCandidate, MemoryItem
from tau3_retail_evolver.models.policy import GenerationRequest, Policy


def run_fast_loop(
    tasks: Sequence[RetailTask],
    env: RetailEnv,
    policy: Policy,
    store: InMemoryMemoryStore,
    memory_config: MemoryConfig,
    rollout_config: RolloutConfig,
    output_dir: str | Path,
) -> RunResult:
    output_path = Path(output_dir) / "rollouts" / "events.jsonl"
    successes = 0
    for index, task in enumerate(tasks, start=1):
        event = _run_one_task(task, env, policy, store, memory_config, rollout_config)
        append_jsonl(output_path, event.to_dict())
        successes += int(event.success)
        if index % memory_config.maintenance_period == 0:
            append_jsonl(output_path, {**event.to_dict(), "maintenance_due": True})
    return RunResult(num_tasks=len(tasks), num_successes=successes, output_path=str(output_path))


def _run_one_task(
    task: RetailTask,
    env: RetailEnv,
    policy: Policy,
    store: InMemoryMemoryStore,
    memory_config: MemoryConfig,
    rollout_config: RolloutConfig,
) -> RolloutEvent:
    observation = env.reset(task)
    query = f"{task.instruction}\n{env.metadata(task)}\n{observation}"
    candidates_by_tier = retrieve_candidates(query, store, memory_config.tiers, memory_config.retrieve_top_k)
    selection_text = policy.generate(
        GenerationRequest(
            prompt=build_selection_prompt(task.instruction, candidates_by_tier),
            purpose="sel",
            temperature=rollout_config.temperature,
            top_p=rollout_config.top_p,
        )
    ).text
    selected = _parse_selection(selection_text, candidates_by_tier)
    history: list[str] = []
    trajectory: list[dict[str, object]] = []
    total_return = 0.0
    done = False
    for _ in range(rollout_config.max_episode_steps):
        action = policy.generate(
            GenerationRequest(
                prompt=build_action_prompt(task.instruction, observation, selected, history),
                purpose="act",
                temperature=rollout_config.temperature,
                top_p=rollout_config.top_p,
            )
        ).text.strip()
        step = env.step(action)
        trajectory.append({"observation": observation, "action": action, "reward": step.reward, "info": step.info})
        history.append(f"action={action} reward={step.reward}")
        total_return += step.reward
        observation = step.observation
        done = step.done
        if done:
            break
    success = env.success(trajectory[-1]["info"] if trajectory else {}, total_return, done)
    written_ids = _write_memories(task, policy, store, trajectory, total_return)
    return RolloutEvent(
        task_id=task.task_id,
        task_group=env.task_group(task),
        instruction=task.instruction,
        candidates_by_tier={tier: [candidate.__dict__ for candidate in candidates] for tier, candidates in candidates_by_tier.items()},
        selected_memory_ids=[item.memory_id for item in selected],
        trajectory=trajectory,
        return_value=total_return,
        success=success,
        written_memory_ids=written_ids,
    )


def _parse_selection(text: str, candidates_by_tier: dict[str, list[MemoryCandidate]]) -> list[MemoryItem]:
    wanted = {part.strip() for part in text.replace("\n", ",").split(",") if part.strip()}
    selected: list[MemoryItem] = []
    for candidates in candidates_by_tier.values():
        for candidate in candidates:
            if candidate.memory_id in wanted:
                selected.append(candidate.to_item())
    return selected


def _write_memories(
    task: RetailTask,
    policy: Policy,
    store: InMemoryMemoryStore,
    trajectory: list[dict[str, object]],
    return_value: float,
) -> list[str]:
    text = policy.generate(
        GenerationRequest(
            prompt=build_writing_prompt(task.instruction, trajectory, return_value),
            purpose="write",
        )
    ).text.strip()
    if not text:
        return []
    written_ids: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        parts = [part.strip() for part in line.split("|", maxsplit=1)]
        if len(parts) != 2:
            continue
        tier, memory_text = parts
        memory_id = f"{task.task_id}-{tier}-{idx}"
        store.add(MemoryItem(memory_id=memory_id, tier=tier, text=memory_text, metadata={"source_task": task.task_id}))
        written_ids.append(memory_id)
    return written_ids
```

Create `src/tau3_retail_evolver/fast_loop/__init__.py`:

```python
from tau3_retail_evolver.fast_loop.runner import run_fast_loop

__all__ = ["run_fast_loop"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_fast_loop.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tau3_retail_evolver/fast_loop src/tau3_retail_evolver/io tests/test_fast_loop.py
git commit -m "feat: add fast loop rollout logging"
```

---

### Task 6: Outcome-Calibrated Attribution

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/__init__.py`
- Create: `src/tau3_retail_evolver/slow_loop/attribution.py`
- Create: `tests/test_attribution.py`

**Interfaces:**
- Consumes: rollout events from `fast_loop/events.py`
- Produces: `MemoryScore(memory_id, tier, value, confidence, selected_count, retrieved_count)`
- Produces: `compute_memory_scores(events, tier_priors, min_score=0.01) -> dict[str, MemoryScore]`

- [ ] **Step 1: Write failing attribution tests**

Create `tests/test_attribution.py`:

```python
from tau3_retail_evolver.slow_loop.attribution import compute_memory_scores


def test_selected_memory_gets_positive_score_when_selected_returns_are_higher():
    events = [
        {
            "task_group": "refund",
            "return_value": 1.0,
            "candidates_by_tier": {"tip": [{"memory_id": "tip-1", "tier": "tip", "text": "refund", "score": 0.5}]},
            "selected_memory_ids": ["tip-1"],
        },
        {
            "task_group": "refund",
            "return_value": 0.0,
            "candidates_by_tier": {"tip": [{"memory_id": "tip-1", "tier": "tip", "text": "refund", "score": 0.5}]},
            "selected_memory_ids": [],
        },
    ]

    scores = compute_memory_scores(events, tier_priors={"tip": 1.0}, min_score=0.01)

    assert "tip-1" in scores
    assert scores["tip-1"].value > 0
    assert scores["tip-1"].confidence > 0
    assert scores["tip-1"].selected_count == 1
    assert scores["tip-1"].retrieved_count == 2


def test_low_score_is_filtered():
    events = [
        {
            "task_group": "refund",
            "return_value": 0.5,
            "candidates_by_tier": {"tip": [{"memory_id": "tip-1", "tier": "tip", "text": "refund", "score": 0.5}]},
            "selected_memory_ids": ["tip-1"],
        }
    ]

    scores = compute_memory_scores(events, tier_priors={"tip": 1.0}, min_score=0.01)

    assert scores == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_attribution.py -v`

Expected: FAIL with missing `slow_loop`.

- [ ] **Step 3: Implement attribution calculation**

Create `src/tau3_retail_evolver/slow_loop/attribution.py`:

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class MemoryScore:
    memory_id: str
    tier: str
    value: float
    confidence: float
    selected_count: int
    retrieved_count: int


def compute_memory_scores(
    events: list[dict[str, Any]],
    tier_priors: dict[str, float],
    min_score: float = 0.01,
) -> dict[str, MemoryScore]:
    selected_returns: dict[tuple[str, str], list[float]] = defaultdict(list)
    unselected_returns: dict[tuple[str, str], list[float]] = defaultdict(list)
    memory_tiers: dict[str, str] = {}
    retrieved_counts: dict[str, int] = defaultdict(int)
    selected_counts: dict[str, int] = defaultdict(int)
    for event in events:
        group = str(event["task_group"])
        return_value = float(event["return_value"])
        selected_ids = set(event.get("selected_memory_ids", []))
        for candidates in event.get("candidates_by_tier", {}).values():
            for candidate in candidates:
                memory_id = str(candidate["memory_id"])
                tier = str(candidate["tier"])
                memory_tiers[memory_id] = tier
                retrieved_counts[memory_id] += 1
                key = (group, memory_id)
                if memory_id in selected_ids:
                    selected_returns[key].append(return_value)
                    selected_counts[memory_id] += 1
                else:
                    unselected_returns[key].append(return_value)
    scores: dict[str, MemoryScore] = {}
    for memory_id, tier in memory_tiers.items():
        attribution = 0.0
        groups = {group for group, mid in selected_returns if mid == memory_id} | {
            group for group, mid in unselected_returns if mid == memory_id
        }
        for group in groups:
            pos = selected_returns.get((group, memory_id), [])
            neg = unselected_returns.get((group, memory_id), [])
            if not pos or not neg:
                continue
            rho = len(pos) / (len(pos) + len(neg))
            attribution += rho * (_mean(pos) - _mean(neg))
        confidence = 1.0 - 1.0 / sqrt(1.0 + selected_counts[memory_id])
        value = tier_priors.get(tier, 1.0) * confidence * attribution
        if value >= min_score:
            scores[memory_id] = MemoryScore(
                memory_id=memory_id,
                tier=tier,
                value=value,
                confidence=confidence,
                selected_count=selected_counts[memory_id],
                retrieved_count=retrieved_counts[memory_id],
            )
    return scores


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
```

Create `src/tau3_retail_evolver/slow_loop/__init__.py`:

```python
from tau3_retail_evolver.slow_loop.attribution import MemoryScore, compute_memory_scores

__all__ = ["MemoryScore", "compute_memory_scores"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_attribution.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tau3_retail_evolver/slow_loop tests/test_attribution.py
git commit -m "feat: add outcome calibrated attribution"
```

---

### Task 7: Slow-Loop OPD Example Construction

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/examples.py`
- Create: `tests/test_examples.py`

**Interfaces:**
- Consumes: rollout events and `MemoryScore`
- Produces: `OPDExample(kind, public_input, privileged_input, student_prefix_source, metadata)`
- Produces: `build_opd_examples(events, scores, teacher_memory_cap) -> list[OPDExample]`

- [ ] **Step 1: Write failing example tests**

Create `tests/test_examples.py`:

```python
from tau3_retail_evolver.slow_loop.attribution import MemoryScore
from tau3_retail_evolver.slow_loop.examples import build_opd_examples


def test_builds_selection_action_writing_examples():
    events = [
        {
            "task_id": "retail-001",
            "task_group": "refund",
            "instruction": "Refund order A100",
            "return_value": 1.0,
            "candidates_by_tier": {"tip": [{"memory_id": "tip-1", "tier": "tip", "text": "Ask order id", "score": 0.7}]},
            "selected_memory_ids": ["tip-1"],
            "trajectory": [{"observation": "start", "action": "refund:A100", "reward": 1.0, "info": {"success": True}}],
            "written_memory_ids": ["retail-001-tip-1"],
        }
    ]
    scores = {
        "tip-1": MemoryScore("tip-1", "tip", 0.3, 0.2, 1, 2),
        "retail-001-tip-1": MemoryScore("retail-001-tip-1", "tip", 0.4, 0.2, 1, 2),
    }

    examples = build_opd_examples(events, scores, teacher_memory_cap=20)

    kinds = {example.kind for example in examples}
    assert {"sel", "act", "write"} <= kinds
    selection = next(example for example in examples if example.kind == "sel")
    assert "Refund order A100" in selection.public_input
    assert "V(tip-1)=0.300000" in selection.privileged_input
    action = next(example for example in examples if example.kind == "act")
    assert "successful trajectory" in action.privileged_input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_examples.py -v`

Expected: FAIL with missing `slow_loop.examples`.

- [ ] **Step 3: Implement OPD example construction**

Create `src/tau3_retail_evolver/slow_loop/examples.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tau3_retail_evolver.slow_loop.attribution import MemoryScore


@dataclass(frozen=True)
class OPDExample:
    kind: str
    public_input: str
    privileged_input: str
    student_prefix_source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_opd_examples(
    events: list[dict[str, Any]],
    scores: dict[str, MemoryScore],
    teacher_memory_cap: int,
) -> list[OPDExample]:
    examples: list[OPDExample] = []
    successful_by_group = _successful_trajectories_by_group(events)
    for event in events:
        examples.append(_selection_example(event, scores, teacher_memory_cap))
        examples.append(_action_example(event, scores, successful_by_group))
        if event.get("written_memory_ids"):
            examples.append(_writing_example(event, scores))
        if event.get("maintenance_actions"):
            examples.append(_maintenance_example(event, scores))
    return examples


def _selection_example(event: dict[str, Any], scores: dict[str, MemoryScore], teacher_memory_cap: int) -> OPDExample:
    candidates = [
        candidate
        for tier_candidates in event.get("candidates_by_tier", {}).values()
        for candidate in tier_candidates
    ][:teacher_memory_cap]
    public = f"Task: {event['instruction']}\nCandidates: {candidates}"
    scored = [_score_line(str(candidate["memory_id"]), scores) for candidate in candidates]
    privileged = "Candidate memory values:\n" + "\n".join(scored)
    return OPDExample("sel", public, privileged, "student_selection", {"task_id": event["task_id"]})


def _action_example(
    event: dict[str, Any],
    scores: dict[str, MemoryScore],
    successful_by_group: dict[str, list[dict[str, Any]]],
) -> OPDExample:
    selected = event.get("selected_memory_ids", [])
    valuable = [memory_id for memory_id in selected if memory_id in scores and scores[memory_id].value > 0]
    demo = successful_by_group.get(str(event["task_group"]), [event])[0]
    public = f"Task: {event['instruction']}\nPublic trajectory prefix: {event.get('trajectory', [])}"
    privileged = f"Valuable selected memories: {valuable}\nsuccessful trajectory: {demo.get('trajectory', [])}"
    return OPDExample("act", public, privileged, "student_action", {"task_id": event["task_id"]})


def _writing_example(event: dict[str, Any], scores: dict[str, MemoryScore]) -> OPDExample:
    written = event.get("written_memory_ids", [])
    public = f"Task: {event['instruction']}\nTrajectory: {event.get('trajectory', [])}\nReturn: {event.get('return_value', 0.0)}"
    privileged = "Written memory future values:\n" + "\n".join(_score_line(str(memory_id), scores) for memory_id in written)
    return OPDExample("write", public, privileged, "student_write", {"task_id": event["task_id"]})


def _maintenance_example(event: dict[str, Any], scores: dict[str, MemoryScore]) -> OPDExample:
    public = f"Repository maintenance state for task {event['task_id']}"
    privileged = f"Diagnostics: {[score.__dict__ for score in scores.values()]}"
    return OPDExample("maint", public, privileged, "student_maintenance", {"task_id": event["task_id"]})


def _successful_trajectories_by_group(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("success"):
            grouped.setdefault(str(event["task_group"]), []).append(event)
    return grouped


def _score_line(memory_id: str, scores: dict[str, MemoryScore]) -> str:
    score = scores.get(memory_id)
    value = 0.0 if score is None else score.value
    return f"V({memory_id})={value:.6f}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_examples.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tau3_retail_evolver/slow_loop/examples.py tests/test_examples.py
git commit -m "feat: build slow loop opd examples"
```

---

### Task 8: Token-Level KL Loss And LoRA Training Entry Point

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/loss.py`
- Create: `src/tau3_retail_evolver/slow_loop/train.py`
- Create: `tests/test_loss.py`
- Create: `tests/test_train_dry_run.py`

**Interfaces:**
- Consumes: OPD examples from `slow_loop/examples.py`
- Produces: `token_kl_loss(student_logits, teacher_logits, attention_mask=None) -> torch.Tensor`
- Produces: `TrainingSummary(output_dir, num_examples, dry_run)`
- Produces: `run_lora_opd_training(config, examples_path, output_dir=None, dry_run=False) -> TrainingSummary`

- [ ] **Step 1: Write failing loss and dry-run tests**

Create `tests/test_loss.py`:

```python
import torch

from tau3_retail_evolver.slow_loop.loss import token_kl_loss


def test_token_kl_loss_is_zero_for_identical_logits():
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    loss = token_kl_loss(logits, logits)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_token_kl_loss_respects_attention_mask():
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    teacher = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
    mask = torch.tensor([[0, 1]])

    loss = token_kl_loss(student, teacher, attention_mask=mask)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
```

Create `tests/test_train_dry_run.py`:

```python
from pathlib import Path

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.io.jsonl import write_jsonl
from tau3_retail_evolver.slow_loop.train import run_lora_opd_training


def test_training_dry_run_reads_examples_and_reports_lora_defaults(tmp_path: Path):
    examples_path = tmp_path / "examples.jsonl"
    write_jsonl(
        examples_path,
        [
            {
                "kind": "act",
                "public_input": "Task",
                "privileged_input": "Teacher",
                "student_prefix_source": "student_action",
                "metadata": {"task_id": "retail-001"},
            }
        ],
    )
    cfg = load_config("configs/default.yaml")

    summary = run_lora_opd_training(cfg, examples_path, output_dir=tmp_path / "adapter", dry_run=True)

    assert summary.num_examples == 1
    assert summary.dry_run is True
    assert summary.lora_r == 32
    assert summary.lora_alpha == 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loss.py tests/test_train_dry_run.py -v`

Expected: FAIL with missing `loss.py` and `train.py`.

- [ ] **Step 3: Implement KL loss**

Create `src/tau3_retail_evolver/slow_loop/loss.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def token_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if attention_mask is None:
        return kl.mean()
    mask = attention_mask.to(dtype=kl.dtype, device=kl.device)
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom
```

- [ ] **Step 4: Implement training dry run and dependency-gated trainer**

Create `src/tau3_retail_evolver/slow_loop/train.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tau3_retail_evolver.config import ProjectConfig
from tau3_retail_evolver.io.jsonl import read_jsonl


@dataclass(frozen=True)
class TrainingSummary:
    output_dir: str
    num_examples: int
    dry_run: bool
    lora_r: int
    lora_alpha: int


def run_lora_opd_training(
    config: ProjectConfig,
    examples_path: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> TrainingSummary:
    examples = read_jsonl(examples_path)
    target_dir = Path(output_dir or config.training.output_dir)
    if dry_run:
        return TrainingSummary(str(target_dir), len(examples), True, config.lora.r, config.lora.alpha)
    if not config.lora.use_peft:
        raise ValueError("OPD-Evolver project requires LoRA/PEFT training; set lora.use_peft=true")
    return _run_transformers_training(config, examples, target_dir)


def _run_transformers_training(config: ProjectConfig, examples: list[dict[str, object]], output_dir: Path) -> TrainingSummary:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install torch, transformers, peft, and accelerate to run non-dry OPD training") from exc
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    tokenized = [_tokenize_example(tokenizer, example, config.model.max_prompt_length) for example in examples]
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        bf16=config.model.torch_dtype == "bf16",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return TrainingSummary(str(output_dir), len(examples), False, config.lora.r, config.lora.alpha)


def _tokenize_example(tokenizer, example: dict[str, object], max_length: int) -> dict[str, object]:
    text = f"{example['public_input']}\n\nPrivileged teacher context:\n{example['privileged_input']}"
    encoded = tokenizer(text, truncation=True, max_length=max_length)
    encoded["labels"] = list(encoded["input_ids"])
    return encoded
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_loss.py tests/test_train_dry_run.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tau3_retail_evolver/slow_loop/loss.py src/tau3_retail_evolver/slow_loop/train.py tests/test_loss.py tests/test_train_dry_run.py
git commit -m "feat: add opd loss and lora training entry point"
```

---

### Task 9: CLI Pipeline And End-To-End Smoke Test

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/run_rollout.py`
- Create: `scripts/build_attribution.py`
- Create: `scripts/build_opd_dataset.py`
- Create: `scripts/train_lora.py`
- Create: `scripts/run_iteration.py`
- Create: `tests/test_cli_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `load_config`, `run_fast_loop`, `compute_memory_scores`, `build_opd_examples`, `run_lora_opd_training`
- Produces CLI commands:
  - `python -m scripts.run_rollout --config configs/default.yaml --run-dir runs/dev --mock`
  - `python -m scripts.build_attribution --config configs/default.yaml --events runs/dev/rollouts/events.jsonl --output runs/dev/attribution/scores.jsonl`
  - `python -m scripts.build_opd_dataset --events runs/dev/rollouts/events.jsonl --scores runs/dev/attribution/scores.jsonl --output runs/dev/opd_examples/examples.jsonl`
  - `python -m scripts.train_lora --config configs/default.yaml --examples runs/dev/opd_examples/examples.jsonl --dry-run`
  - `python -m scripts.run_iteration --config configs/default.yaml --run-dir runs/dev --mock --dry-run-train`

- [ ] **Step 1: Write failing CLI smoke test**

Create `tests/test_cli_smoke.py`:

```python
import subprocess
import sys
from pathlib import Path


def test_mock_iteration_cli_creates_expected_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_iteration",
        "--config",
        "configs/default.yaml",
        "--run-dir",
        str(run_dir),
        "--mock",
        "--dry-run-train",
    ]

    result = subprocess.run(cmd, check=False, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert (run_dir / "rollouts" / "events.jsonl").exists()
    assert (run_dir / "attribution" / "scores.jsonl").exists()
    assert (run_dir / "opd_examples" / "examples.jsonl").exists()
    assert "dry_run=True" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_smoke.py -v`

Expected: FAIL with missing `scripts.run_iteration`.

- [ ] **Step 3: Implement attribution and dataset CLIs**

Create `scripts/__init__.py`:

```python
```

Create `scripts/build_attribution.py`:

```python
from __future__ import annotations

import argparse

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.io.jsonl import read_jsonl, write_jsonl
from tau3_retail_evolver.slow_loop.attribution import compute_memory_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    events = read_jsonl(args.events)
    scores = compute_memory_scores(events, cfg.memory.tier_priors, cfg.memory.score_threshold)
    write_jsonl(args.output, [score.__dict__ for score in scores.values()])
    print(f"wrote {len(scores)} scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/build_opd_dataset.py`:

```python
from __future__ import annotations

import argparse

from tau3_retail_evolver.io.jsonl import read_jsonl, write_jsonl
from tau3_retail_evolver.slow_loop.attribution import MemoryScore
from tau3_retail_evolver.slow_loop.examples import build_opd_examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-memory-cap", type=int, default=20)
    args = parser.parse_args(argv)
    events = read_jsonl(args.events)
    score_rows = read_jsonl(args.scores)
    scores = {row["memory_id"]: MemoryScore(**row) for row in score_rows}
    examples = build_opd_examples(events, scores, args.teacher_memory_cap)
    write_jsonl(args.output, [example.to_dict() for example in examples])
    print(f"wrote {len(examples)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement rollout and train CLIs**

Create `scripts/run_rollout.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.base import RetailTask
from tau3_retail_evolver.envs.mock_retail import MockRetailEnv
from tau3_retail_evolver.fast_loop.runner import run_fast_loop
from tau3_retail_evolver.memory.store import InMemoryMemoryStore
from tau3_retail_evolver.memory.types import MemoryItem
from tau3_retail_evolver.models.policy import FakePolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    if not args.mock:
        raise RuntimeError("Only --mock is wired in the first implementation; real tau3 retail uses the adapter boundary.")
    cfg = load_config(args.config, overrides=(f"rollout.run_dir={args.run_dir}",))
    tasks = [RetailTask("retail-001", "Refund order A100", "refund", success_action="refund:A100")]
    env = MockRetailEnv(tasks)
    store = InMemoryMemoryStore([MemoryItem("tip-1", "tip", "Refunds need exact order id")])
    policy = FakePolicy({"sel": "tip-1", "act": "refund:A100", "write": "tip|Refund success requires exact order id"})
    result = run_fast_loop(tasks, env, policy, store, cfg.memory, cfg.rollout, Path(args.run_dir))
    print(f"wrote rollouts to {result.output_path}; successes={result.num_successes}/{result.num_tasks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/train_lora.py`:

```python
from __future__ import annotations

import argparse

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.slow_loop.train import run_lora_opd_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    summary = run_lora_opd_training(cfg, args.examples, output_dir=args.output_dir, dry_run=args.dry_run)
    print(
        f"training output_dir={summary.output_dir} num_examples={summary.num_examples} "
        f"dry_run={summary.dry_run} lora_r={summary.lora_r} lora_alpha={summary.lora_alpha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Implement one-iteration CLI**

Create `scripts/run_iteration.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import build_attribution, build_opd_dataset, run_rollout, train_lora


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run-train", action="store_true")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    rollout_args = ["--config", args.config, "--run-dir", str(run_dir)]
    if args.mock:
        rollout_args.append("--mock")
    run_rollout.main(rollout_args)
    events = run_dir / "rollouts" / "events.jsonl"
    scores = run_dir / "attribution" / "scores.jsonl"
    examples = run_dir / "opd_examples" / "examples.jsonl"
    build_attribution.main(["--config", args.config, "--events", str(events), "--output", str(scores)])
    build_opd_dataset.main(["--events", str(events), "--scores", str(scores), "--output", str(examples)])
    train_args = ["--config", args.config, "--examples", str(examples), "--output-dir", str(run_dir / "checkpoints")]
    if args.dry_run_train:
        train_args.append("--dry-run")
    train_lora.main(train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Update README with runnable commands**

Append to `README.md`:

```markdown
## Local smoke test

```bash
python -m scripts.run_iteration \
  --config configs/default.yaml \
  --run-dir runs/dev \
  --mock \
  --dry-run-train
```

This creates:

- `runs/dev/rollouts/events.jsonl`
- `runs/dev/attribution/scores.jsonl`
- `runs/dev/opd_examples/examples.jsonl`

## Real tau3 retail integration

The real retail harness should be connected behind
`tau3_retail_evolver.envs.tau3_retail.make_tau3_retail_env`. The OPD code only
depends on the adapter protocol, so rollout, attribution, example construction,
and LoRA training do not need to change when the real tau3 retail package is
available.
```

- [ ] **Step 7: Run smoke test to verify it passes**

Run: `pytest tests/test_cli_smoke.py -v`

Expected: PASS.

- [ ] **Step 8: Run all tests**

Run: `pytest -v`

Expected: PASS for all unit and smoke tests.

- [ ] **Step 9: Commit**

```bash
git add scripts README.md tests/test_cli_smoke.py
git commit -m "feat: add opd pipeline cli"
```

---

### Task 10: Documentation Cross-Reference And Final Verification

**Files:**
- Modify: `README.md`
- Create: `docs/opd_evolver_mapping.md`

**Interfaces:**
- Consumes: design doc at `docs/superpowers/specs/2026-07-09-tau3-retail-opd-evolver-design.md`
- Produces: documentation mapping project modules to OPD-Evolver paper sections and the official repository's public demo scope.

- [ ] **Step 1: Write the mapping document**

Create `docs/opd_evolver_mapping.md`:

```markdown
# OPD-Evolver Mapping For Tau3 Retail

## References

- Paper: OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation, arXiv 2606.17628v1.
- Official repository: https://github.com/bingreeky/opd-evolver.

## Paper-To-Project Mapping

- Fast loop Algorithm 1 lines 1-12 maps to `tau3_retail_evolver.fast_loop.runner`.
- Four memory tiers map to `tau3_retail_evolver.memory`.
- Outcome-calibrated attribution maps to `tau3_retail_evolver.slow_loop.attribution`.
- Unified hindsight self-distillation maps to `tau3_retail_evolver.slow_loop.examples`, `loss`, and `train`.
- Tau3 retail-specific environment behavior is isolated in `tau3_retail_evolver.envs`.

## Official Repository Reference Scope

The official repository is used as an engineering reference for Python project
layout, rollout artifacts, memory scoring/data construction, and executor OPD
training entry points. The repository README states that the published training
path is an executor OPD demonstration rather than the complete selector,
reflection, and full schedule stack. This project therefore implements the full
four-decision lifecycle from the paper while keeping the command-line pipeline
and JSONL artifact style close to the released repository.
```

- [ ] **Step 2: Link mapping doc from README**

Add this paragraph to `README.md` after the primary references:

```markdown
See `docs/opd_evolver_mapping.md` for the exact mapping between this tau3
retail implementation, the OPD-Evolver paper, and the public official
repository.
```

- [ ] **Step 3: Run documentation checks**

Run: `rg -n "OPD-Evolver|2606.17628|bingreeky/opd-evolver|Qwen3.5-9B|lora_r|lora_alpha" README.md docs`

Expected: output includes README, the design spec, and `docs/opd_evolver_mapping.md`.

- [ ] **Step 4: Run all tests**

Run: `pytest -v`

Expected: PASS.

- [ ] **Step 5: Check git status**

Run: `git status --short --branch`

Expected: only intentional documentation files and implementation files are modified before the commit.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/opd_evolver_mapping.md
git commit -m "docs: map tau3 retail implementation to opd evolver"
```

---

## Self-Review

- Spec coverage: The plan covers project setup, tau3 retail adapter, four-tier memory, fast loop, attribution, slow-loop examples, OPD KL loss, LoRA dry-run/non-dry training entry, CLI pipeline, and OPD-Evolver references.
- Scope: The plan produces working, testable software without requiring tau3-bench or Qwen3.5-9B downloads. Real tau3 retail integration remains behind the adapter boundary defined in the approved design.
- Type consistency: `MemoryItem`, `MemoryCandidate`, `MemoryScore`, `OPDExample`, `GenerationRequest`, and `ProjectConfig` are introduced before later tasks consume them.
- Verification: Every task includes a failing test, implementation, passing test command, and commit step.
