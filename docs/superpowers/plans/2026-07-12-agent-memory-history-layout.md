# Agent Memory History Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store one continuously evolving Memory per Agent namespace under `history/agents/<agent_id>/memory/`, independently of `run_id`, while keeping evaluation streaming Memory quarantined.

**Architecture:** Add a validated `memory.agent_id` configuration value and a single path resolver whose default project root is derived from the installed source file rather than the process working directory. A small factory is the only production entry point for opening training Memory, so future fast/slow loops cannot accidentally construct `runs/<run_id>/memory`. Existing `MemoryRepository` persistence, snapshots, embeddings, and lifecycle operations remain unchanged.

**Tech Stack:** Python 3.12, pathlib, Pydantic 2, PyYAML, pytest, existing JSON Memory Repository.

## Global Constraints

- Default `agent_id` is exactly `retail`.
- Training Memory root is exactly `history/agents/<agent_id>/memory/` beneath the project root.
- `run_id` must not be accepted by the training Memory path or factory API.
- `agent_id` and evaluation `run_id` allow only ASCII letters, digits, `-`, and `_`.
- `/history/` is ignored by Git; no `.gitkeep` or generated Memory is committed.
- The same Agent namespace accumulates experience across all training rounds.
- Different Agent namespaces never share or implicitly merge Memory.
- Streaming evaluation Memory is created only under `history/evaluations/<run_id>/<agent_id>/quarantine/`.
- Unit tests write only beneath pytest `tmp_path`, never the real project `history/`.

---

### Task 1: Validate Agent Namespace and Resolve History Paths

**Files:**
- Create: `src/tau3_retail_evolver/memory/paths.py`
- Modify: `src/tau3_retail_evolver/config.py`
- Modify: `configs/default.yaml`
- Modify: `.gitignore`
- Modify: `tests/unit/test_config.py`
- Create: `tests/unit/memory/test_paths.py`

**Interfaces:**
- Produces: `MemoryConfig.agent_id: str`
- Produces: `project_root() -> Path`
- Produces: `training_memory_root(agent_id: str, *, root: Path | None = None) -> Path`
- Produces: `evaluation_quarantine_root(run_id: str, agent_id: str, *, root: Path | None = None) -> Path`

- [ ] **Step 1: Write failing configuration tests**

Add the default assertion to `test_default_config_has_the_required_retail_environment`:

```python
assert config.memory.agent_id == "retail"
```

Add explicit validation tests:

```python
@pytest.mark.parametrize("agent_id", ("", ".", "..", "retail/other", r"retail\\other", "零售"))
def test_memory_config_rejects_unsafe_agent_id(agent_id: str) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        ProjectConfig.model_validate(
            {
                "tau2": {
                    "repo_path": "external/tau2-bench",
                    "domain": "retail",
                    "train_split": "train",
                    "eval_split": "test",
                    "user_llm": "test-user",
                },
                "model": {"base_model": "Qwen/Qwen3.5-9B"},
                "memory": {"agent_id": agent_id},
            }
        )
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/test_config.py -q --basetemp=.pytest-tmp/history-config-red
```

Expected: FAIL because `MemoryConfig` has no `agent_id` field or rejects the new YAML key.

- [ ] **Step 3: Implement configuration validation**

In `src/tau3_retail_evolver/config.py`, add `_AGENT_ID` beside the existing environment-variable regex and replace `MemoryConfig` with:

```python
_AGENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class MemoryConfig(_ConfigModel):
    agent_id: str = "retail"
    tiers: tuple[Literal["trajectory", "tip", "skill", "tool"], ...] = (
        "trajectory",
        "tip",
        "skill",
        "tool",
    )
    retrieve_top_k: int = 50
    teacher_memory_cap: int = 20
    score_threshold: float = 0.01
    maintenance_period: int = 30
    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cuda"
    embedding_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    embedding_max_length: int = Field(default=2048, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)
    embedding_cache: bool = True

    @field_validator("agent_id")
    @classmethod
    def agent_id_must_be_a_safe_slug(cls, value: str) -> str:
        if not _AGENT_ID.fullmatch(value) or value in {".", ".."}:
            raise ValueError("agent_id must contain only ASCII letters, digits, '-' or '_'")
        return value
```

Add this key to `configs/default.yaml`:

```yaml
memory:
  agent_id: retail
```

Update the exact `config.memory.model_dump()` expectation in `tests/unit/test_config.py` to include `"agent_id": "retail"`.

- [ ] **Step 4: Write failing path tests**

Create `tests/unit/memory/test_paths.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from tau3_retail_evolver.memory.paths import (
    evaluation_quarantine_root,
    project_root,
    training_memory_root,
)


def test_training_memory_is_stable_across_run_ids(tmp_path: Path) -> None:
    first_run_id = "iteration-0001"
    second_run_id = "iteration-0002"

    first = training_memory_root("retail", root=tmp_path)
    second = training_memory_root("retail", root=tmp_path)

    assert first_run_id != second_run_id
    assert first == second == tmp_path.resolve() / "history" / "agents" / "retail" / "memory"


def test_agent_namespaces_are_isolated(tmp_path: Path) -> None:
    retail = training_memory_root("retail", root=tmp_path)
    airline = training_memory_root("airline", root=tmp_path)

    assert retail != airline
    assert retail.parent.name == "retail"
    assert airline.parent.name == "airline"


def test_training_path_does_not_depend_on_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert training_memory_root("retail", root=tmp_path) == (
        tmp_path.resolve() / "history" / "agents" / "retail" / "memory"
    )


def test_default_project_root_does_not_depend_on_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(tmp_path)

    assert project_root() == expected


@pytest.mark.parametrize("agent_id", ("", ".", "..", "retail/other", r"retail\\other"))
def test_path_resolver_rejects_unsafe_agent_id(tmp_path: Path, agent_id: str) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        training_memory_root(agent_id, root=tmp_path)


def test_streaming_evaluation_uses_quarantine(tmp_path: Path) -> None:
    assert evaluation_quarantine_root(
        "eval-0001", "retail", root=tmp_path
    ) == (
        tmp_path.resolve()
        / "history"
        / "evaluations"
        / "eval-0001"
        / "retail"
        / "quarantine"
    )
```

- [ ] **Step 5: Run path tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/memory/test_paths.py -q --basetemp=.pytest-tmp/history-paths-red
```

Expected: collection ERROR because `tau3_retail_evolver.memory.paths` does not exist.

- [ ] **Step 6: Implement the centralized resolver**

Create `src/tau3_retail_evolver/memory/paths.py`:

```python
from __future__ import annotations

from pathlib import Path
import re


_SAFE_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def training_memory_root(agent_id: str, *, root: Path | None = None) -> Path:
    namespace = _validated_slug(agent_id, field="agent_id")
    base = (root or project_root()).resolve()
    return base / "history" / "agents" / namespace / "memory"


def evaluation_quarantine_root(
    run_id: str,
    agent_id: str,
    *,
    root: Path | None = None,
) -> Path:
    evaluation = _validated_slug(run_id, field="run_id")
    namespace = _validated_slug(agent_id, field="agent_id")
    base = (root or project_root()).resolve()
    return base / "history" / "evaluations" / evaluation / namespace / "quarantine"


def _validated_slug(value: str, *, field: str) -> str:
    if not _SAFE_SLUG.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must contain only ASCII letters, digits, '-' or '_'")
    return value
```

Add `/history/` to `.gitignore` immediately after `runs/`.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run:

```powershell
python -m pytest tests/unit/test_config.py tests/unit/memory/test_paths.py -q --basetemp=.pytest-tmp/history-task1-green
```

Expected: all selected tests PASS.

- [ ] **Step 8: Commit Task 1**

```powershell
git add .gitignore configs/default.yaml src/tau3_retail_evolver/config.py src/tau3_retail_evolver/memory/paths.py tests/unit/test_config.py tests/unit/memory/test_paths.py
git commit -m "feat: add agent memory history paths"
```

---

### Task 2: Open Persistent Training Memory Through One Factory

**Files:**
- Create: `src/tau3_retail_evolver/memory/factory.py`
- Modify: `src/tau3_retail_evolver/memory/__init__.py`
- Create: `tests/unit/memory/test_factory.py`

**Interfaces:**
- Consumes: `MemoryConfig.agent_id`
- Consumes: `training_memory_root(agent_id, root=...)`
- Produces: `open_training_memory(config: MemoryConfig, *, root: Path | None = None) -> MemoryRepository`

- [ ] **Step 1: Write failing persistence and isolation tests**

Create `tests/unit/memory/test_factory.py`:

```python
from __future__ import annotations

from pathlib import Path

from tau3_retail_evolver.config import MemoryConfig
from tau3_retail_evolver.memory.factory import open_training_memory


def test_same_agent_accumulates_memory_across_repository_reopens(tmp_path: Path) -> None:
    config = MemoryConfig(agent_id="retail")
    first_round = open_training_memory(config, root=tmp_path)
    created = first_round.add(
        tier="tip",
        content="Confirm identity before issuing a refund.",
        source_task_ids=("retail-task-1",),
        created_round=1,
    )

    next_round = open_training_memory(config, root=tmp_path)

    assert next_round.root == tmp_path.resolve() / "history" / "agents" / "retail" / "memory"
    assert next_round.get(created.id) == created


def test_different_agents_do_not_share_memory(tmp_path: Path) -> None:
    retail = open_training_memory(MemoryConfig(agent_id="retail"), root=tmp_path)
    created = retail.add(
        tier="skill",
        content="Inspect the retail order before modification.",
        source_task_ids=("retail-task-1",),
        created_round=1,
    )

    airline = open_training_memory(MemoryConfig(agent_id="airline"), root=tmp_path)

    assert airline.get(created.id) is None
    assert airline.list() == []
```

- [ ] **Step 2: Run factory tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/memory/test_factory.py -q --basetemp=.pytest-tmp/history-factory-red
```

Expected: collection ERROR because `tau3_retail_evolver.memory.factory` does not exist.

- [ ] **Step 3: Implement the factory without accepting `run_id`**

Create `src/tau3_retail_evolver/memory/factory.py`:

```python
from __future__ import annotations

from pathlib import Path

from tau3_retail_evolver.config import MemoryConfig
from tau3_retail_evolver.memory.paths import training_memory_root
from tau3_retail_evolver.memory.repository import MemoryRepository


def open_training_memory(
    config: MemoryConfig,
    *,
    root: Path | None = None,
) -> MemoryRepository:
    return MemoryRepository(training_memory_root(config.agent_id, root=root))
```

Replace `src/tau3_retail_evolver/memory/__init__.py` with:

```python
"""Four-tier JSON memory storage and retrieval."""

from tau3_retail_evolver.memory.embeddings import build_embedding_provider
from tau3_retail_evolver.memory.factory import open_training_memory
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.types import (
    MEMORY_TIERS,
    MemoryItem,
    MemorySnapshot,
    MemoryStatus,
    MemoryTier,
)

__all__ = [
    "MEMORY_TIERS",
    "MemoryItem",
    "MemoryRepository",
    "ReadOnlyMemoryRepository",
    "MemorySnapshot",
    "MemoryStatus",
    "MemoryTier",
    "build_embedding_provider",
    "open_training_memory",
]
```

The factory intentionally has no `run_id` parameter. `MemoryRepository` creates the resolved directory on first use.

- [ ] **Step 4: Run factory and existing Repository tests**

Run:

```powershell
python -m pytest tests/unit/memory/test_factory.py tests/unit/memory/test_repository.py -q --basetemp=.pytest-tmp/history-task2-green
```

Expected: all selected tests PASS and all writes remain beneath `tmp_path/history/`.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/tau3_retail_evolver/memory/__init__.py src/tau3_retail_evolver/memory/factory.py tests/unit/memory/test_factory.py
git commit -m "feat: open persistent agent memory"
```

---

### Task 3: Align the Staged Plan and Run the Stage Gate

**Files:**
- Modify: `docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md`
- Verify: `docs/superpowers/specs/2026-07-09-tau3-retail-opd-evolver-design.md`

**Interfaces:**
- Consumes: `open_training_memory(config, root=...)`
- Produces: one canonical documented layout with no `runs/<run_id>/memory` references.

- [ ] **Step 1: Replace obsolete staged-plan paths**

In `运行时产物约定`, replace the current `memory/...` bullet with the following three bullets:

```markdown
- `runs/<run_id>/` stores rollout JSONL, attribution JSONL, OPD examples, checkpoints, evaluation output and manifest only.
- `history/agents/<agent_id>/memory/` stores the continuously evolving four-tier Memory, embedding cache and immutable snapshots.
- `history/evaluations/<run_id>/<agent_id>/quarantine/` stores streaming evaluation Memory that training loaders must reject.
```

In Task 7.1, add `open_training_memory(config.memory)` to the interface and state that the iteration records the returned Repository's pre/post snapshot IDs without resetting it. In Task 8.1, replace `runs/<run_id>/eval/test_streaming/quarantine/` with `history/evaluations/<run_id>/<agent_id>/quarantine/` and require use of `evaluation_quarantine_root(run_id, agent_id)`.

- [ ] **Step 2: Verify documentation has no obsolete training Memory path**

Run:

```powershell
rg -n "runs/<run_id>/memory|runs/.*/memory" docs/superpowers/specs/2026-07-09-tau3-retail-opd-evolver-design.md docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md
```

Expected: no matches.

- [ ] **Step 3: Verify `history/` is ignored**

Run:

```powershell
git check-ignore -v history/agents/retail/memory/tip_memory.json
```

Expected: output identifies the `/history/` rule in `.gitignore`.

- [ ] **Step 4: Run focused and full verification**

Run:

```powershell
python -m pytest tests/unit/memory tests/unit/test_config.py -q --basetemp=.pytest-tmp/history-focused
python -m pytest -q --basetemp=.pytest-tmp/history-full
python -m compileall -q src
git diff --check
```

Expected: all tests PASS except the two existing explicitly skipped Tau2 integration tests; compile and diff checks exit 0.

- [ ] **Step 5: Run lifecycle audit**

Run:

```powershell
rg -n "MemoryRepository\(|open_training_memory\(|training_memory_root\(" src scripts tests
rg -n -i "sqlite|runs/.*/memory" src scripts configs tests
git status --short
```

Expected: production Memory creation uses the factory or centralized resolver; no SQLite or run-local training Memory implementation exists; only intended files are modified.

- [ ] **Step 6: Commit Task 3**

```powershell
git add docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md
git commit -m "docs: align stages with agent memory history"
```
