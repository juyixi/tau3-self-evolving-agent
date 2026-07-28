# Stage 5 Outcome Attribution and OPD Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build an auditable Stage 5 pipeline that converts same-policy Tau2 retail train runs into normalized evidence, paper-exact memory attribution, and four leakage-safe OPD datasets.

**Architecture:** Source runs remain immutable. A strict loader validates policy and Memory snapshot lineage, an event state machine materializes an evidence ledger, attribution implements paper Eq.11-Eq.12, and typed builders emit the four Eq.13 views before an independent auditor verifies hashes, chronology, split isolation, and public/privileged boundaries.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, canonical JSON/JSONL, existing Tau2 runtime metadata and file-backed Memory snapshots.

## Global Constraints

- Only `split=train`, `mode=learn`, `memory_enabled=true` source runs may produce Stage 5 artifacts.
- Every dataset build uses one `iteration`, `model_revision`, and `adapter_revision`; source runs must form a continuous Memory snapshot chain.
- The paper is authoritative for Eq.11-Eq.16; implement selected mean minus not-selected mean and sum group contributions.
- Memory remains JSON under `history/agents/<agent_id>/memory/`; evidence, attribution, OPD data, and audits are JSONL/JSON run artifacts.
- `test`, `base`, evaluation quarantine, evaluator rubrics, golden arguments, credentials, and privileged diagnostics must never enter public training inputs.
- Stage 5 creates paired public/privileged contexts only; Stage 6 samples student completions online and evaluates the teacher on the same student token prefixes.
- Existing `memory.score_threshold=0.01` and `memory.teacher_memory_cap=20` remain authoritative.
- New Slow Loop defaults are tier priors `trajectory=0.9`, `tip=0.8`, `skill=1.0`, `tool=1.2`, redundancy threshold `0.90`, and at most `50` redundancy pairs.
- Use TDD for every production behavior: write one focused test, verify the expected failure, implement the minimum behavior, and rerun focused plus affected regression tests.
- Do not stage or modify the pre-existing working-tree state in `docs/superpowers/plans/2026-07-13-stage-4-fast-loop.md`.

---

## File Map

**Create:**

- `src/tau3_retail_evolver/slow_loop/__init__.py` - public Stage 5 package boundary.
- `src/tau3_retail_evolver/slow_loop/task_grouping.py` - privileged Retail task metadata to anonymous group signatures.
- `src/tau3_retail_evolver/slow_loop/source_runs.py` - immutable source-run loading, hashing, and policy/snapshot lineage.
- `src/tau3_retail_evolver/slow_loop/evidence.py` - schema-2 event state machine and normalized evidence models.
- `src/tau3_retail_evolver/slow_loop/attribution.py` - Eq.11-Eq.12 scoring and diagnostics.
- `src/tau3_retail_evolver/slow_loop/examples.py` - Eq.13 `sel/act/write/maint` builders.
- `src/tau3_retail_evolver/slow_loop/leakage.py` - recursive public/privileged and credential guards.
- `src/tau3_retail_evolver/slow_loop/dataset.py` - deterministic artifact writing, manifest, atomic publication, and audit orchestration.
- `src/tau3_retail_evolver/slow_loop/audit.py` - independent dataset audit.
- `scripts/build_opd_dataset.py` - build CLI.
- `scripts/audit_opd_dataset.py` - audit CLI.
- `tests/unit/slow_loop/test_task_grouping.py`
- `tests/unit/slow_loop/test_source_runs.py`
- `tests/unit/slow_loop/test_evidence.py`
- `tests/unit/slow_loop/test_attribution.py`
- `tests/unit/slow_loop/test_leakage.py`
- `tests/unit/slow_loop/test_examples.py`
- `tests/unit/slow_loop/test_dataset.py`
- `tests/unit/scripts/test_build_opd_dataset.py`
- `tests/unit/scripts/test_audit_opd_dataset.py`
- `tests/integration/test_stage5_opd_dataset.py`

**Modify:**

- `src/tau3_retail_evolver/config.py` - typed Slow Loop data configuration.
- `configs/default.yaml` - declared Stage 5 defaults.
- `src/tau3_retail_evolver/fast_loop/events.py` - event schema 2.
- `src/tau3_retail_evolver/fast_loop/maintenance.py` - remove privileged usage diagnostics from student/public maintenance input.
- `scripts/run_fast_loop.py` - real task signatures and Memory namespace provenance.
- `src/tau3_retail_evolver/io/jsonl.py` - strict JSONL iteration shared by Stage 5.
- Existing config, Fast Loop, manifest, and script tests affected by schema/config changes.
- `docs/stage-4-fast-loop-validation.md` - schema-2 five-task deferred validation command and Stage 6 gate.

---

### Task 1: Schema 2, Slow Loop Config, and Retail Task Grouping

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/__init__.py`
- Create: `src/tau3_retail_evolver/slow_loop/task_grouping.py`
- Create: `tests/unit/slow_loop/test_task_grouping.py`
- Modify: `src/tau3_retail_evolver/config.py`
- Modify: `configs/default.yaml`
- Modify: `src/tau3_retail_evolver/fast_loop/events.py`
- Modify: `src/tau3_retail_evolver/fast_loop/maintenance.py`
- Modify: `scripts/run_fast_loop.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/fast_loop/test_maintenance.py`
- Modify: `tests/unit/scripts/test_run_fast_loop.py`

**Interfaces:**
- Consumes: Tau2 Retail `tasks.json`, `ProjectConfig`, existing `RunContext` and run manifest writer.
- Produces: `RetailTaskGroups.from_file(path: Path)`, `RetailTaskGroups.signature_for(task_id: str) -> str`, `SlowLoopConfig`, schema-2 events, and manifest `rollout_options.memory_agent_id`.

- [x] **Step 1: Write failing grouping and configuration tests**

```python
def test_group_signature_uses_only_canonical_mutating_action_names(tmp_path: Path) -> None:
    tasks = [
        {
            "id": "7",
            "evaluation_criteria": {
                "actions": [
                    {"name": "get_order_details", "arguments": {"order_id": "secret-a"}},
                    {"name": "return_delivered_order_items", "arguments": {"item_ids": ["x"]}},
                    {"name": "find_user_id_by_email", "arguments": {"email": "hidden@example.com"}},
                ]
            },
        },
        {
            "id": "8",
            "evaluation_criteria": {
                "actions": [
                    {"name": "return_delivered_order_items", "arguments": {"item_ids": ["different"]}},
                    {"name": "get_user_details", "arguments": {"user_id": "other"}},
                ]
            },
        },
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")

    groups = RetailTaskGroups.from_file(path)

    assert groups.signature_for("7") == groups.signature_for("8")
    assert groups.signature_for("7").startswith("retail-actions-v1:")
    assert "return_delivered_order_items" not in groups.signature_for("7")


def test_unknown_action_fails_closed(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, actions=[{"name": "new_unclassified_action"}])
    with pytest.raises(ValueError, match="unclassified retail action"):
        RetailTaskGroups.from_file(path)


def test_default_config_exposes_stage5_data_parameters() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.slow_loop.tier_priors == {
        "trajectory": 0.9,
        "tip": 0.8,
        "skill": 1.0,
        "tool": 1.2,
    }
    assert config.slow_loop.redundancy_threshold == 0.90
    assert config.slow_loop.max_redundancy_pairs == 50


@pytest.mark.parametrize(
    "tier_priors",
    [
        {"trajectory": 0.9, "tip": 0.8, "skill": 1.0},
        {"trajectory": 0.9, "tip": 0.8, "skill": 1.0, "tool": 1.2, "other": 1.0},
    ],
)
def test_slow_loop_tier_priors_require_exact_keys(tier_priors: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="tier_priors"):
        SlowLoopConfig(tier_priors=tier_priors)
```

- [x] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/unit/slow_loop/test_task_grouping.py tests/unit/test_config.py -v`

Expected: collection/import failure for missing `slow_loop.task_grouping` or missing `ProjectConfig.slow_loop`.

- [x] **Step 3: Implement the typed config and grouping API**

```python
class SlowLoopConfig(_ConfigModel):
    tier_priors: dict[Literal["trajectory", "tip", "skill", "tool"], float] = Field(
        default_factory=lambda: {
            "trajectory": 0.9,
            "tip": 0.8,
            "skill": 1.0,
            "tool": 1.2,
        }
    )
    redundancy_threshold: float = Field(default=0.90, ge=-1.0, le=1.0)
    max_redundancy_pairs: int = Field(default=50, ge=0)


groups = RetailTaskGroups.from_file(Path("data/tau2/retail/tasks.json"))
task_group_signature = groups.signature_for("0")
assert task_group_signature.startswith("retail-actions-v1:")
```

Validate `tier_priors` with a model validator so its keys are exactly `trajectory`, `tip`, `skill`, and `tool`. Use exact read-only and mutating allowlists from the approved spec. Hash canonical JSON containing `domain`, `grouping_revision`, and sorted unique mutating names. Reject missing task IDs, duplicate IDs, malformed criteria, or unclassified names.

- [x] **Step 4: Add schema-2 Fast Loop regression tests**

```python
def assert_schema2_fast_loop_artifacts(run_path: Path) -> None:
    manifest = json.loads((run_path / "manifest.json").read_text())
    events = _read_jsonl(run_path / "rollouts" / "events.jsonl")
    assert manifest["rollout_options"]["memory_agent_id"] == "retail"
    assert {event["schema_version"] for event in events} == {2}
    assert all(event["task_group"].startswith("retail-actions-v1:") for event in events)


def test_public_maintenance_diagnostics_exclude_privileged_usage(tmp_path: Path) -> None:
    diagnostics = bounded_diagnostics(_seed_repository(tmp_path))
    item = diagnostics["tip"]["items"][0]
    assert set(item) == {"id", "content", "version", "status"}
```

- [x] **Step 5: Run the new tests and verify RED**

Run: `python -m pytest tests/unit/fast_loop/test_maintenance.py tests/unit/scripts/test_run_fast_loop.py -v`

Expected: failures showing schema version `1`, hard-coded `retail`, missing namespace, and public usage fields.

- [x] **Step 6: Integrate signatures and remove privileged maintenance fields**

Set `SCHEMA_VERSION = 2`. Resolve `RetailTaskGroups` from `runtime.retail_tasks_path` before creating `RunContext`. Pass `{task_id: groups.signature_for(task_id)}` and add `memory_agent_id` plus `task_grouping_revision` to `rollout_options`. Restrict public maintenance items to ID/content/version/status; derive usage only in Stage 5.

- [x] **Step 7: Run focused and affected regression tests**

Run: `python -m pytest tests/unit/slow_loop/test_task_grouping.py tests/unit/test_config.py tests/unit/fast_loop/test_maintenance.py tests/unit/scripts/test_run_fast_loop.py -v`

Expected: PASS.

- [x] **Step 8: Commit Task 1**

```bash
git add configs/default.yaml src/tau3_retail_evolver/config.py src/tau3_retail_evolver/slow_loop src/tau3_retail_evolver/fast_loop/events.py src/tau3_retail_evolver/fast_loop/maintenance.py scripts/run_fast_loop.py tests/unit/slow_loop/test_task_grouping.py tests/unit/test_config.py tests/unit/fast_loop/test_maintenance.py tests/unit/scripts/test_run_fast_loop.py
git commit -m "feat: add stage 5 task grouping provenance"
```

---

### Task 2: Strict Source Run Loader and JSONL Reader

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/source_runs.py`
- Create: `tests/unit/slow_loop/test_source_runs.py`
- Modify: `src/tau3_retail_evolver/io/jsonl.py`
- Modify: `tests/unit/io/test_jsonl.py`

**Interfaces:**
- Consumes: explicit source-run paths, official `RetailTaskCatalog`, `ProjectConfig`, and `history/agents/<agent_id>/memory/snapshots`.
- Produces: `SourceRun`, `SourceRunSet`, `load_source_runs(paths, *, catalog, memory_root) -> SourceRunSet`, and `iter_jsonl_objects(path) -> Iterator[dict[str, Any]]`.

- [x] **Step 1: Write strict JSONL reader tests**

```python
def test_iter_jsonl_objects_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok":true}\nnot-json\n', encoding="utf-8")
    iterator = iter_jsonl_objects(path)
    assert next(iterator) == {"ok": True}
    with pytest.raises(ValueError, match=r"events.jsonl:2"):
        next(iterator)


def test_iter_jsonl_objects_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        list(iter_jsonl_objects(path))
```

- [x] **Step 2: Run JSONL tests and verify RED**

Run: `python -m pytest tests/unit/io/test_jsonl.py -v`

Expected: import failure for `iter_jsonl_objects`.

- [x] **Step 3: Implement the strict reader and rerun tests**

```python
def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be a JSON object at {path}:{line_number}")
            yield value
```

Expected: `tests/unit/io/test_jsonl.py` PASS.

- [x] **Step 4: Write source policy and snapshot-lineage tests**

```python
def test_source_runs_require_same_on_policy_revision_and_continuous_snapshots(tmp_path: Path) -> None:
    first = _write_source_run(tmp_path, "run-a", before=0, after=2, input_snapshot="s0", output_snapshot="s1")
    second = _write_source_run(tmp_path, "run-b", before=2, after=4, input_snapshot="s1", output_snapshot="s2")
    loaded = load_source_runs([second, first], catalog=_catalog("1", "2", "3", "4"), memory_root=_snapshots(tmp_path, "s0", "s1", "s2"))
    assert [run.run_id for run in loaded.runs] == ["run-a", "run-b"]


@pytest.mark.parametrize("mutation", ["test_split", "memory_disabled", "adapter_mismatch", "snapshot_gap"])
def test_source_runs_fail_closed_on_invalid_lineage(tmp_path: Path, mutation: str) -> None:
    paths, catalog, memory_root = _invalid_source_fixture(tmp_path, mutation)
    with pytest.raises(ValueError, match=_EXPECTED_ERRORS[mutation]):
        load_source_runs(paths, catalog=catalog, memory_root=memory_root)


def test_source_runs_allow_repeated_tasks_across_distinct_passes(tmp_path: Path) -> None:
    paths, catalog, memory_root = _continuous_repeated_task_fixture(tmp_path)
    loaded = load_source_runs(paths, catalog=catalog, memory_root=memory_root)
    assert [run.manifest["task_ids"] for run in loaded.runs] == [("1",), ("1",)]
```

Stage 8 扩展约束：单个 source run 内 task ID 必须唯一；不同 on-policy pass
可以重复同一个 train task。episode 身份始终使用 `run_id:task_id`，因此重复采样
不会覆盖 evidence、attribution 或 OPD example。

- [x] **Step 5: Run source loader tests and verify RED**

Run: `python -m pytest tests/unit/slow_loop/test_source_runs.py -v`

Expected: import failure for `slow_loop.source_runs`.

- [x] **Step 6: Implement immutable source models and fail-closed validation**

```python
@dataclass(frozen=True, slots=True)
class SourceRun:
    path: Path
    run_id: str
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    events_path: Path
    manifest_sha256: str
    summary_sha256: str
    events_sha256: str


@dataclass(frozen=True, slots=True)
class SourceRunSet:
    runs: tuple[SourceRun, ...]
    iteration: int
    model_revision: str
    adapter_revision: str | None
    tau2_commit: str
    split_hash: str
    memory_agent_id: str
```

Validate manifest schema, summary counts, event schema/mode/provenance, exact train membership, unique task IDs, policy revision equality, task-range continuity, snapshot equality, snapshot directory existence, and path exclusion from `history/evaluations`.

- [x] **Step 7: Run focused source tests**

Run: `python -m pytest tests/unit/io/test_jsonl.py tests/unit/slow_loop/test_source_runs.py -v`

Expected: PASS.

- [x] **Step 8: Commit Task 2**

```bash
git add src/tau3_retail_evolver/io/jsonl.py src/tau3_retail_evolver/slow_loop/source_runs.py tests/unit/io/test_jsonl.py tests/unit/slow_loop/test_source_runs.py
git commit -m "feat: validate stage 5 source run lineage"
```

---

### Task 3: Event-to-Evidence Ledger

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/evidence.py`
- Create: `tests/unit/slow_loop/test_evidence.py`

**Interfaces:**
- Consumes: `SourceRunSet`, schema-2 event rows, and verified snapshot directories.
- Produces: `EpisodeEvidence`, `MaintenanceEvidence`, `EvidenceLedger`, and `build_evidence(source_runs, *, memory_root) -> EvidenceLedger`.

- [x] **Step 1: Write the happy-path episode reconstruction test**

```python
def test_build_evidence_reconstructs_candidate_selection_trajectory_and_write(tmp_path: Path) -> None:
    source_runs, memory_root = _schema2_source_with_committed_write(tmp_path)
    ledger = build_evidence(source_runs, memory_root=memory_root)
    episode = ledger.episodes[0]
    assert episode.task_id == "1"
    assert episode.task_group.startswith("retail-actions-v1:")
    assert [candidate.memory_id for candidate in episode.candidates] == ["mem-tip-a", "mem-tool-b"]
    assert episode.selected_memory_ids == ("mem-tip-a",)
    assert episode.trajectory[0].action == "lookup_order(order_id='1')"
    assert episode.final_reward == 1.0
    assert episode.committed_new_memory_ids == ("mem-skill-c",)
    assert episode.replayed_memory_ids == ()
```

- [x] **Step 2: Write invalid-state-machine and snapshot tests**

```python
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("selection_before_retrieval", "MemorySelected before candidates"),
        ("selected_not_candidate", "selected memory is not a candidate"),
        ("duplicate_finish", "duplicate EpisodeFinished"),
        ("proposal_without_commit", "incomplete write lifecycle"),
        ("candidate_missing_from_snapshot", "candidate missing from snapshot"),
        ("candidate_version_mismatch", "candidate version mismatch"),
    ],
)
def test_build_evidence_rejects_invalid_lifecycle(tmp_path: Path, mutation: str, message: str) -> None:
    source_runs, memory_root = _mutated_schema2_source(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        build_evidence(source_runs, memory_root=memory_root)
```

- [x] **Step 3: Run evidence tests and verify RED**

Run: `python -m pytest tests/unit/slow_loop/test_evidence.py -v`

Expected: import failure for `slow_loop.evidence`.

- [x] **Step 4: Implement typed evidence models and per-task state machines**

```python
class MemoryCandidateEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    memory_id: str
    memory_version: int
    tier: MemoryTier
    rank: int
    similarity: float
    content: str
    content_sha256: str


class EpisodeEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    evidence_schema_version: Literal[1] = 1
    episode_id: str
    run_id: str
    iteration: int
    task_id: str
    task_group: str
    model_revision: str
    adapter_revision: str | None
    memory_snapshot_id: str
    seed: int
    candidates: tuple[MemoryCandidateEvidence, ...]
    selected_memory_ids: tuple[str, ...]
    trajectory: tuple[TrajectoryStepEvidence, ...]
    final_reward: float
    terminated: bool
    truncated: bool
    proposed_memory_ids: tuple[str, ...]
    committed_new_memory_ids: tuple[str, ...]
    replayed_memory_ids: tuple[str, ...]
    source_event_sha256: str
```

Create one state object per task. Require exact event progression, consistent common provenance, monotonically increasing turns, and a committed or explicitly empty write lifecycle. Load candidates from `snapshots/<memory_snapshot_id>` with `ReadOnlyMemoryRepository`; compare ID/tier/version and hash content.

- [x] **Step 5: Add maintenance evidence tests and implementation**

```python
def test_maintenance_evidence_uses_public_repository_view_and_prior_history(tmp_path: Path) -> None:
    source_runs, memory_root = _source_with_maintenance(tmp_path)
    ledger = build_evidence(source_runs, memory_root=memory_root)
    maintenance = ledger.maintenance[0]
    assert maintenance.maintenance_round == 1
    assert maintenance.trigger_task_index == 30
    assert "usage_count" not in maintenance.public_repository[0]
    assert maintenance.commands[0]["operation"] == "merge"
    assert len(maintenance.prior_episode_ids) == 30
```

Reconstruct `H_qQ` from preceding `EpisodeEvidence` in completed-task order. Accept only `MaintenanceStarted -> MaintenanceProposed -> MaintenanceCommitted` with matching round and snapshot provenance.

- [x] **Step 6: Run evidence and affected Fast Loop tests**

Run: `python -m pytest tests/unit/slow_loop/test_evidence.py tests/unit/fast_loop/test_runner.py tests/unit/fast_loop/test_maintenance.py -v`

Expected: PASS.

- [x] **Step 7: Commit Task 3**

```bash
git add src/tau3_retail_evolver/slow_loop/evidence.py tests/unit/slow_loop/test_evidence.py
git commit -m "feat: materialize stage 5 evidence ledger"
```

---

### Task 4: Paper-Exact Memory Attribution

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/attribution.py`
- Create: `tests/unit/slow_loop/test_attribution.py`

**Interfaces:**
- Consumes: `EvidenceLedger`, tier priors, and score threshold.
- Produces: `MemoryGroupScore`, `MemoryScore`, and `compute_memory_scores(ledger, *, tier_priors, score_threshold) -> tuple[MemoryScore, ...]`.

- [x] **Step 1: Write a hand-calculated multi-group Eq.11-Eq.12 test**

```python
def test_compute_memory_scores_matches_paper_equations() -> None:
    ledger = _ledger_for_memory(
        "mem-tip-a",
        [
            ("returns", True, 1.0),
            ("returns", True, 0.5),
            ("returns", False, 0.0),
            ("returns", False, 0.5),
            ("exchange", True, 1.0),
            ("exchange", False, 0.0),
        ],
    )
    score = compute_memory_scores(
        ledger,
        tier_priors={"trajectory": 0.9, "tip": 0.8, "skill": 1.0, "tool": 1.2},
        score_threshold=0.01,
    )[0]
    expected_a_hat = (2 / 4) * (0.75 - 0.25) + (1 / 2) * (1.0 - 0.0)
    expected_gamma = 1.0 - 1.0 / math.sqrt(1.0 + 3)
    assert score.attribution == pytest.approx(expected_a_hat)
    assert score.confidence == pytest.approx(expected_gamma)
    assert score.value == pytest.approx(0.8 * expected_gamma * expected_a_hat)
```

- [x] **Step 2: Write evidence-control, negative, and future-write tests**

```python
def test_unretrieved_tasks_never_enter_candidate_control() -> None:
    score = _score_with_large_unretrieved_reward()
    assert score.groups[0].retrieved_count == 2


def test_one_sided_group_is_omitted_and_no_groups_is_null() -> None:
    score = _score_selected_only()
    assert score.status == "insufficient_evidence"
    assert score.value is None


def test_creator_episode_cannot_value_its_own_write() -> None:
    score = _score_new_memory_with_creator_and_future_use()
    assert score.source_episode_ids == ("future-selected", "future-not-selected")
```

- [x] **Step 3: Run attribution tests and verify RED**

Run: `python -m pytest tests/unit/slow_loop/test_attribution.py -v`

Expected: import failure for `slow_loop.attribution`.

- [x] **Step 4: Implement exact scoring and full diagnostics**

```python
class MemoryGroupScore(BaseModel):
    group: str
    selected_count: int
    not_selected_count: int
    selected_reward_mean: float
    not_selected_reward_mean: float
    rho: float
    delta: float
    contribution: float
    source_episode_ids: tuple[str, ...]


class MemoryScore(BaseModel):
    memory_id: str
    tier: MemoryTier
    observed_versions: tuple[int, ...]
    groups: tuple[MemoryGroupScore, ...]
    selected_count: int
    confidence: float
    tier_prior: float
    attribution: float | None
    value: float | None
    status: Literal["scored", "insufficient_evidence"]
    qualified_for_supervision: bool
```

Index each episode's candidates and selected IDs. For each group, include only retrieved episodes and require both sides. Sum `rho * delta`; do not average groups and do not add recency. Exclude evidence whose episode order is not later than the Memory's creator episode. Sort all groups and Memory scores deterministically.

- [x] **Step 5: Run attribution and evidence regression tests**

Run: `python -m pytest tests/unit/slow_loop/test_attribution.py tests/unit/slow_loop/test_evidence.py -v`

Expected: PASS.

- [x] **Step 6: Commit Task 4**

```bash
git add src/tau3_retail_evolver/slow_loop/attribution.py tests/unit/slow_loop/test_attribution.py
git commit -m "feat: compute outcome calibrated memory attribution"
```

---

### Task 5: Leakage Guard and Four Eq.13 Example Builders

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/leakage.py`
- Create: `src/tau3_retail_evolver/slow_loop/examples.py`
- Create: `tests/unit/slow_loop/test_leakage.py`
- Create: `tests/unit/slow_loop/test_examples.py`

**Interfaces:**
- Consumes: `EvidenceLedger`, `MemoryScore` rows, score threshold, teacher cap, and redundancy settings.
- Produces: `OPDExample`, `build_selection_examples`, `build_action_examples`, `build_writing_examples`, `build_maintenance_examples`, and `audit_example_boundaries(example) -> None`.

- [x] **Step 1: Write recursive leakage tests**

```python
@pytest.mark.parametrize(
    "payload",
    [
        {"nested": {"memoryValue": 0.4}},
        {"diagnostics": [{"last_used": "2026-01-01"}]},
        {"evaluation_criteria": {"actions": []}},
        {"url": "https://user:secret@example.com/path"},
    ],
)
def test_public_input_rejects_privileged_or_secret_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        audit_public_input("sel", payload)


def test_action_public_input_rejects_memory_even_under_alias() -> None:
    with pytest.raises(ValueError, match="action public input contains memory"):
        audit_public_input("act", {"context": {"selectedMemories": ["mem-tip-a"]}})
```

- [x] **Step 2: Run leakage tests and verify RED**

Run: `python -m pytest tests/unit/slow_loop/test_leakage.py -v`

Expected: import failure for `slow_loop.leakage`.

- [x] **Step 3: Implement normalized-key recursive guards**

```python
def normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


FORBIDDEN_PUBLIC_KEYS = frozenset(
    normalized_key(key)
    for key in (
        "api_key",
        "base_url",
        "evaluator",
        "value",
        "confidence",
        "usage_count",
        "success_count",
        "last_used_at",
    )
)
```

Implement `audit_public_input` as a recursive mapping/sequence walker over normalized keys. Reuse `is_credential_key` and the manifest URL policy. Reject credential keys/URLs, evaluator keys, privileged diagnostic variants, and every Memory alias for `act`. Permit stable IDs to appear in both public and privileged fields only for `sel`, `write`, and `maint` join relationships.

- [x] **Step 4: Write selection and action example tests**

```python
def test_selection_example_keeps_all_scored_candidates_and_separates_values() -> None:
    example = build_selection_examples(_ledger(), _scores())[0]
    assert example.kind == "sel"
    assert "value" not in json.dumps(example.public_input).lower()
    assert {row["memory_id"] for row in example.privileged_hindsight["candidate_scores"]} == {"mem-a", "mem-b"}
    assert example.sampling_contract["mode"] == "online"


def test_action_example_removes_memory_from_public_and_uses_same_group_success() -> None:
    example = build_action_examples(_failed_episode_ledger_with_group_success(), _scores())[0]
    public_text = json.dumps(example.public_input)
    assert "mem-a" not in public_text
    assert example.privileged_hindsight["valuable_selected_memories"][0]["memory_id"] == "mem-a"
    assert example.privileged_hindsight["successful_trajectory"]["task_group"] == example.provenance["task_group"]
```

- [x] **Step 5: Write writing and maintenance example tests**

```python
def test_writing_example_uses_only_future_scored_committed_memories() -> None:
    example = build_writing_examples(_ledger_with_future_write_evidence(), _scores())[0]
    assert example.kind == "write"
    assert example.privileged_hindsight["written_memory_scores"][0]["creator_episode_id"] != "future-use"


def test_maintenance_example_keeps_usage_and_redundancy_privileged() -> None:
    example = build_maintenance_examples(
        _maintenance_ledger(),
        _scores(),
        teacher_memory_cap=20,
        redundancy_threshold=0.90,
        max_redundancy_pairs=50,
    )[0]
    assert "usage" not in json.dumps(example.public_input).lower()
    assert "usage" in json.dumps(example.privileged_hindsight).lower()
    assert len(example.privileged_hindsight["memory_diagnostics"]) <= 20
    assert len(example.privileged_hindsight["redundancy_pairs"]) <= 50
```

- [x] **Step 6: Run example tests and verify RED**

Run: `python -m pytest tests/unit/slow_loop/test_examples.py -v`

Expected: import failure for `slow_loop.examples`.

- [x] **Step 7: Implement typed OPD examples and deterministic builders**

```python
class OPDExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[2] = 2
    example_id: str
    kind: Literal["sel", "act", "write", "maint"]
    public_input: dict[str, Any]
    privileged_hindsight: dict[str, Any]
    response_schema: dict[str, Any]
    sampling_contract: dict[str, Any]
    provenance: dict[str, Any]
```

Use canonical hashes for example IDs. Build `sel` at most once per task, `act`
exactly once per successful task with a nonempty complete trajectory, `write` at
most once per task with future-qualified committed writes, and `maint` once per
committed round. The schema-2 correction explicitly forbids expanding an `act`
trajectory into per-turn samples or pairing a task with another task's successful
trajectory. Compute redundancy only for same-revision, same-dimension embeddings;
select deterministic high/low value, high-usage, and redundancy endpoint buckets
under the cap.

- [x] **Step 8: Run leakage, example, attribution, and config tests**

Run: `python -m pytest tests/unit/slow_loop/test_leakage.py tests/unit/slow_loop/test_examples.py tests/unit/slow_loop/test_attribution.py tests/unit/test_config.py -v`

Expected: PASS.

- [x] **Step 9: Commit Task 5**

```bash
git add src/tau3_retail_evolver/slow_loop/leakage.py src/tau3_retail_evolver/slow_loop/examples.py tests/unit/slow_loop/test_leakage.py tests/unit/slow_loop/test_examples.py
git commit -m "feat: build four privileged opd views"
```

---

### Task 6: Deterministic Dataset Publication, Manifest, Auditor, and CLIs

**Files:**
- Create: `src/tau3_retail_evolver/slow_loop/dataset.py`
- Create: `src/tau3_retail_evolver/slow_loop/audit.py`
- Create: `scripts/build_opd_dataset.py`
- Create: `scripts/audit_opd_dataset.py`
- Create: `tests/unit/slow_loop/test_dataset.py`
- Create: `tests/unit/scripts/test_build_opd_dataset.py`
- Create: `tests/unit/scripts/test_audit_opd_dataset.py`

**Interfaces:**
- Consumes: explicit source-run paths, project config, Tau2 runtime metadata, evidence ledger, scores, and examples.
- Produces: `build_opd_dataset(request: DatasetBuildRequest) -> DatasetBuildResult`, `audit_dataset(path: Path) -> AuditReport`, and two module CLIs.

- [x] **Step 1: Write deterministic writer and no-overwrite tests**

```python
def test_dataset_build_writes_canonical_files_and_manifest_hashes(tmp_path: Path) -> None:
    result = build_opd_dataset(_build_request(tmp_path))
    root = result.dataset_dir
    assert (root / "evidence" / "episodes.jsonl").is_file()
    assert (root / "attribution" / "memory_scores.jsonl").is_file()
    assert all((root / "datasets" / f"{kind}.jsonl").is_file() for kind in ("sel", "act", "write", "maint"))
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    for relative_path, artifact in manifest["artifacts"].items():
        assert _sha256(root / relative_path) == artifact["sha256"]


def test_dataset_build_refuses_existing_build_id(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    build_opd_dataset(request)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_opd_dataset(request)
```

- [x] **Step 2: Write independent audit mutation tests**

```python
@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("duplicate_example", "duplicate_example_id"),
        ("public_value_leak", "public_privileged_leak"),
        ("artifact_hash_changed", "artifact_hash_mismatch"),
        ("missing_online_contract", "missing_online_sampling_contract"),
        ("test_task_id", "non_train_source"),
    ],
)
def test_auditor_fails_closed_on_mutated_dataset(tmp_path: Path, mutation: str, error_code: str) -> None:
    dataset_dir = _built_then_mutated_dataset(tmp_path, mutation)
    report = audit_dataset(dataset_dir)
    assert report.passed is False
    assert error_code in {error.code for error in report.errors}
```

- [x] **Step 3: Run dataset tests and verify RED**

Run: `python -m pytest tests/unit/slow_loop/test_dataset.py -v`

Expected: import failure for `slow_loop.dataset` or `slow_loop.audit`.

- [x] **Step 4: Implement canonical artifacts and atomic publication**

```python
@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    source_run_paths: tuple[Path, ...]
    dataset_build_id: str
    output_root: Path
    config_path: Path
    project_root: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    dataset_dir: Path
    manifest: Mapping[str, Any]
    audit_report: Mapping[str, Any]
```

Write canonical JSONL rows sorted by stable IDs. Record source hashes, exact revisions, snapshot chain, resolved config, counts, skip reasons, line counts, and artifact SHA256. Build in a temporary sibling directory; run internal audit; publish only when audit passes.

- [x] **Step 5: Implement independent audit without trusting builder objects**

Re-read manifest and every artifact from disk. Recompute hashes and counts, validate schemas and unique IDs, rerun `audit_example_boundaries`, require the manifest split hash to equal `RetailTaskCatalog.OFFICIAL_SPLIT_SHA256`, verify every source task ID belongs to the manifest's exact official train task ID set, and validate attribution chronology/provenance references. Do not reuse in-memory builder objects.

- [x] **Step 6: Write CLI parsing and stdout tests**

```python
def test_build_cli_requires_explicit_source_runs_and_prints_canonical_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(build_script, "build_opd_dataset", lambda request: _result(tmp_path))
    assert build_script.main([
        "--config", "configs/default.yaml",
        "--source-run", "runs/a",
        "--source-run", "runs/b",
        "--dataset-build-id", "opd-iter0-001",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["dataset_build_id"] == "opd-iter0-001"


def test_audit_cli_returns_nonzero_when_report_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audit_script, "audit_dataset", lambda path: _failed_report())
    assert audit_script.main(["--dataset-dir", str(tmp_path)]) == 1
```

- [x] **Step 7: Implement both CLIs**

Expose repeatable `--source-run`, `--dataset-build-id`, `--config`, `--output-root`, and optional `--project-root`. Validate safe slug build IDs. Print only canonical summaries; never print task content, Memory content, prompts, or secrets.

- [x] **Step 8: Run focused dataset and CLI tests**

Run: `python -m pytest tests/unit/slow_loop/test_dataset.py tests/unit/scripts/test_build_opd_dataset.py tests/unit/scripts/test_audit_opd_dataset.py -v`

Expected: PASS.

- [x] **Step 9: Commit Task 6**

```bash
git add src/tau3_retail_evolver/slow_loop/dataset.py src/tau3_retail_evolver/slow_loop/audit.py scripts/build_opd_dataset.py scripts/audit_opd_dataset.py tests/unit/slow_loop/test_dataset.py tests/unit/scripts/test_build_opd_dataset.py tests/unit/scripts/test_audit_opd_dataset.py
git commit -m "feat: publish and audit stage 5 opd datasets"
```

---

### Task 7: End-to-End Integration, Documentation, and Stage Gate

**Files:**
- Create: `tests/integration/test_stage5_opd_dataset.py`
- Modify: `docs/stage-4-fast-loop-validation.md`
- Modify: `docs/superpowers/plans/2026-07-20-stage-5-outcome-attribution-opd-dataset.md`

**Interfaces:**
- Consumes: complete Stage 5 pipeline and schema-2 Fast Loop contract.
- Produces: deterministic two-run integration proof, documented real five-task follow-up, and completed plan checklist.

- [x] **Step 1: Write a two-run end-to-end integration test**

```python
def test_two_continuous_schema2_runs_build_and_audit_four_opd_views(tmp_path: Path) -> None:
    source_runs, config_path, project_root = _write_two_continuous_runs_with_all_views(tmp_path)
    first = build_opd_dataset(DatasetBuildRequest(
        source_run_paths=source_runs,
        dataset_build_id="opd-iter0-first",
        output_root=tmp_path / "runs",
        config_path=config_path,
        project_root=project_root,
    ))
    second = build_opd_dataset(DatasetBuildRequest(
        source_run_paths=tuple(reversed(source_runs)),
        dataset_build_id="opd-iter0-second",
        output_root=tmp_path / "runs",
        config_path=config_path,
        project_root=project_root,
    ))
    assert first.audit_report["passed"] is True
    assert second.audit_report["passed"] is True
    assert _content_hashes(first.dataset_dir) == _content_hashes(second.dataset_dir)
    assert all(_jsonl_count(first.dataset_dir / "datasets" / f"{kind}.jsonl") > 0 for kind in ("sel", "act", "write", "maint"))
```

- [x] **Step 2: Run integration test and verify RED**

Run: `python -m pytest tests/integration/test_stage5_opd_dataset.py -v`

Expected: failure until cross-module manifest normalization excludes build-specific path/ID fields from content-comparison hashes.

- [x] **Step 3: Make deterministic build normalization pass**

Ensure source run ordering, evidence IDs, attribution ordering, example ordering, and artifact serialization are independent of CLI input order. Compare content artifacts, not `dataset_build_id` or absolute output paths, in the determinism assertion.

- [x] **Step 4: Complete and document the real five-task schema-2 validation**

The AutoDL validation completed on 2026-07-23 with Qwen, the Tau2 user simulator, the embedding model, and evaluator credentials available together. Two continuous schema-2 runs produced `opd-iter0-5tasks-20260723g`; the independent audit passed with 5 evidence episodes, 11 Memory scores, 4 selection examples, and 1 writing example. Action and maintenance remained empty under the five-task evidence thresholds, so real 30-task four-view coverage remains a follow-up rather than a Stage 6 minimum-gate blocker.

- [x] **Step 5: Run all Stage 5 and Stage 4 unit/integration tests**

Run: `python -m pytest tests/unit tests/integration/test_fast_loop_tau2_retail.py tests/integration/test_stage5_opd_dataset.py -v`

Expected: PASS, excluding tests already marked skip by their declared external-runtime preconditions.

- [x] **Step 6: Run the complete automated suite**

Run: `python -m pytest -v`

Expected: all runnable tests PASS; only declared external-integration skips remain.

- [x] **Step 7: Run repository hygiene checks**

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected: only intended Stage 5 files and the plan checkbox update before commit.

- [x] **Step 8: Update all completed plan checkboxes and commit Task 7**

```bash
git add tests/integration/test_stage5_opd_dataset.py docs/stage-4-fast-loop-validation.md docs/superpowers/plans/2026-07-20-stage-5-outcome-attribution-opd-dataset.md
git commit -m "test: verify stage 5 opd dataset pipeline"
```

---

## Final Review Checklist

- [x] `memory_scores.jsonl` matches hand-calculated Eq.11-Eq.12 fixtures.
- [x] No recency term or candidate-pool baseline entered attribution.
- [x] Every source task is official train data from one policy iteration.
- [x] All candidate content is resolved from the event's frozen snapshot.
- [x] Creator episodes cannot value their own writes.
- [x] `act.public_input` contains no Memory data.
- [x] Usage, confidence, value, and redundancy remain teacher-only.
- [x] All four dataset files exist and empty categories carry structured reasons.
- [x] Stage 6 online sampling contract is present in every example.
- [x] Dataset build is deterministic and independently auditable.
- [x] Schema-1, Memory-disabled, failed, test/base, stale, duplicate, and broken-lineage inputs fail closed.
- [x] Full runnable test suite and `git diff --check` pass.
- [x] Real five-task schema-2 build and independent audit passed on `opd-iter0-5tasks-20260723g`; real 30-task four-view coverage remains a follow-up.
