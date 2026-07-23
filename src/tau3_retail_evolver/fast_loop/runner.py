from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol, TypeVar
import unicodedata

from tau3_retail_evolver.credential_policy import is_credential_key
from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.fast_loop.decisions import (
    ActionDecision,
    Decision,
    SelectionDecision,
    WriteDecision,
    parse_decision,
)
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.prompts import (
    LifecyclePrompt,
    build_action_prompt,
    build_selection_prompt,
    build_write_prompt,
    project_public_context,
)
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import MemoryCandidate, Retriever
from tau3_retail_evolver.memory.types import (
    MemoryItem,
    canonical_content,
    stable_memory_id,
)
from tau3_retail_evolver.runs.manifest import sanitize_artifact_data


@dataclass(frozen=True, slots=True)
class FastLoopConfig:
    retrieve_top_k: int = 50
    max_episode_steps: int = 40
    memory_enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.memory_enabled) is not bool:
            raise ValueError("memory_enabled must be a bool")
        if self.retrieve_top_k < 1 or self.max_episode_steps < 1:
            raise ValueError("fast-loop limits must be positive")


@dataclass(frozen=True, slots=True)
class LifecycleResponse:
    raw_output: str
    sampling_params: Mapping[str, float]
    latency_s: float


class FastLoopPolicy(Protocol):
    def generate(self, prompt: LifecyclePrompt) -> LifecycleResponse: ...

    def repair(
        self,
        prompt: LifecyclePrompt,
        raw_output: str,
        error: str,
    ) -> LifecycleResponse: ...


class FastLoopEnvironment(Protocol):
    def reset(self, *, seed: int) -> ResetResult: ...

    def step(self, action: str) -> StepResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    task_id: str
    final_reward: float
    steps: int
    terminal_evaluation: Mapping[str, Any]
    simulation_result: Mapping[str, Any]
    selected_memory_ids: tuple[str, ...]
    written_memory_ids: tuple[str, ...]
    truncated: bool


DecisionT = TypeVar("DecisionT", bound=Decision)
_RESERVED_METADATA_KEYS = frozenset(
    {
        "source_run_id",
        "source_iteration",
        "source_final_reward",
        "selected_memory_ids",
    }
)


def run_fast_loop_episode(
    *,
    task_id: str,
    task_instruction: str,
    environment: FastLoopEnvironment,
    policy: FastLoopPolicy,
    repository: MemoryRepository | None,
    retriever: Retriever | None,
    config: FastLoopConfig,
    context: RunContext,
) -> EpisodeResult:
    """Run one learning episode and emit the evidence needed by later attribution."""
    validate_fast_loop_dependencies(
        config=config,
        repository=repository,
        retriever=retriever,
    )
    _require_learning_context(context)
    write_failure_emitted = False
    try:
        reset = environment.reset(seed=context.seed)
        reset_policy, reset_tools = _public_reset_info(reset.info)
        public_context = project_public_context(
            task_instruction=task_instruction,
            policy=reset_policy,
            tools=reset_tools,
            observation=reset.observation,
            history=(),
        )
        public_task_instruction = public_context["task_instruction"]
        public_policy = public_context["policy"]
        public_tools = public_context["tools"]
        public_observation = public_context["observation"]
        _emit(
            context,
            task_id,
            "EpisodeStarted",
            observation=public_observation,
            policy=public_policy,
            tools=public_tools,
        )

        if config.memory_enabled:
            assert repository is not None
            assert retriever is not None
            query = _retrieval_query(
                public_task_instruction,
                public_policy,
                public_tools,
                public_observation,
            )
            candidates = retriever.retrieve(
                query,
                repository,
                top_k=config.retrieve_top_k,
            )
            candidate_evidence = [
                _candidate_evidence(candidate) for candidate in candidates
            ]
            _emit(
                context,
                task_id,
                "MemoryCandidatesRetrieved",
                query_hash=candidates[0].query_hash if candidates else _query_hash(query),
                retriever_revision=(
                    candidates[0].retriever_revision
                    if candidates
                    else retriever.provider.model_revision
                ),
                candidates=candidate_evidence,
            )

            selection_prompt = build_selection_prompt(
                task_instruction=public_task_instruction,
                policy=public_policy,
                tools=public_tools,
                observation=public_observation,
                candidates=candidates,
            )
            selection, selection_audit = _generate_decision(
                policy,
                selection_prompt,
                SelectionDecision,
                candidate_ids=[candidate.memory_id for candidate in candidates],
                label="selection",
            )
            selected_ids = selection.memory_ids
            candidates_by_id = {
                candidate.memory_id: candidate for candidate in candidates
            }
            selected = [candidates_by_id[memory_id] for memory_id in selected_ids]
            _emit(
                context,
                task_id,
                "MemorySelected",
                selected_memory_ids=list(selected_ids),
                selected=[_candidate_evidence(candidate) for candidate in selected],
                raw_output_sha256=_output_hash(selection_audit["raw_output"]),
                repaired_output_sha256=(
                    _output_hash(selection_audit["repaired_output"])
                    if selection_audit["repaired_output"] is not None
                    else None
                ),
                parse_failed=selection_audit["error"] is not None,
                repair_used=selection_audit["repaired_output"] is not None,
                sampling_params=selection_audit["sampling_params"],
                latency_s=selection_audit["latency_s"],
            )
        else:
            selected: list[MemoryCandidate] = []
            selected_ids = ()
            _emit(context, task_id, "MemoryDisabled", reason="config")

        observation = public_observation
        last_nonblank_observation = public_observation
        trajectory: list[dict[str, Any]] = []
        terminal_evaluation: Mapping[str, Any] = {}
        simulation_result: Mapping[str, Any] = {}
        final_reward = 0.0
        truncated = False
        project_truncated = False
        steps = 0
        while steps < config.max_episode_steps:
            action_prompt = build_action_prompt(
                task_instruction=public_task_instruction,
                policy=public_policy,
                tools=public_tools,
                observation=observation,
                memories=selected,
                include_memory_context=config.memory_enabled,
            )
            action, action_audit = _generate_decision(
                policy,
                action_prompt,
                ActionDecision,
                label="action",
            )
            _emit(
                context,
                task_id,
                "DecisionMade",
                turn=steps,
                observation=observation,
                parsed_action=action.action,
                sampling_params=action_audit["sampling_params"],
                latency_s=action_audit["latency_s"],
                repair_used=action_audit["repaired_output"] is not None,
            )
            step = environment.step(action.action)
            public_info = _public_step_info(step.info)
            _emit(
                context,
                task_id,
                "EnvironmentStepped",
                turn=steps,
                action=action.action,
                observation=step.observation,
                reward=step.reward,
                done=step.done,
                terminated=step.terminated,
                truncated=step.truncated,
                public_info=public_info,
            )
            _validate_step_boundary(step)
            trajectory.append(
                {
                    "observation": observation,
                    "action": action.action,
                    "next_observation": step.observation,
                    "reward": step.reward,
                    "done": step.done,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    **public_info,
                }
            )
            steps += 1
            observation = step.observation
            if observation.strip():
                last_nonblank_observation = observation
            final_reward = step.reward
            if step.done:
                terminal_evaluation = _terminal_json_mapping(
                    step.info, "reward_info", task_id
                )
                simulation_result = _terminal_json_mapping(
                    step.info, "simulation_run", task_id
                )
                truncated = step.truncated
                break
        else:
            truncated = True
            project_truncated = True

        _emit(
            context,
            task_id,
            "EpisodeFinished",
            steps=steps,
            final_reward=final_reward,
            terminal_evaluation=terminal_evaluation,
            simulation_result=simulation_result,
            truncated=truncated,
            project_truncated=project_truncated,
        )

        written_ids: tuple[str, ...] = ()
        if config.memory_enabled:
            assert repository is not None
            write_prompt = build_write_prompt(
                task_instruction=public_task_instruction,
                policy=public_policy,
                tools=public_tools,
                observation=last_nonblank_observation,
                trajectory=trajectory,
                terminal_evaluation=terminal_evaluation,
            )
            write_decision, write_audit = _generate_decision(
                policy,
                write_prompt,
                WriteDecision,
                validator=_validate_write_decision,
                label="write",
            )
            proposals = [
                _write_proposal(memory, task_id, context, final_reward, selected_ids)
                for memory in write_decision.memories
            ]
            _emit(
                context,
                task_id,
                "MemoryWriteProposed",
                proposals=[proposal["evidence"] for proposal in proposals],
                repair_used=write_audit["repaired_output"] is not None,
                sampling_params=write_audit["sampling_params"],
                latency_s=write_audit["latency_s"],
            )
            try:
                written_ids, replayed_ids = _persist_proposals(repository, proposals)
            except BaseException as error:
                committed_ids = getattr(error, "_fast_loop_committed_ids", ())
                replayed_ids = getattr(error, "_fast_loop_replayed_ids", ())
                write_failure_emitted = True
                try:
                    _emit(
                        context,
                        task_id,
                        "MemoryWriteFailed",
                        committed_memory_ids=list(committed_ids),
                        replayed_memory_ids=list(replayed_ids),
                        error=_sanitized_error(error),
                    )
                except Exception as evidence_error:
                    error.add_note(
                        f"Fast-loop write failure evidence also failed: {evidence_error}"
                    )
                raise
            _emit(
                context,
                task_id,
                "MemoryWriteCommitted",
                written_memory_ids=list(written_ids),
                replayed_memory_ids=list(replayed_ids),
            )
        result = EpisodeResult(
            task_id=task_id,
            final_reward=final_reward,
            steps=steps,
            terminal_evaluation=terminal_evaluation,
            simulation_result=simulation_result,
            selected_memory_ids=selected_ids,
            written_memory_ids=written_ids,
            truncated=truncated,
        )
    except BaseException as error:
        if not write_failure_emitted:
            _emit_failure(context, task_id, error)
        _close_after_failure(environment, error)
        raise

    try:
        environment.close()
    except BaseException as error:
        _emit_failure(context, task_id, error)
        raise
    return result


def _require_learning_context(context: RunContext) -> None:
    if context.split != "train":
        raise ValueError("fast-loop learning requires the train split")
    if context.mode != RunMode.LEARN:
        raise ValueError("fast-loop learning requires LEARN mode")


def validate_fast_loop_dependencies(
    *,
    config: FastLoopConfig,
    repository: MemoryRepository | None,
    retriever: Retriever | None,
) -> None:
    if not isinstance(config, FastLoopConfig):
        raise ValueError("fast-loop config must be a FastLoopConfig")
    if config.memory_enabled:
        if not isinstance(repository, MemoryRepository) or repository.is_read_only:
            raise ValueError("fast-loop learning requires a mutable MemoryRepository")
        if not isinstance(retriever, Retriever):
            raise ValueError("enabled memory requires a Retriever")
    elif repository is not None or retriever is not None:
        raise ValueError("disabled memory requires no repository or retriever")


def _public_reset_info(info: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    if "policy" not in info or "tools" not in info:
        raise RuntimeError("Tau2 reset info is missing public policy or tools")
    tools = info["tools"]
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise RuntimeError("Tau2 public tools must be a sequence")
    return info["policy"], list(tools)


def _retrieval_query(
    task_instruction: str,
    policy: Any,
    tools: Sequence[Any],
    observation: str,
) -> str:
    tool_names = sorted(set(_find_named_values(tools)))
    return json.dumps(
        {
            "task_instruction": task_instruction,
            "policy": policy,
            "tool_names": tool_names,
            "observation": observation,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _find_named_values(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "name" and isinstance(nested, str):
                names.append(nested)
            else:
                names.extend(_find_named_values(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            names.extend(_find_named_values(nested))
    return names


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _output_hash(output: str) -> str:
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def _candidate_evidence(candidate: MemoryCandidate) -> dict[str, Any]:
    return {
        "memory_id": candidate.memory_id,
        "memory_version": candidate.memory_version,
        "tier": candidate.tier.value,
        "rank": candidate.rank,
        "similarity": candidate.similarity,
    }


def _generate_decision(
    policy: FastLoopPolicy,
    prompt: LifecyclePrompt,
    decision_type: type[DecisionT],
    *,
    candidate_ids: Sequence[str] | None = None,
    validator: Callable[[DecisionT], Any] | None = None,
    label: str,
) -> tuple[DecisionT, dict[str, Any]]:
    response = policy.generate(prompt)
    result = parse_decision(
        response.raw_output,
        decision_type,
        validator=validator,
        candidate_ids=candidate_ids,
    )
    repaired_output: str | None = None
    initial_error = result.error
    if result.decision is None:
        repair = policy.repair(prompt, response.raw_output, result.error or "invalid output")
        repaired_output = repair.raw_output
        result = parse_decision(
            repair.raw_output,
            decision_type,
            validator=validator,
            candidate_ids=candidate_ids,
        )
    if result.decision is None:
        raise ValueError(f"invalid {label} decision after repair: {result.error}")
    return result.decision, {
        "raw_output": response.raw_output,
        "repaired_output": repaired_output,
        "error": initial_error,
        "sampling_params": dict(response.sampling_params),
        "latency_s": response.latency_s,
    }


def _validate_step_boundary(step: StepResult) -> None:
    if step.done != (step.terminated or step.truncated):
        raise RuntimeError("Tau2 step terminal flags are inconsistent")


def _public_step_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return sanitize_artifact_data(
        {key: info[key] for key in ("parse_error",) if key in info}
    )


def _terminal_json_mapping(
    info: Mapping[str, Any],
    field: str,
    task_id: str,
) -> Mapping[str, Any]:
    value = info.get(field)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"terminal {field} JSON is invalid for task {task_id}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"terminal {field} must be a JSON object for task {task_id}")
    try:
        parsed = json.loads(
            json.dumps(sanitize_artifact_data(value), allow_nan=False)
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"terminal {field} is not JSON safe for task {task_id}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"terminal {field} must be a JSON object for task {task_id}")
    return parsed


def _write_proposal(
    memory: Any,
    task_id: str,
    context: RunContext,
    final_reward: float,
    selected_ids: tuple[str, ...],
) -> dict[str, Any]:
    _validate_write_metadata(memory.metadata)
    metadata = _without_forbidden_metadata(dict(memory.metadata))
    _validate_write_metadata(metadata)
    for key in _RESERVED_METADATA_KEYS:
        metadata.pop(key, None)
    metadata.update(
        source_run_id=context.run_id,
        source_iteration=context.iteration,
        source_final_reward=final_reward,
        selected_memory_ids=list(selected_ids),
    )
    memory_id = stable_memory_id(memory.tier, memory.content)
    add_kwargs = {
        "tier": memory.tier,
        "content": memory.content,
        "source_task_ids": (task_id,),
        "created_round": context.iteration,
        "metadata": metadata,
        "retrieval_text": memory.retrieval_text,
    }
    return {
        "memory_id": memory_id,
        "add_kwargs": add_kwargs,
        "evidence": {
            "memory_id": memory_id,
            "tier": memory.tier.value,
            "content": memory.content,
            "retrieval_text": memory.retrieval_text or memory.content,
            "metadata": metadata,
            "source_task_ids": [task_id],
            "created_round": context.iteration,
        },
    }


def _validate_write_decision(decision: WriteDecision) -> WriteDecision:
    for memory in decision.memories:
        _validate_write_metadata(memory.metadata)
    return decision


def _validate_write_metadata(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = unicodedata.normalize("NFKC", str(key))
            compact_key = "".join(
                character
                for character in normalized_key.casefold()
                if character.isalnum()
            )
            if "attributionscore" in compact_key:
                raise ValueError("write metadata must not contain attribution score")
            if is_credential_key(normalized_key):
                raise ValueError("write metadata must not contain credential fields")
            _validate_write_metadata(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _validate_write_metadata(nested)


def _without_forbidden_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_forbidden_metadata(nested)
            for key, nested in value.items()
            if not _is_forbidden_metadata_key(key)
        }
    if isinstance(value, list):
        return [_without_forbidden_metadata(item) for item in value]
    return value


def _is_forbidden_metadata_key(key: Any) -> bool:
    normalized_key = unicodedata.normalize("NFKC", str(key))
    compact_key = "".join(
        character for character in normalized_key.casefold() if character.isalnum()
    )
    return "attributionscore" in compact_key or is_credential_key(normalized_key)


def _persist_proposals(
    repository: MemoryRepository,
    proposals: Sequence[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    written: list[str] = []
    committed: list[str] = []
    replayed: list[str] = []
    for proposal in proposals:
        try:
            item = repository.add(**proposal["add_kwargs"])
        except ValueError as error:
            try:
                existing = repository.get(proposal["memory_id"])
            except BaseException as lookup_error:
                _attach_write_progress(lookup_error, committed, replayed)
                lookup_error.add_note(
                    f"Replay lookup followed rejected add ({type(error).__name__})"
                )
                raise
            if not _is_safe_replay(existing, proposal):
                _attach_write_progress(error, committed, replayed)
                raise
            item = existing
            replayed.append(item.id)
        except BaseException as error:
            _attach_write_progress(error, committed, replayed)
            raise
        else:
            committed.append(item.id)
        written.append(item.id)
    return tuple(written), tuple(replayed)


def _is_safe_replay(existing: MemoryItem | None, proposal: Mapping[str, Any]) -> bool:
    if existing is None:
        return False
    kwargs = proposal["add_kwargs"]
    return (
        existing.id == proposal["memory_id"]
        and existing.tier == kwargs["tier"]
        and canonical_content(existing.content) == canonical_content(kwargs["content"])
        and existing.source_task_ids == tuple(kwargs["source_task_ids"])
    )


def _attach_write_progress(
    error: BaseException,
    committed: Sequence[str],
    replayed: Sequence[str],
) -> None:
    try:
        setattr(error, "_fast_loop_committed_ids", tuple(committed))
        setattr(error, "_fast_loop_replayed_ids", tuple(replayed))
    except Exception:
        error.add_note("Fast-loop write progress was recorded before failure")


def _emit(
    context: RunContext,
    task_id: str,
    event_type: str,
    **payload: Any,
) -> None:
    context.event_writer.append(
        sanitize_artifact_data(context.event(event_type, task_id, **payload))
    )


def _emit_failure(context: RunContext, task_id: str, error: BaseException) -> None:
    try:
        _emit(
            context,
            task_id,
            "EpisodeFailed",
            error=_sanitized_error(error),
        )
    except Exception as evidence_error:
        error.add_note(f"Fast-loop failure evidence also failed: {evidence_error}")


def _sanitized_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": "operation failed"}


def _close_after_failure(
    environment: FastLoopEnvironment,
    error: BaseException,
) -> None:
    try:
        environment.close()
    except BaseException as cleanup_error:
        error.add_note(f"Tau2 cleanup also failed: {cleanup_error}")
