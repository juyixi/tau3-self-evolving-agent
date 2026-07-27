from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import threading
from typing import Any
import uuid

from tau3_retail_evolver.envs.runtime import Tau2RunDomainRuntime
from tau3_retail_evolver.fast_loop.action_codec import TAU2_STOP_ACTION
from tau3_retail_evolver.fast_loop.decisions import ActionDecision, SelectionDecision, WriteDecision
from tau3_retail_evolver.fast_loop.events import RunContext
from tau3_retail_evolver.fast_loop.prompts import (
    build_action_prompt,
    build_selection_prompt,
    build_write_prompt,
    project_public_context,
)
from tau3_retail_evolver.fast_loop.runner import (
    EpisodeResult,
    FastLoopConfig,
    FastLoopPolicy,
    _apply_write_quotas,
    _accumulate_token_usage,
    _candidate_evidence,
    _empty_write_decision,
    _emit_skipped_memory_write,
    _generate_decision,
    _output_hash,
    _persist_proposals,
    _query_hash,
    _retrieval_query,
    _validate_write_decision,
    _write_proposal,
)
from tau3_retail_evolver.memory.outcomes import classify_episode_memory
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import MemoryCandidate, Retriever
from tau3_retail_evolver.memory.types import MemoryStatus
from tau3_retail_evolver.runs.manifest import sanitize_artifact_data


MemoryView = MemoryRepository | ReadOnlyMemoryRepository | None
ContextFactory = Callable[[str, int], RunContext]
_REGISTRY_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class Tau2BatchEpisode:
    seed: int
    result: EpisodeResult


@dataclass(frozen=True, slots=True)
class Tau2BatchFailure:
    task_id: str
    seed: int
    stage: str
    error_type: str


@dataclass(frozen=True, slots=True)
class Tau2BatchResult:
    episodes: tuple[Tau2BatchEpisode, ...]
    failures: tuple[Tau2BatchFailure, ...]
    tau2_results: Any


@dataclass(slots=True)
class _AgentState:
    messages: list[Any]
    selected: tuple[MemoryCandidate, ...] = ()
    started: bool = False
    turn: int = 0


@dataclass(slots=True)
class _BufferedWriter:
    events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def flush_to(self, writer: Any) -> None:
        for event in self.events:
            writer.append(event)


def run_tau2_fast_loop_batch(
    *,
    runtime: Tau2RunDomainRuntime,
    domain: str,
    split: str,
    task_ids: Sequence[str],
    run_seed: int,
    max_concurrency: int,
    user_llm: str,
    user_llm_args: Mapping[str, Any],
    agent_model: str,
    policy: FastLoopPolicy,
    repository: MemoryView,
    retriever: Retriever | None,
    config: FastLoopConfig,
    context_factory: ContextFactory,
    task_instruction: str,
    write_memory: bool,
    memory_disabled_reason: str = "protocol",
) -> Tau2BatchResult:
    """Execute a Tau2 task batch through the official concurrent ``run_domain`` API."""
    tasks = tuple(str(task_id) for task_id in task_ids)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("run_domain task IDs must be non-empty and unique")
    if max_concurrency < 1:
        raise ValueError("run_domain max_concurrency must be positive")
    if config.memory_enabled:
        if repository is None or retriever is None:
            raise ValueError("memory-enabled run_domain execution requires repository and Retriever")
    elif repository is not None or retriever is not None:
        raise ValueError("memory-disabled run_domain execution requires no Memory dependencies")
    if write_memory and not isinstance(repository, MemoryRepository):
        raise ValueError("run_domain Memory writes require a mutable MemoryRepository")

    retrieval_lock = threading.Lock()
    agent_name = f"tau3_fast_loop_{uuid.uuid4().hex}"
    agent_factory = _make_agent_factory(
        runtime=runtime,
        policy=policy,
        repository=repository,
        retriever=retriever,
        config=config,
        task_instruction=task_instruction,
        memory_disabled_reason=memory_disabled_reason,
        retrieval_lock=retrieval_lock,
    )
    with _REGISTRY_LOCK:
        runtime.registry.register_agent_factory(agent_factory, agent_name)
        try:
            tau2_results = runtime.run_domain(
                runtime.text_run_config(
                    domain=domain,
                    task_split_name=split,
                    task_ids=list(tasks),
                    agent=agent_name,
                    llm_agent=agent_model,
                    llm_args_agent={},
                    user="user_simulator",
                    llm_user=user_llm,
                    llm_args_user=dict(user_llm_args),
                    num_trials=1,
                    max_steps=max(4, config.max_episode_steps * 4),
                    max_errors=10,
                    # Tau2 includes raw llm_args in its Results metadata. Keep the
                    # result in memory and persist only our sanitized artifacts.
                    save_to=None,
                    max_concurrency=min(max_concurrency, len(tasks)),
                    seed=run_seed,
                    log_level="WARNING",
                    max_retries=2,
                    retry_delay=1.0,
                    auto_resume=False,
                    hallucination_retries=0,
                    enforce_communication_protocol=True,
                )
            )
        finally:
            runtime.registry._agent_factories.pop(agent_name, None)

    by_task = {str(simulation.task_id): simulation for simulation in tau2_results.simulations}
    missing = set(tasks) - set(by_task)
    if missing:
        raise RuntimeError(f"Tau2 run_domain omitted task results: {sorted(missing)}")

    episodes: list[Tau2BatchEpisode] = []
    failures: list[Tau2BatchFailure] = []
    for task_id in tasks:
        simulation = by_task[task_id]
        seed = int(simulation.seed if simulation.seed is not None else run_seed)
        canonical_context = context_factory(task_id, seed)
        buffer = _BufferedWriter()
        # Keep the caller's run-level seed in canonical events. Tau2 derives a
        # per-simulation seed internally; it remains available on
        # Tau2BatchEpisode without breaking source-run provenance checks.
        episode_context = replace(canonical_context, event_writer=buffer)
        if _is_infrastructure_failure(simulation):
            _emit_task_failure(canonical_context, task_id, stage="run_domain")
            failures.append(
                Tau2BatchFailure(
                    task_id=task_id,
                    seed=seed,
                    stage="run_domain",
                    error_type=_simulation_error_type(simulation),
                )
            )
            continue

        memory_ids_before = _memory_ids(repository)
        try:
            result = _finalize_simulation(
                runtime=runtime,
                simulation=simulation,
                policy=policy,
                repository=repository,
                config=config,
                context=episode_context,
                write_memory=write_memory,
            )
        except Exception as error:
            discarded = _discard_new_memories(
                repository,
                memory_ids_before,
                updated_round=canonical_context.iteration,
            )
            _emit_task_failure(
                canonical_context,
                task_id,
                stage="finalize",
                discarded_memory_ids=discarded,
            )
            failures.append(
                Tau2BatchFailure(
                    task_id=task_id,
                    seed=seed,
                    stage="finalize",
                    error_type=type(error).__name__,
                )
            )
            continue
        buffer.flush_to(canonical_context.event_writer)
        episodes.append(Tau2BatchEpisode(seed=seed, result=result))

    return Tau2BatchResult(
        episodes=tuple(episodes),
        failures=tuple(failures),
        tau2_results=tau2_results,
    )


def _make_agent_factory(
    *,
    runtime: Tau2RunDomainRuntime,
    policy: FastLoopPolicy,
    repository: MemoryView,
    retriever: Retriever | None,
    config: FastLoopConfig,
    task_instruction: str,
    memory_disabled_reason: str,
    retrieval_lock: threading.Lock,
) -> Callable[..., Any]:
    assistant_type = runtime.assistant_message
    tool_call_type = runtime.tool_call
    multi_tool_type = runtime.multi_tool_message

    class Tau3FastLoopAgent(runtime.half_duplex_agent):
        def __init__(
            self,
            tools: list[Any],
            domain_policy: str,
            **kwargs: Any,
        ) -> None:
            del kwargs
            super().__init__(tools=tools, domain_policy=domain_policy)
            self._public_tools = [tool.openai_schema for tool in tools]

        def get_init_state(self, message_history: list[Any] | None = None) -> _AgentState:
            return _AgentState(messages=list(message_history or ()))

        def set_seed(self, seed: int) -> None:
            self._seed = seed

        @classmethod
        def is_stop(cls, message: Any) -> bool:
            return TAU2_STOP_ACTION in str(getattr(message, "content", "") or "")

        def generate_next_message(
            self,
            message: Any,
            state: _AgentState,
        ) -> tuple[Any, _AgentState]:
            if isinstance(message, multi_tool_type):
                state.messages.extend(message.tool_messages)
            elif message is not None:
                state.messages.append(message)

            if state.turn >= config.max_episode_steps:
                stop = assistant_type(role="assistant", content=TAU2_STOP_ACTION)
                state.messages.append(stop)
                return stop, state

            observation = _render_observation(state.messages)
            start_marker: dict[str, Any] | None = None
            if not state.started:
                public_context = project_public_context(
                    task_instruction=task_instruction,
                    policy=self.domain_policy,
                    tools=self._public_tools,
                    observation=observation,
                    history=(),
                )
                if config.memory_enabled:
                    assert repository is not None
                    assert retriever is not None
                    query = _retrieval_query(
                        public_context["task_instruction"],
                        public_context["policy"],
                        public_context["tools"],
                        public_context["observation"],
                    )
                    with retrieval_lock:
                        candidates = retriever.retrieve(
                            query,
                            repository,
                            top_k=config.retrieve_top_k,
                        )
                    selection_prompt = build_selection_prompt(
                        task_instruction=public_context["task_instruction"],
                        policy=public_context["policy"],
                        tools=public_context["tools"],
                        observation=public_context["observation"],
                        candidates=candidates,
                    )
                    selection, audit = _generate_decision(
                        policy,
                        selection_prompt,
                        SelectionDecision,
                        candidate_ids=[candidate.memory_id for candidate in candidates],
                        label="selection",
                    )
                    candidates_by_id = {
                        candidate.memory_id: candidate for candidate in candidates
                    }
                    state.selected = tuple(
                        candidates_by_id[memory_id] for memory_id in selection.memory_ids
                    )
                    start_marker = {
                        **public_context,
                        "memory_enabled": True,
                        "query_hash": (
                            candidates[0].query_hash if candidates else _query_hash(query)
                        ),
                        "retriever_revision": (
                            candidates[0].retriever_revision
                            if candidates
                            else retriever.provider.model_revision
                        ),
                        "candidates": [
                            _candidate_evidence(candidate) for candidate in candidates
                        ],
                        "selected": [
                            _candidate_evidence(candidate) for candidate in state.selected
                        ],
                        "selected_memory_ids": list(selection.memory_ids),
                        "selection_audit": _audit_marker(audit),
                    }
                else:
                    start_marker = {
                        **public_context,
                        "memory_enabled": False,
                        "memory_disabled_reason": memory_disabled_reason,
                        "selected": [],
                        "selected_memory_ids": [],
                    }
                state.started = True

            action_prompt = build_action_prompt(
                task_instruction=task_instruction,
                policy=self.domain_policy,
                tools=self._public_tools,
                observation=observation,
                memories=state.selected,
                include_memory_context=config.memory_enabled,
            )
            action, audit = _generate_decision(
                policy,
                action_prompt,
                ActionDecision,
                label="action",
            )
            marker = {
                "schema_version": 1,
                "turn": state.turn,
                "observation": observation,
                "action": action.action,
                "action_audit": _audit_marker(audit),
            }
            if start_marker is not None:
                marker["start"] = start_marker
            assistant = _assistant_for_action(
                assistant_type=assistant_type,
                tool_call_type=tool_call_type,
                action=action.action,
                turn=state.turn,
                audit=audit,
                marker=marker,
            )
            state.messages.append(assistant)
            state.turn += 1
            return assistant, state

    def factory(tools: list[Any], domain_policy: str, **kwargs: Any) -> Any:
        return Tau3FastLoopAgent(
            tools=tools,
            domain_policy=domain_policy,
            **kwargs,
        )

    return factory


def _assistant_for_action(
    *,
    assistant_type: type,
    tool_call_type: type,
    action: str,
    turn: int,
    audit: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> Any:
    usage = None
    if audit["prompt_tokens"] is not None and audit["completion_tokens"] is not None:
        usage = {
            "prompt_tokens": audit["prompt_tokens"],
            "completion_tokens": audit["completion_tokens"],
        }
    tool = _parse_tool_action(action)
    common = {
        "role": "assistant",
        "usage": usage,
        "generation_time_seconds": audit["latency_s"],
        "raw_data": {"tau3_fast_loop": sanitize_artifact_data(dict(marker))},
    }
    if tool is not None:
        name, arguments = tool
        return assistant_type(
            **common,
            tool_calls=[
                tool_call_type(
                    id=f"tau3-tool-{turn}",
                    name=name,
                    arguments=arguments,
                    requestor="assistant",
                )
            ],
        )
    return assistant_type(**common, content=action)


def _parse_tool_action(action: str) -> tuple[str, dict[str, Any]] | None:
    try:
        value = json.loads(action)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        name = value.get("name")
        arguments = value.get("arguments")
        if isinstance(name, str) and isinstance(arguments, Mapping):
            return name, dict(arguments)

    try:
        expression = ast.parse(action, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return None
    if expression.args or any(keyword.arg is None for keyword in expression.keywords):
        return None
    try:
        arguments = {
            str(keyword.arg): ast.literal_eval(keyword.value)
            for keyword in expression.keywords
        }
    except (ValueError, TypeError):
        return None
    return expression.func.id, arguments


def _render_observation(messages: Sequence[Any]) -> str:
    rows: list[str] = []
    for message in messages:
        role = str(getattr(message, "role", "message"))
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            rows.append(f"{role}: {content.strip()}")
        for tool_call in list(getattr(message, "tool_calls", None) or ()):
            arguments = json.dumps(
                dict(getattr(tool_call, "arguments", {}) or {}),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            rows.append(
                f"{role} tool_call: {getattr(tool_call, 'name', '')}({arguments})"
            )
    rendered = "\n".join(rows).strip()
    if not rendered:
        raise RuntimeError("Tau2 conversation has no public observation")
    return rendered


def _audit_marker(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_output_sha256": _output_hash(audit["raw_output"]),
        "repaired_output_sha256": (
            _output_hash(audit["repaired_output"])
            if audit["repaired_output"] is not None
            else None
        ),
        "parse_failed": audit["error"] is not None,
        "repair_used": audit["repaired_output"] is not None,
        "fallback_used": audit["fallback_used"],
        "sampling_params": dict(audit["sampling_params"]),
        "latency_s": audit["latency_s"],
        "prompt_tokens": audit["prompt_tokens"],
        "completion_tokens": audit["completion_tokens"],
    }


def _finalize_simulation(
    *,
    runtime: Tau2RunDomainRuntime,
    simulation: Any,
    policy: FastLoopPolicy,
    repository: MemoryView,
    config: FastLoopConfig,
    context: RunContext,
    write_memory: bool,
) -> EpisodeResult:
    messages = list(simulation.messages or ())
    decisions = [
        (index, message, message.raw_data["tau3_fast_loop"])
        for index, message in enumerate(messages)
        if isinstance(message, runtime.assistant_message)
        and isinstance(getattr(message, "raw_data", None), Mapping)
        and isinstance(message.raw_data.get("tau3_fast_loop"), Mapping)
    ]
    if not decisions:
        raise RuntimeError("Tau2 simulation contains no fast-loop decisions")
    start = decisions[0][2].get("start")
    if not isinstance(start, Mapping):
        raise RuntimeError("Tau2 simulation is missing fast-loop start evidence")
    task_id = str(simulation.task_id)

    _emit(
        context,
        task_id,
        "EpisodeStarted",
        observation=start["observation"],
        policy=start["policy"],
        tools=start["tools"],
    )
    selected_ids = tuple(str(value) for value in start["selected_memory_ids"])
    response_count = 0
    response_parse_error_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    token_usage_complete = True
    if start["memory_enabled"]:
        selection_audit = start["selection_audit"]
        _emit(
            context,
            task_id,
            "MemoryCandidatesRetrieved",
            query_hash=start["query_hash"],
            retriever_revision=start["retriever_revision"],
            candidates=start["candidates"],
        )
        _emit(
            context,
            task_id,
            "MemorySelected",
            selected_memory_ids=list(selected_ids),
            selected=start["selected"],
            **selection_audit,
        )
        response_count += 1
        response_parse_error_count += int(selection_audit["parse_failed"])
        prompt_tokens, completion_tokens, token_usage_complete = _accumulate_marker_usage(
            prompt_tokens,
            completion_tokens,
            token_usage_complete,
            selection_audit,
        )
    else:
        _emit(
            context,
            task_id,
            "MemoryDisabled",
            reason=start["memory_disabled_reason"],
        )

    final_reward = _final_reward(simulation)
    terminal_evaluation = _model_dump_mapping(simulation.reward_info)
    simulation_result = _model_dump_mapping(simulation)
    termination_reason = str(
        getattr(getattr(simulation, "termination_reason", ""), "value", None)
        or getattr(simulation, "termination_reason", "")
    )
    project_truncated = (
        termination_reason in {"max_steps", "timeout"}
        or len(decisions) >= config.max_episode_steps
    )
    trajectory: list[dict[str, Any]] = []
    parse_error_count = 0
    last_observation = str(start["observation"])
    for position, (message_index, _message, marker) in enumerate(decisions):
        turn = position
        observation = str(marker["observation"])
        next_observation = (
            str(decisions[position + 1][2]["observation"])
            if position + 1 < len(decisions)
            else _render_observation(messages)
        )
        is_last = position == len(decisions) - 1
        public_info: dict[str, Any] = {}
        next_message_index = (
            decisions[position + 1][0] if position + 1 < len(decisions) else len(messages)
        )
        if any(
            isinstance(item, runtime.tool_message) and bool(getattr(item, "error", False))
            for item in messages[message_index + 1 : next_message_index]
        ):
            public_info["parse_error"] = "tau2_tool_error"
            parse_error_count += 1
        audit = marker["action_audit"]
        response_count += 1
        response_parse_error_count += int(audit["parse_failed"])
        prompt_tokens, completion_tokens, token_usage_complete = _accumulate_marker_usage(
            prompt_tokens,
            completion_tokens,
            token_usage_complete,
            audit,
        )
        _emit(
            context,
            task_id,
            "DecisionMade",
            turn=turn,
            observation=observation,
            parsed_action=marker["action"],
            sampling_params=audit["sampling_params"],
            latency_s=audit["latency_s"],
            repair_used=audit["repair_used"],
            prompt_tokens=audit["prompt_tokens"],
            completion_tokens=audit["completion_tokens"],
        )
        _emit(
            context,
            task_id,
            "EnvironmentStepped",
            turn=turn,
            action=marker["action"],
            observation=next_observation,
            reward=final_reward if is_last else 0.0,
            done=is_last,
            terminated=is_last and not project_truncated,
            truncated=is_last and project_truncated,
            public_info=public_info,
        )
        trajectory.append(
            {
                "observation": observation,
                "action": marker["action"],
                "next_observation": next_observation,
                "reward": final_reward if is_last else 0.0,
                "done": is_last,
                "terminated": is_last and not project_truncated,
                "truncated": is_last and project_truncated,
                **public_info,
            }
        )
        if next_observation.strip():
            last_observation = next_observation

    _emit(
        context,
        task_id,
        "EpisodeFinished",
        steps=len(decisions),
        final_reward=final_reward,
        terminal_evaluation=terminal_evaluation,
        simulation_result=simulation_result,
        truncated=project_truncated,
        project_truncated=project_truncated,
    )

    written_ids: tuple[str, ...] = ()
    if write_memory:
        assert isinstance(repository, MemoryRepository)
        memory_policy = classify_episode_memory(
            final_reward=final_reward,
            terminal_evaluation=terminal_evaluation,
            truncated=project_truncated,
        )
        if not memory_policy.should_generate:
            _emit_skipped_memory_write(context, task_id, memory_policy)
        else:
            write_prompt = build_write_prompt(
                task_instruction=str(start["task_instruction"]),
                policy=start["policy"],
                tools=start["tools"],
                observation=last_observation,
                trajectory=trajectory,
                terminal_evaluation=terminal_evaluation,
                memory_outcome=memory_policy.prompt_payload(),
            )
            write_decision, write_audit = _generate_decision(
                policy,
                write_prompt,
                WriteDecision,
                validator=lambda decision: _validate_write_decision(
                    decision,
                    tools=start["tools"],
                    task_id=task_id,
                    context=context,
                    final_reward=final_reward,
                    trajectory=trajectory,
                    memory_policy=memory_policy,
                ),
                label="write",
                invalid_fallback=_empty_write_decision,
            )
            response_count += 1
            response_parse_error_count += int(write_audit["error"] is not None)
            prompt_tokens, completion_tokens, token_usage_complete = (
                _accumulate_token_usage(
                    prompt_tokens,
                    completion_tokens,
                    token_usage_complete,
                    write_audit,
                )
            )
            accepted_memories, dropped_by_tier = _apply_write_quotas(
                write_decision,
                config,
            )
            proposals = [
                _write_proposal(
                    memory,
                    task_id,
                    context,
                    final_reward,
                    selected_ids,
                    tools=start["tools"],
                    trajectory=trajectory,
                    memory_policy=memory_policy,
                )
                for memory in accepted_memories
            ]
            _emit(
                context,
                task_id,
                "MemoryWriteProposed",
                proposals=[proposal["evidence"] for proposal in proposals],
                repair_used=write_audit["repaired_output"] is not None,
                invalid_output_skipped=write_audit["fallback_used"],
                sampling_params=write_audit["sampling_params"],
                latency_s=write_audit["latency_s"],
                prompt_tokens=write_audit["prompt_tokens"],
                completion_tokens=write_audit["completion_tokens"],
                dropped_by_tier=dropped_by_tier,
                outcome_class=memory_policy.outcome_class.value,
                polarity=memory_policy.polarity.value,
            )
            written_ids, replayed_ids = _persist_proposals(repository, proposals)
            _emit(
                context,
                task_id,
                "MemoryWriteCommitted",
                written_memory_ids=list(written_ids),
                replayed_memory_ids=list(replayed_ids),
            )

    return EpisodeResult(
        task_id=task_id,
        final_reward=final_reward,
        steps=len(decisions),
        terminal_evaluation=terminal_evaluation,
        simulation_result=simulation_result,
        selected_memory_ids=selected_ids,
        written_memory_ids=written_ids,
        truncated=project_truncated,
        parse_error_count=parse_error_count,
        response_parse_error_count=response_parse_error_count,
        response_count=response_count,
        completed=True,
        project_truncated=project_truncated,
        agent_prompt_tokens=prompt_tokens if token_usage_complete else None,
        agent_completion_tokens=completion_tokens if token_usage_complete else None,
    )


def _accumulate_marker_usage(
    prompt_total: int,
    completion_total: int,
    complete: bool,
    marker: Mapping[str, Any],
) -> tuple[int, int, bool]:
    prompt = marker["prompt_tokens"]
    completion = marker["completion_tokens"]
    if prompt is None or completion is None:
        return prompt_total, completion_total, False
    return prompt_total + int(prompt), completion_total + int(completion), complete


def _final_reward(simulation: Any) -> float:
    reward_info = getattr(simulation, "reward_info", None)
    reward = getattr(reward_info, "reward", None)
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise RuntimeError("Tau2 simulation has no terminal reward")
    return float(reward)


def _model_dump_mapping(value: Any) -> dict[str, Any]:
    if value is None or not callable(getattr(value, "model_dump", None)):
        raise RuntimeError("Tau2 terminal artifact is unavailable")
    dumped = value.model_dump(mode="json")
    if not isinstance(dumped, Mapping):
        raise RuntimeError("Tau2 terminal artifact must be a mapping")
    return sanitize_artifact_data(dict(dumped))


def _is_infrastructure_failure(simulation: Any) -> bool:
    reason = getattr(getattr(simulation, "termination_reason", None), "value", None)
    reason = str(reason or getattr(simulation, "termination_reason", ""))
    return reason == "infrastructure_error" or simulation.reward_info is None


def _simulation_error_type(simulation: Any) -> str:
    info = getattr(simulation, "info", None)
    if isinstance(info, Mapping) and isinstance(info.get("error_type"), str):
        return info["error_type"]
    return "InfrastructureError"


def _emit(
    context: RunContext,
    task_id: str,
    event_type: str,
    **payload: Any,
) -> None:
    context.event_writer.append(
        sanitize_artifact_data(context.event(event_type, task_id, **payload))
    )


def _emit_task_failure(
    context: RunContext,
    task_id: str,
    *,
    stage: str,
    discarded_memory_ids: Sequence[str] = (),
) -> None:
    _emit(
        context,
        task_id,
        "TaskFailed",
        error={"type": "Tau2BatchFailure", "message": "operation failed"},
        stage=stage,
        discarded_memory_ids=list(discarded_memory_ids),
    )


def _memory_ids(repository: MemoryView) -> set[str] | None:
    if not isinstance(repository, MemoryRepository):
        return None
    return {item.id for item in repository.list(status=None)}


def _discard_new_memories(
    repository: MemoryView,
    before: set[str] | None,
    *,
    updated_round: int,
) -> tuple[str, ...]:
    if not isinstance(repository, MemoryRepository) or before is None:
        return ()
    new_ids = tuple(
        sorted(item.id for item in repository.list(status=None) if item.id not in before)
    )
    for memory_id in new_ids:
        repository.update_status(
            memory_id,
            MemoryStatus.RETIRED,
            updated_round=updated_round,
        )
    return new_ids
