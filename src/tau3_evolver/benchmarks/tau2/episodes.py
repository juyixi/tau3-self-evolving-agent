from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from tau3_evolver.benchmarks.tau2.agent import audit_marker, render_observation
from tau3_evolver.fast_loop.context import LifecycleContext
from tau3_evolver.fast_loop.contracts import (
    EpisodeResult,
    FastLoopPolicy,
    PendingEpisode,
    WriteDecision,
)
from tau3_evolver.fast_loop.decision import (
    accumulate_token_usage,
    generate_decision,
)
from tau3_evolver.fast_loop.prompts import build_write_prompt
from tau3_evolver.fast_loop.settings import FastLoopConfig
from tau3_evolver.fast_loop.writing import (
    apply_write_quotas,
    empty_write_decision,
    validate_write_decision,
)
from tau3_evolver.fast_loop.outcomes import EpisodeMemoryPolicy, classify_episode_memory
from tau3_evolver.memory.tier_contracts import (
    TIER_SCHEMA_VERSION,
    MaterializedTierMemory,
    materialize_rule_trajectory_memory,
    materialize_tier_memory,
)
from tau3_evolver.memory.types import stable_memory_id
from tau3_evolver.security.redaction import redact_public_data


def finalize_simulation(
    *,
    runtime: Any,
    simulation: Any,
    policy: FastLoopPolicy,
    config: FastLoopConfig,
    context: LifecycleContext,
    propose_experience: bool,
) -> PendingEpisode:
    messages = list(simulation.messages or ())
    decisions = [
        (index, message, message.raw_data["tau3_agent"])
        for index, message in enumerate(messages)
        if isinstance(message, runtime.assistant_message_type)
        and isinstance(getattr(message, "raw_data", None), Mapping)
        and isinstance(message.raw_data.get("tau3_agent"), Mapping)
    ]
    if not decisions:
        raise RuntimeError("Tau2 simulation contains no Tau3 Agent decisions")
    start = decisions[0][2].get("start")
    if not isinstance(start, Mapping):
        raise RuntimeError("Tau2 simulation is missing Tau3 Agent start evidence")
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
            incompatible_tool_memories=start["incompatible_tool_memories"],
            incompatible_tool_memory_count=len(start["incompatible_tool_memories"]),
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
        prompt_tokens, completion_tokens, token_usage_complete = _accumulate_usage(
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
        observation = str(marker["observation"])
        next_observation = (
            str(decisions[position + 1][2]["observation"])
            if position + 1 < len(decisions)
            else render_observation(messages)
        )
        is_last = position == len(decisions) - 1
        public_info: dict[str, Any] = {}
        next_message_index = (
            decisions[position + 1][0]
            if position + 1 < len(decisions)
            else len(messages)
        )
        if any(
            isinstance(item, runtime.tool_message_type)
            and bool(getattr(item, "error", False))
            for item in messages[message_index + 1 : next_message_index]
        ):
            public_info["parse_error"] = "tau2_tool_error"
            parse_error_count += 1
        audit = marker["action_audit"]
        response_count += 1
        response_parse_error_count += int(audit["parse_failed"])
        prompt_tokens, completion_tokens, token_usage_complete = _accumulate_usage(
            prompt_tokens,
            completion_tokens,
            token_usage_complete,
            audit,
        )
        _emit(
            context,
            task_id,
            "DecisionMade",
            turn=position,
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
            turn=position,
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
        truncated=project_truncated,
        project_truncated=project_truncated,
    )

    proposals: tuple[dict[str, Any], ...] = ()
    if propose_experience:
        memory_policy = classify_episode_memory(
            final_reward=final_reward,
            terminal_evaluation=terminal_evaluation,
            truncated=project_truncated,
        )
        trajectory_proposal = _trajectory_proposal(
            task_instruction=str(start["task_instruction"]),
            task_id=task_id,
            context=context,
            final_reward=final_reward,
            selected_ids=selected_ids,
            trajectory=trajectory,
            memory_policy=memory_policy,
        )
        learned: list[dict[str, Any]] = []
        dropped_by_tier: dict[str, int] = {}
        write_audit: dict[str, Any] = {
            "repaired_output": None,
            "error": None,
            "fallback_used": False,
            "sampling_params": {},
            "latency_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        if memory_policy.should_generate:
            write_prompt = build_write_prompt(
                task_instruction=str(start["task_instruction"]),
                policy=start["policy"],
                tools=start["tools"],
                observation=last_observation,
                trajectory=trajectory,
                terminal_evaluation=terminal_evaluation,
                memory_outcome=memory_policy.prompt_payload(),
            )
            write_decision, write_audit = generate_decision(
                policy,
                write_prompt,
                WriteDecision,
                validator=lambda decision: validate_write_decision(
                    decision,
                    tools=start["tools"],
                    task_id=task_id,
                    context=context,
                    final_reward=final_reward,
                    trajectory=trajectory,
                    memory_policy=memory_policy,
                ),
                label="write",
                invalid_fallback=empty_write_decision,
            )
            response_count += 1
            response_parse_error_count += int(write_audit["error"] is not None)
            prompt_tokens, completion_tokens, token_usage_complete = accumulate_token_usage(
                prompt_tokens,
                completion_tokens,
                token_usage_complete,
                write_audit,
            )
            accepted, dropped_by_tier = apply_write_quotas(write_decision, config)
            learned = [
                _learned_proposal(
                    memory,
                    task_id=task_id,
                    context=context,
                    final_reward=final_reward,
                    selected_ids=selected_ids,
                    tools=start["tools"],
                    trajectory=trajectory,
                    memory_policy=memory_policy,
                )
                for memory in accepted
            ]
        proposals = (trajectory_proposal, *learned)
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
            polarity=(
                memory_policy.polarity.value
                if memory_policy.polarity is not None
                else None
            ),
            skipped_reason=memory_policy.skip_reason,
        )

    return PendingEpisode(
        result=EpisodeResult(
            task_id=task_id,
            final_reward=final_reward,
            steps=len(decisions),
            terminal_evaluation=terminal_evaluation,
            selected_memory_ids=selected_ids,
            written_memory_ids=(),
            truncated=project_truncated,
            parse_error_count=parse_error_count,
            response_parse_error_count=response_parse_error_count,
            response_count=response_count,
            completed=True,
            project_truncated=project_truncated,
            agent_prompt_tokens=prompt_tokens if token_usage_complete else None,
            agent_completion_tokens=(
                completion_tokens if token_usage_complete else None
            ),
        ),
        proposals=proposals,
    )


def _trajectory_proposal(
    *,
    task_instruction: str,
    task_id: str,
    context: LifecycleContext,
    final_reward: float,
    selected_ids: tuple[str, ...],
    trajectory: Sequence[Mapping[str, Any]],
    memory_policy: EpisodeMemoryPolicy,
) -> dict[str, Any]:
    materialized = materialize_rule_trajectory_memory(
        task_instruction=task_instruction,
        run_id=context.run_id,
        task_id=task_id,
        task_group=context.task_group_for(task_id),
        final_reward=final_reward,
        outcome_class=memory_policy.outcome_class.value,
        trajectory=trajectory,
    )
    return _proposal(
        materialized,
        task_id=task_id,
        context=context,
        metadata={
            "source_run_id": context.run_id,
            "source_final_reward": final_reward,
            "selected_memory_ids": list(selected_ids),
            "classification_rule": materialized.classification_rule,
            "polarity": (
                memory_policy.polarity.value
                if memory_policy.polarity is not None
                else "caution"
            ),
            "outcome_class": memory_policy.outcome_class.value,
            "generation_mode": "rule",
        },
        generation_mode="rule",
    )


def _learned_proposal(
    memory: Any,
    *,
    task_id: str,
    context: LifecycleContext,
    final_reward: float,
    selected_ids: tuple[str, ...],
    tools: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    memory_policy: EpisodeMemoryPolicy,
) -> dict[str, Any]:
    materialized = materialize_tier_memory(
        tier=memory.tier,
        payload=memory.payload,
        retrieval_text=memory.retrieval_text,
        tools=tools,
        run_id=context.run_id,
        task_id=task_id,
        task_group=context.task_group_for(task_id),
        final_reward=final_reward,
        trajectory=trajectory,
    )
    metadata = dict(memory.metadata)
    metadata.update(
        source_run_id=context.run_id,
        source_final_reward=final_reward,
        selected_memory_ids=list(selected_ids),
        classification_rule=materialized.classification_rule,
        polarity=memory_policy.polarity.value,
        outcome_class=memory_policy.outcome_class.value,
    )
    return _proposal(
        materialized,
        task_id=task_id,
        context=context,
        metadata=metadata,
        generation_mode="llm",
    )


def _proposal(
    materialized: MaterializedTierMemory,
    *,
    task_id: str,
    context: LifecycleContext,
    metadata: Mapping[str, Any],
    generation_mode: str,
) -> dict[str, Any]:
    memory_id = stable_memory_id(materialized.tier, materialized.content)
    add_kwargs = {
        "tier": materialized.tier,
        "tier_schema_version": TIER_SCHEMA_VERSION,
        "payload": materialized.payload,
        "content": materialized.content,
        "source_task_ids": (task_id,),
        "created_round": context.memory_generation,
        "metadata": dict(metadata),
        "retrieval_text": materialized.retrieval_text,
    }
    return {
        "memory_id": memory_id,
        "add_kwargs": add_kwargs,
        "evidence": {
            "memory_id": memory_id,
            "generation_mode": generation_mode,
            "tier": materialized.tier.value,
            "tier_schema_version": TIER_SCHEMA_VERSION,
            "payload": materialized.payload,
            "content": materialized.content,
            "retrieval_text": materialized.retrieval_text,
            "metadata": dict(metadata),
            "source_task_ids": [task_id],
            "memory_generation": context.memory_generation,
        },
    }


def _emit(
    context: LifecycleContext, task_id: str, event_type: str, **payload: Any
) -> None:
    context.event_writer.append(
        redact_public_data(context.event(event_type, task_id, **payload))
    )


def _accumulate_usage(
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
    reward = getattr(getattr(simulation, "reward_info", None), "reward", None)
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise RuntimeError("Tau2 simulation has no terminal reward")
    return float(reward)


def _model_dump_mapping(value: Any) -> dict[str, Any]:
    if value is None or not callable(getattr(value, "model_dump", None)):
        raise RuntimeError("Tau2 terminal artifact is unavailable")
    dumped = value.model_dump(mode="json")
    if not isinstance(dumped, Mapping):
        raise RuntimeError("Tau2 terminal artifact must be a mapping")
    return redact_public_data(dict(dumped))
