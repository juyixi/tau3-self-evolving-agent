from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import threading
from typing import Any

from tau3_evolver.agent.actions import assistant_for_action
from tau3_evolver.agent.state import Tau3AgentState
from tau3_evolver.benchmarks.tau2.runtime import Tau2RuntimeBinding
from tau3_evolver.agent.action_codec import TAU2_STOP_ACTION
from tau3_evolver.agent.decisions import ActionDecision, SelectionDecision
from tau3_evolver.agent.prompts import (
    build_action_prompt,
    build_selection_prompt,
    project_public_context,
)
from tau3_evolver.agent.policy import (
    FastLoopConfig,
    FastLoopPolicy,
    _candidate_evidence,
    _generate_decision,
    _query_hash,
    _retrieval_query,
    _validate_selection_limits,
)
from tau3_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.retrieval import MemoryCandidate, Retriever
from tau3_evolver.memory.tier_contracts import (
    ToolPayload,
    validate_stored_tier_payload,
    validate_tool_payload_against_tools,
)
from tau3_evolver.memory.types import MemoryTier


MemoryView = MemoryRepository | ReadOnlyMemoryRepository | None


def create_tau3_agent_factory(
    *,
    runtime: Tau2RuntimeBinding,
    benchmark: str,
    policy: FastLoopPolicy,
    repository: MemoryView,
    retriever: Retriever | None,
    config: FastLoopConfig,
    memory_source_namespace: str | None,
    cross_domain_memory: bool,
    retrieval_lock: threading.Lock | None = None,
) -> Callable[..., Any]:
    """Create the per-task Tau3 Agent factory registered with Tau2."""

    if config.memory_enabled and (repository is None or retriever is None):
        raise ValueError("memory-enabled Agent requires a repository and Retriever")
    if not config.memory_enabled and (repository is not None or retriever is not None):
        raise ValueError("memory-disabled Agent must not receive Memory dependencies")

    lock = retrieval_lock or threading.Lock()
    assistant_type = runtime.assistant_message_type
    tool_call_type = runtime.tool_call_type
    multi_tool_type = runtime.multi_tool_message_type

    class Tau3Agent(runtime.half_duplex_agent_type):
        def __init__(
            self,
            tools: list[Any],
            domain_policy: str,
            task: Any | None = None,
            **kwargs: Any,
        ) -> None:
            del kwargs
            super().__init__(tools=tools, domain_policy=domain_policy)
            self._task = task
            self._public_tools = [deepcopy(tool.openai_schema) for tool in tools]
            self._seed = 0

        def get_init_state(
            self, message_history: list[Any] | None = None
        ) -> Tau3AgentState:
            return Tau3AgentState(messages=list(message_history or ()))

        def set_seed(self, seed: int) -> None:
            self._seed = seed

        @classmethod
        def is_stop(cls, message: Any) -> bool:
            return TAU2_STOP_ACTION in str(getattr(message, "content", "") or "")

        def generate_next_message(
            self, message: Any, state: Tau3AgentState
        ) -> tuple[Any, Tau3AgentState]:
            if isinstance(message, multi_tool_type):
                state.messages.extend(message.tool_messages)
            elif message is not None:
                state.messages.append(message)

            if state.turn >= config.max_episode_steps:
                stop = assistant_type(role="assistant", content=TAU2_STOP_ACTION)
                state.messages.append(stop)
                return stop, state

            observation = render_observation(state.messages)
            start_marker: dict[str, Any] | None = None
            if not state.started:
                public_context = project_public_context(
                    task_instruction=f"Resolve the user's {benchmark} service request.",
                    policy=self.domain_policy,
                    tools=self._public_tools,
                    observation=observation,
                    history=(),
                )
                start_marker = self._start(public_context, state)
                state.started = True

            action_prompt = build_action_prompt(
                task_instruction=f"Resolve the user's {benchmark} service request.",
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
            marker: dict[str, Any] = {
                "schema_version": 1,
                "turn": state.turn,
                "observation": observation,
                "action": action.action,
                "action_audit": audit_marker(audit),
            }
            if start_marker is not None:
                marker["start"] = start_marker
            assistant = assistant_for_action(
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

        def _start(
            self,
            public_context: Mapping[str, Any],
            state: Tau3AgentState,
        ) -> dict[str, Any]:
            if not config.memory_enabled:
                return {
                    **public_context,
                    "memory_enabled": False,
                    "memory_disabled_reason": "request",
                    "memory_source_namespace": None,
                    "cross_domain_memory": False,
                    "selected": [],
                    "selected_memory_ids": [],
                    "incompatible_tool_memories": [],
                }

            assert repository is not None
            assert retriever is not None
            query = _retrieval_query(
                str(public_context["task_instruction"]),
                public_context["policy"],
                public_context["tools"],
                str(public_context["observation"]),
            )
            with lock:
                candidates = retriever.retrieve(
                    query,
                    repository,
                    top_k=config.retrieve_top_k,
                    tier_quotas=config.retrieval_tier_quotas(),
                    mmr_lambdas=config.retrieval_mmr_lambdas(),
                    global_mmr_lambda=config.retrieval_global_mmr_lambda,
                )
            compatible, filtered = filter_tool_memory_candidates(
                candidates, self._public_tools
            )
            selection_prompt = build_selection_prompt(
                task_instruction=str(public_context["task_instruction"]),
                policy=public_context["policy"],
                tools=public_context["tools"],
                observation=str(public_context["observation"]),
                candidates=compatible,
            )
            selection, audit = _generate_decision(
                policy,
                selection_prompt,
                SelectionDecision,
                candidate_ids=[candidate.memory_id for candidate in compatible],
                validator=lambda decision: _validate_selection_limits(
                    decision, compatible, config
                ),
                label="selection",
            )
            by_id = {candidate.memory_id: candidate for candidate in compatible}
            state.selected = tuple(by_id[memory_id] for memory_id in selection.memory_ids)
            return {
                **public_context,
                "memory_enabled": True,
                "memory_source_namespace": memory_source_namespace,
                "cross_domain_memory": cross_domain_memory,
                "query_hash": compatible[0].query_hash if compatible else _query_hash(query),
                "retriever_revision": (
                    compatible[0].retriever_revision
                    if compatible
                    else retriever.provider.model_revision
                ),
                "candidates": [_candidate_evidence(item) for item in compatible],
                "selected": [_candidate_evidence(item) for item in state.selected],
                "selected_memory_ids": list(selection.memory_ids),
                "selection_audit": audit_marker(audit),
                "incompatible_tool_memories": filtered,
            }

    def factory(
        tools: list[Any], domain_policy: str, task: Any | None = None, **kwargs: Any
    ) -> Any:
        return Tau3Agent(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            **kwargs,
        )

    return factory


def filter_tool_memory_candidates(
    candidates: Sequence[MemoryCandidate],
    tools: Sequence[Mapping[str, Any]],
) -> tuple[list[MemoryCandidate], list[dict[str, str]]]:
    names = {
        str(schema.get("name"))
        for tool in tools
        for schema in [tool.get("function") if isinstance(tool.get("function"), Mapping) else tool]
        if isinstance(schema, Mapping) and isinstance(schema.get("name"), str)
    }
    compatible: list[MemoryCandidate] = []
    filtered: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate.tier is not MemoryTier.TOOL:
            compatible.append(candidate)
            continue
        reason = "invalid_tool_memory"
        try:
            payload = validate_stored_tier_payload(
                MemoryTier.TOOL, candidate.item.payload
            )
            if not isinstance(payload, ToolPayload):
                raise ValueError("invalid tool payload")
            reason = (
                "unavailable_tool"
                if payload.tool_name not in names
                else "incompatible_schema"
            )
            validate_tool_payload_against_tools(payload, tools)
        except (TypeError, ValueError):
            filtered.append(
                {
                    "memory_id": candidate.memory_id,
                    "source_tool": _source_tool_name(candidate),
                    "reason": reason,
                }
            )
            continue
        compatible.append(candidate)
    return compatible, filtered


def _source_tool_name(candidate: MemoryCandidate) -> str:
    payload = candidate.item.payload
    if isinstance(payload, Mapping) and isinstance(payload.get("tool_name"), str):
        return payload["tool_name"]
    return "unknown"


def render_observation(messages: Sequence[Any]) -> str:
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


def audit_marker(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_output_sha256": hashlib.sha256(
            str(audit["raw_output"]).encode("utf-8")
        ).hexdigest(),
        "repaired_output_sha256": (
            hashlib.sha256(str(audit["repaired_output"]).encode("utf-8")).hexdigest()
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
