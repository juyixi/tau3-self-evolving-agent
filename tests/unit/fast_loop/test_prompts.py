from __future__ import annotations

import json
import math

import pytest

from tau3_retail_evolver.fast_loop.prompts import (
    LifecyclePrompt,
    MAX_PROMPT_MEMORY_CONTENT_CHARS,
    build_action_prompt,
    build_maintenance_prompt,
    build_selection_prompt,
    build_write_prompt,
)
from tau3_retail_evolver.memory.retrieval import MemoryCandidate
from tau3_retail_evolver.memory.types import MemoryItem, MemoryTier


def _candidate() -> MemoryCandidate:
    item = MemoryItem(
        id="memory-1",
        tier=MemoryTier.TIP,
        content="Confirm the order before changing it.",
        retrieval_text="confirm order change",
        embedding=(0.1, 0.2),
        embedding_model_revision="embedding-v1",
        metadata={"attribution_score": 0.99, "private": "do not prompt"},
        source_task_ids=("hidden-task",),
        created_round=1,
        updated_round=1,
        version=2,
    )
    return MemoryCandidate(
        memory_id=item.id,
        memory_version=item.version,
        tier=item.tier,
        rank=1,
        similarity=0.8,
        retriever_revision="embedding-v1",
        query_hash="private-query-hash",
        item=item,
    )


def _public_context() -> dict[str, object]:
    return {
        "task_instruction": "Help the customer with their order.",
        "policy": "Follow the official return policy.",
        "tools": [{"type": "function", "function": {"name": "lookup_order"}}],
        "observation": "I need to update my address.",
        "history": [{"role": "user", "content": "My order is 123."}],
    }


class _Tau2Tool:
    @property
    def openai_schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object"},
            },
        }


def test_public_context_normalizes_official_tau2_tool_objects() -> None:
    context = _public_context()
    context["tools"] = [_Tau2Tool()]

    prompt = build_action_prompt(**context, memories=())

    assert prompt.payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object"},
            },
        }
    ]


def _diagnostic_item(index: int = 1, tier: str = "tip") -> dict[str, object]:
    return {
        "id": f"memory-{index}",
        "tier": tier,
        "content": "Confirm the order before changing it.",
        "version": 1,
        "status": "active",
    }


def _maintenance_diagnostics() -> dict[str, object]:
    return {
        tier: {"items": [_diagnostic_item(tier=tier)]}
        for tier in ("trajectory", "tip", "skill", "tool")
    }


def test_selection_and_action_prompts_project_only_public_memory_data() -> None:
    selection = build_selection_prompt(**_public_context(), candidates=(_candidate(),))
    action = build_action_prompt(**_public_context(), memories=(_candidate(),))

    for prompt in (selection, action):
        serialized = prompt.model_dump_json()
        assert "memory-1" in serialized
        assert "attribution_score" not in serialized
        assert "private-query-hash" not in serialized
        assert "embedding-v1" not in serialized
        assert "hidden-task" not in serialized
        assert "do not prompt" not in serialized
        assert prompt.payload["memories"][0] == {
            "id": "memory-1",
            "tier": "tip",
            "content": "Confirm the order before changing it.",
            "version": 2,
            "rank": 1,
            "similarity": 0.8,
            "polarity": "positive",
            "outcome_class": "success",
        }


def test_action_prompt_marks_legacy_failed_tip_as_caution() -> None:
    item = MemoryItem(
        id="failed-tip",
        tier=MemoryTier.TIP,
        content="Avoid transferring a supported request.",
        retrieval_text="premature transfer",
        metadata={"source_final_reward": 0.0},
        source_task_ids=("failed-task",),
        created_round=0,
        updated_round=0,
    )

    prompt = build_action_prompt(**_public_context(), memories=(item,))

    assert prompt.payload["memories"][0]["polarity"] == "caution"
    assert prompt.payload["memories"][0]["outcome_class"] == "task_failure"


def test_memory_content_is_bounded_in_model_prompts() -> None:
    candidate = _candidate().item
    oversized = MemoryItem(
        id=candidate.id,
        tier=candidate.tier,
        content="x" * (MAX_PROMPT_MEMORY_CONTENT_CHARS + 1),
        retrieval_text=candidate.retrieval_text,
        embedding=candidate.embedding,
        embedding_model_revision=candidate.embedding_model_revision,
        source_task_ids=candidate.source_task_ids,
        created_round=candidate.created_round,
        updated_round=candidate.updated_round,
        version=candidate.version,
    )

    prompt = build_selection_prompt(**_public_context(), candidates=(oversized,))

    assert prompt.payload["memories"][0]["content"] == "x" * MAX_PROMPT_MEMORY_CONTENT_CHARS


def test_action_prompt_can_exclude_memory_context() -> None:
    prompt = build_action_prompt(
        **_public_context(),
        memories=(_candidate(),),
        include_memory_context=False,
    )

    assert "memories" not in prompt.payload


def test_write_prompt_allows_terminal_public_evaluation_but_rejects_hidden_criteria() -> None:
    prompt = build_write_prompt(
        **_public_context(),
        trajectory=[{"role": "assistant", "content": "Address updated."}],
        terminal_evaluation={"reward": 1.0, "official_result": "success"},
    )

    assert prompt.payload["trajectory"] == [{"role": "assistant", "content": "Address updated."}]
    assert "trajectory_format" not in prompt.payload
    assert prompt.payload["terminal_evaluation"] == {"official_result": "success", "reward": 1.0}
    with pytest.raises(ValueError, match="hidden"):
        build_write_prompt(
            **_public_context(),
            trajectory=[],
            terminal_evaluation={"evaluation_criteria": "private rubric"},
        )


def test_write_prompt_includes_outcome_write_contract() -> None:
    memory_outcome = {
        "final_reward": 0.0,
        "outcome_class": "task_failure",
        "polarity": "caution",
        "allowed_tiers": ["tip", "trajectory"],
    }

    prompt = build_write_prompt(
        **_public_context(),
        trajectory=[{"role": "assistant", "content": "Transferred too early."}],
        terminal_evaluation={"reward": 0.0},
        memory_outcome=memory_outcome,
    )

    assert prompt.payload["memory_outcome"] == memory_outcome


def test_write_prompt_compacts_cumulative_transcript_without_losing_step_metadata() -> None:
    initial = "user: Please exchange order 123."
    after_lookup = (
        f"{initial}\n"
        "assistant: lookup_order(order_id='123')\n"
        "tool: delivered"
    )
    final = (
        f"{after_lookup}\n"
        "assistant: exchange_order(order_id='123')\n"
        "tool: exchange requested"
    )
    context = _public_context()
    context["observation"] = final

    prompt = build_write_prompt(
        **context,
        trajectory=[
            {
                "observation": initial,
                "action": "lookup_order(order_id='123')",
                "next_observation": after_lookup,
                "reward": 0.0,
                "done": False,
            },
            {
                "observation": after_lookup,
                "action": "exchange_order(order_id='123')",
                "next_observation": final,
                "reward": 1.0,
                "done": False,
            },
            {
                "observation": final,
                "action": "done()",
                "next_observation": "",
                "reward": 1.0,
                "done": True,
            },
        ],
        terminal_evaluation={"reward": 1.0},
    )

    assert prompt.payload["observation"] == final
    assert prompt.payload["trajectory_format"] == "final_observation_plus_actions_v1"
    assert prompt.payload["trajectory"] == [
        {
            "turn": 0,
            "action": "lookup_order(order_id='123')",
            "reward": 0.0,
            "done": False,
        },
        {
            "turn": 1,
            "action": "exchange_order(order_id='123')",
            "reward": 1.0,
            "done": False,
        },
        {
            "turn": 2,
            "action": "done()",
            "reward": 1.0,
            "done": True,
        },
    ]


def test_write_prompt_keeps_non_cumulative_trajectory_unchanged() -> None:
    context = _public_context()
    trajectory = [
        {
            "observation": "state one",
            "action": "lookup_order(order_id='123')",
            "next_observation": "independent state two",
            "reward": 0.0,
        }
    ]

    prompt = build_write_prompt(
        **context,
        trajectory=trajectory,
        terminal_evaluation={"reward": 0.0},
    )

    assert prompt.payload["trajectory"] == trajectory
    assert "trajectory_format" not in prompt.payload


def test_maintenance_prompt_keeps_only_caller_diagnostics_and_command_schemas() -> None:
    diagnostics = _maintenance_diagnostics()

    prompt = build_maintenance_prompt(diagnostics=diagnostics)

    assert prompt.payload == {"diagnostics": diagnostics}
    assert {schema["operation"] for schema in prompt.command_schemas} == {
        "lookup",
        "merge",
        "delete",
    }


def test_maintenance_prompt_requires_exactly_four_tier_item_lists() -> None:
    diagnostics = _maintenance_diagnostics()
    del diagnostics["tool"]

    with pytest.raises(ValueError):
        build_maintenance_prompt(diagnostics=diagnostics)


def test_maintenance_prompt_rejects_unknown_item_fields() -> None:
    diagnostics = _maintenance_diagnostics()
    diagnostics["tip"]["items"][0]["arbitrary"] = "hidden"

    with pytest.raises(ValueError):
        build_maintenance_prompt(diagnostics=diagnostics)


@pytest.mark.parametrize("private_field", ["usage_count", "success_count", "last_used"])
def test_maintenance_prompt_rejects_private_usage_fields(private_field: str) -> None:
    diagnostics = _maintenance_diagnostics()
    diagnostics["tip"]["items"][0][private_field] = 1

    with pytest.raises(ValueError):
        build_maintenance_prompt(diagnostics=diagnostics)


def test_maintenance_prompt_rejects_unknown_top_level_fields() -> None:
    diagnostics = _maintenance_diagnostics()
    diagnostics["summary"] = {"items": []}

    with pytest.raises(ValueError):
        build_maintenance_prompt(diagnostics=diagnostics)


def test_maintenance_prompt_rejects_tier_lists_over_the_bound() -> None:
    diagnostics = _maintenance_diagnostics()
    diagnostics["tip"]["items"] = [_diagnostic_item(index) for index in range(101)]

    with pytest.raises(ValueError, match="at most"):
        build_maintenance_prompt(diagnostics=diagnostics)


def test_lifecycle_prompt_is_json_safe_and_rejects_non_json_payloads() -> None:
    prompt = LifecyclePrompt(
        kind="selection",
        payload={"observation": "hello"},
        command_schemas=(),
    )

    assert json.loads(prompt.model_dump_json()) == prompt.model_dump(mode="json")
    with pytest.raises(ValueError, match="JSON"):
        LifecyclePrompt(
            kind="selection",
            payload={"similarity": math.nan},
            command_schemas=(),
        )


def test_lifecycle_prompt_scans_json_safe_nested_tuples_for_hidden_fields() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        LifecyclePrompt(
            kind="selection",
            payload={"policy": ({"Attribution-Score": 0.9},)},
            command_schemas=(),
        )


def test_public_prompt_rejects_forbidden_evaluation_and_metadata_fields() -> None:
    context = _public_context()
    context["policy"] = {"official": "public", "evaluation_criteria": "private"}

    with pytest.raises(ValueError, match="forbidden"):
        build_selection_prompt(**context, candidates=())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("policy", {"outer": {"attributionScore": 0.9}}),
        ("tools", [{"function": {"TEST-TASK-ID": "secret"}}]),
    ),
)
def test_public_context_rejects_nested_forbidden_key_variants(
    field: str, value: object
) -> None:
    context = _public_context()
    context[field] = value

    with pytest.raises(ValueError, match="forbidden"):
        build_selection_prompt(**context, candidates=())


def test_history_uses_an_explicit_role_and_content_whitelist() -> None:
    context = _public_context()
    context["history"] = [{"role": "user", "content": "Hello", "testTaskId": "secret"}]

    with pytest.raises(ValueError, match="history"):
        build_action_prompt(**context, memories=())


@pytest.mark.parametrize(
    ("trajectory", "terminal_evaluation"),
    (
        (
            [{"role": "assistant", "content": "done", "privileged-hindsight": True}],
            {"reward": 1.0},
        ),
        (
            [{"role": "assistant", "content": "done"}],
            {"official": {"evaluatorMetadata": "secret"}},
        ),
    ),
)
def test_write_prompt_rejects_nested_forbidden_key_variants(
    trajectory: list[dict[str, object]], terminal_evaluation: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        build_write_prompt(
            **_public_context(),
            trajectory=trajectory,
            terminal_evaluation=terminal_evaluation,
        )


def test_maintenance_diagnostics_reject_nested_forbidden_key_variants() -> None:
    diagnostics = _maintenance_diagnostics()
    diagnostics["tip"]["items"][0]["attribution.Score"] = 0.9

    with pytest.raises(ValueError, match="forbidden"):
        build_maintenance_prompt(diagnostics=diagnostics)
