from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.tier_contracts import (
    SkillPayload,
    SkillStep,
    TipPayload,
    ToolCallExample,
    ToolPayload,
    TrajectoryDraftPayload,
    materialize_rule_trajectory_memory,
    materialize_tier_memory,
    render_tier_payload,
    validate_tool_payload_against_tools,
)
from tau3_evolver.memory.types import MemoryItem, MemoryTier, stable_memory_id


TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "include_history": {"type": "boolean"},
                },
                "required": ["order_id"],
            },
        },
    },
)


def test_tip_is_one_atomic_rule_and_renders_deterministically() -> None:
    payload = TipPayload(
        condition="Before mutating an order",
        guidance="Confirm the order ID with the customer.",
        rationale="This prevents changing the wrong order.",
        scope=("returns",),
    )

    assert render_tier_payload(MemoryTier.TIP, payload) == (
        "Condition: Before mutating an order\n"
        "Guidance: Confirm the order ID with the customer.\n"
        "Rationale: This prevents changing the wrong order."
    )
    with pytest.raises(ValidationError, match="atomic rule"):
        TipPayload(guidance="1. Look up the order.\n2. Submit the return.")


def test_optional_payload_text_normalizes_blank_values_to_none() -> None:
    tip = TipPayload(
        condition="",
        guidance="Confirm before changing an order.",
        rationale=" ",
    )
    step = SkillStep(order=1, instruction="Look up the order.", success_signal="")

    assert tip.condition is None
    assert tip.rationale is None
    assert step.success_signal is None


def test_skill_requires_a_contiguous_multi_step_workflow() -> None:
    payload = SkillPayload(
        goal="Complete an eligible return",
        preconditions=("The customer supplied an order ID.",),
        steps=(
            SkillStep(order=1, instruction="Look up the order."),
            SkillStep(order=2, instruction="Verify item eligibility."),
        ),
        success_condition="The eligible return is submitted.",
    )

    assert "1. Look up the order." in render_tier_payload(MemoryTier.SKILL, payload)
    with pytest.raises(ValidationError, match="at least 2 items"):
        SkillPayload(
            goal="Complete a return",
            steps=(SkillStep(order=1, instruction="Submit the return."),),
            success_condition="The return is submitted.",
        )
    with pytest.raises(ValidationError, match="contiguous"):
        SkillPayload(
            goal="Complete a return",
            steps=(
                SkillStep(order=1, instruction="Look up the order."),
                SkillStep(order=3, instruction="Submit the return."),
            ),
            success_condition="The return is submitted.",
        )


def test_tool_must_reference_one_available_tool_and_declared_arguments() -> None:
    payload = ToolPayload(
        tool_name="lookup_order",
        purpose="Read the current order state before a mutation.",
        method="Call lookup_order once with the exact customer-supplied order ID.",
        preconditions=("The customer supplied an order ID.",),
        argument_rules={"order_id": "Use the exact order ID."},
        expected_effect="The current order record is returned.",
        example=ToolCallExample(
            name="lookup_order",
            arguments={"order_id": "<customer_order_id>"},
        ),
    )

    assert validate_tool_payload_against_tools(payload, TOOLS) == payload
    with pytest.raises(ValueError, match="unavailable environment tool"):
        validate_tool_payload_against_tools(
            payload.model_copy(update={"tool_name": "invented_tool"}),
            TOOLS,
        )
    with pytest.raises(ValueError, match="undeclared arguments"):
        validate_tool_payload_against_tools(
            payload.model_copy(
                update={"argument_rules": {"order_id": "Exact ID.", "secret": "No."}}
            ),
            TOOLS,
        )
    with pytest.raises(ValueError, match="omits required argument rules"):
        validate_tool_payload_against_tools(
            payload.model_copy(update={"argument_rules": {"include_history": "Optional."}}),
            TOOLS,
        )


def test_tool_contract_supports_real_zero_argument_tools() -> None:
    payload = ToolPayload(
        tool_name="list_categories",
        purpose="List the supported retail categories.",
        method="Call list_categories without arguments and use the returned list.",
        preconditions=("A category choice is required.",),
        expected_effect="The supported categories are returned.",
    )
    tools = (
        {
            "type": "function",
            "function": {
                "name": "list_categories",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    )

    assert validate_tool_payload_against_tools(payload, tools) == payload


def test_trajectory_provenance_and_outcome_are_derived_from_the_episode() -> None:
    materialized = materialize_tier_memory(
        tier=MemoryTier.TRAJECTORY,
        payload=TrajectoryDraftPayload(
            initial_state="The customer asks to return an item.",
            lesson="Verify eligibility before submitting the return.",
        ),
        retrieval_text=None,
        tools=TOOLS,
        run_id="run-7",
        task_id="task-3",
        task_group="retail:return",
        final_reward=0.8,
        trajectory=(
            {
                "action": "lookup_order(order_id='A1')",
                "reward": 0.8,
                "done": True,
            },
        ),
    )

    assert materialized.payload["source_episode_id"] == "run-7:task-3"
    assert materialized.payload["task_group"] == "retail:return"
    assert materialized.payload["final_reward"] == 0.8
    assert materialized.payload["result"] == "partial"
    assert materialized.payload["steps"][0]["order"] == 1
    assert materialized.payload["steps"][0]["action"] == "lookup_order(order_id='A1')"
    assert materialized.payload["steps"][0]["reward"] == 0.8
    assert materialized.payload["steps"][0]["done"] is True


def test_rule_trajectory_is_materialized_from_observed_rollout_without_llm_draft() -> None:
    materialized = materialize_rule_trajectory_memory(
        task_instruction="Look up order A1.",
        run_id="run-8",
        task_id="task-4",
        task_group="retail",
        final_reward=1.0,
        outcome_class="success",
        trajectory=(
            {
                "observation": "Customer asks about order A1.",
                "action": '{"arguments":{"order_id":"A1"},"name":"lookup_order"}',
                "next_observation": "Order A1 is delivered.",
                "reward": 1.0,
                "done": True,
                "terminated": True,
                "truncated": False,
            },
        ),
    )

    payload = materialized.payload
    assert payload["source_episode_id"] == "run-8:task-4"
    assert payload["task_instruction"] == "Look up order A1."
    assert payload["lesson"] is None
    assert payload["steps"][0]["action_name"] == "lookup_order"
    assert payload["steps"][0]["action_arguments"] == {"order_id": "A1"}
    assert payload["steps"][0]["observation"] == "Customer asks about order A1."
    assert payload["steps"][0]["result"] == "Order A1 is delivered."
    assert materialized.classification_rule == "trajectory-runtime-record-v2"


def test_v2_memory_persists_payload_and_rejects_content_drift(tmp_path: Path) -> None:
    payload = TipPayload(guidance="Confirm the order ID before a mutation.")
    content = render_tier_payload(MemoryTier.TIP, payload)
    repository = MemoryRepository(tmp_path / "memory")

    item = repository.add(
        tier=MemoryTier.TIP,
        tier_schema_version=2,
        payload=payload.model_dump(mode="json"),
        content=content,
        source_task_ids=("task-1",),
        created_round=0,
    )
    reopened = MemoryRepository(tmp_path / "memory").get(item.id)

    assert reopened is not None
    assert reopened.tier_schema_version == 2
    assert reopened.payload == payload.model_dump(mode="json")
    invalid = item.model_dump(mode="python")
    invalid["content"] = "Guidance: A different rule."
    with pytest.raises(ValidationError, match="does not match"):
        MemoryItem.model_validate(invalid)
    assert item.id == stable_memory_id(MemoryTier.TIP, content)
