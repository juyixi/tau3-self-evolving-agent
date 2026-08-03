from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from tau3_retail_evolver.fast_loop.decisions import (
    ActionDecision,
    DecisionParseResult,
    MaintenanceDecision,
    SelectionDecision,
    SkillMemoryWrite,
    TipMemoryWrite,
    WriteDecision,
    parse_decision,
)
from tau3_retail_evolver.memory.operations import DeleteCommand, LookupCommand, MergeCommand
from tau3_retail_evolver.memory.tier_contracts import SkillPayload, SkillStep, TipPayload
from tau3_retail_evolver.memory.types import MemoryTier


def test_selection_decision_requires_unique_nonblank_candidate_ids() -> None:
    decision = SelectionDecision(memory_ids=(" memory-1 ", "memory-2"))

    assert decision.memory_ids == ("memory-1", "memory-2")
    with pytest.raises(ValidationError, match="unique"):
        SelectionDecision(memory_ids=("memory-1", "memory-1"))
    with pytest.raises(ValidationError, match="blank"):
        SelectionDecision(memory_ids=(" ",))


def test_selection_parse_validates_all_ids_against_candidates() -> None:
    result = parse_decision(
        '{"memory_ids": ["memory-1", "not-a-candidate"]}',
        SelectionDecision,
        candidate_ids={"memory-1"},
    )

    assert result.decision is None
    assert "not a candidate" in (result.error or "")


def test_selection_parse_requires_candidate_ids() -> None:
    result = parse_decision('{"memory_ids": ["memory-1"]}', SelectionDecision)

    assert result.decision is None
    assert "candidate_ids is required" in (result.error or "")


def test_selection_parse_accepts_only_ids_from_the_required_candidate_set() -> None:
    result = parse_decision(
        '{"memory_ids": ["memory-2", "memory-1"]}',
        SelectionDecision,
        candidate_ids={"memory-1", "memory-2"},
    )

    assert result.decision == SelectionDecision(memory_ids=("memory-2", "memory-1"))
    assert result.error is None


def test_action_and_write_decisions_validate_text_and_json_metadata() -> None:
    assert ActionDecision(action="  reply  ").action == "reply"
    with pytest.raises(ValidationError, match="blank"):
        ActionDecision(action=" ")

    write = TipMemoryWrite(
        tier=MemoryTier.TIP,
        payload=TipPayload(guidance="Confirm the order before changing it."),
        retrieval_text=None,
        metadata={"source": ["public"]},
    )
    assert WriteDecision(memories=(write,)).memories == (write,)
    with pytest.raises(ValidationError, match="JSON"):
        TipMemoryWrite(
            tier=MemoryTier.TIP,
            payload=TipPayload(guidance="Valid content"),
            retrieval_text="Valid retrieval text",
            metadata={"score": math.nan},
        )
    skill_payload = SkillPayload(
        goal="Complete a return",
        steps=(
            SkillStep(order=1, instruction="Look up the order."),
            SkillStep(order=2, instruction="Verify return eligibility."),
        ),
        success_condition="The return is eligible.",
    )
    with pytest.raises(ValidationError):
        WriteDecision.model_validate(
            {
                "memories": [
                    {
                        "tier": "tip",
                        "payload": skill_payload.model_dump(mode="python"),
                    }
                ]
            }
        )
    assert SkillMemoryWrite(
        tier=MemoryTier.SKILL,
        payload=skill_payload,
    ).tier is MemoryTier.SKILL


def test_write_decision_normalizes_blank_optional_text_from_model_output() -> None:
    result = parse_decision(
        json.dumps(
            {
                "memories": [
                    {
                        "tier": "tip",
                        "payload": {
                            "condition": "",
                            "guidance": "Authenticate before reading account data.",
                            "rationale": " ",
                        },
                        "retrieval_text": "",
                    }
                ]
            }
        ),
        WriteDecision,
    )

    assert result.error is None
    assert result.decision is not None
    memory = result.decision.memories[0]
    assert memory.payload.condition is None
    assert memory.payload.rationale is None
    assert memory.retrieval_text is None


def test_write_decision_schema_discriminates_payloads_by_tier() -> None:
    schema = WriteDecision.model_json_schema()
    memory_schema = schema["properties"]["memories"]["items"]

    assert memory_schema["discriminator"]["propertyName"] == "tier"
    assert set(memory_schema["discriminator"]["mapping"]) == {
        "tip",
        "skill",
        "tool",
    }
    assert len(memory_schema["oneOf"]) == 3


def test_maintenance_decision_parses_only_known_typed_commands() -> None:
    decision = MaintenanceDecision.model_validate(
        {
            "commands": (
                {"operation": "lookup", "memory_ids": ["memory-1"]},
                {
                    "operation": "merge",
                    "source_ids": ["memory-1", "memory-2"],
                    "content": "Merged guidance.",
                    "updated_round": 3,
                },
                {
                    "operation": "delete",
                    "memory_ids": ["memory-3"],
                    "updated_round": 3,
                    "reason": "stale",
                },
            )
        }
    )

    assert isinstance(decision.commands[0], LookupCommand)
    assert isinstance(decision.commands[1], MergeCommand)
    assert isinstance(decision.commands[2], DeleteCommand)
    with pytest.raises(ValidationError):
        MaintenanceDecision.model_validate({"commands": [{"operation": "invent"}]})


def test_maintenance_decision_parses_json_review_arrays() -> None:
    result = parse_decision(
        json.dumps(
            {
                "commands": [],
                "reviews": [
                    {
                        "memory_ids": ["memory-1"],
                        "disposition": "keep",
                        "reason": "Still relevant.",
                    }
                ],
            }
        ),
        MaintenanceDecision,
    )

    assert result.decision is not None
    assert result.decision.reviews[0].memory_ids == ("memory-1",)


def test_maintenance_runtime_fields_may_be_omitted_by_model() -> None:
    result = parse_decision(
        json.dumps(
            {
                "commands": [
                    {
                        "operation": "merge",
                        "source_ids": ["memory-1", "memory-2"],
                        "content": "Consolidated guidance.",
                    },
                    {
                        "operation": "delete",
                        "memory_ids": ["memory-3"],
                        "reason": "Obsolete.",
                    },
                ]
            }
        ),
        MaintenanceDecision,
    )

    assert result.decision is not None
    assert all(command.updated_round == 0 for command in result.decision.commands)


@pytest.mark.parametrize("updated_round", ("3", 3.0, True))
def test_maintenance_parse_rejects_coerced_updated_round_scalars(updated_round: object) -> None:
    result = parse_decision(
        json.dumps(
            {
                "commands": [
                    {
                        "operation": "delete",
                        "memory_ids": ["memory-1"],
                        "updated_round": updated_round,
                        "reason": "stale",
                    }
                ]
            }
        ),
        MaintenanceDecision,
    )

    assert result.decision is None
    assert result.error


def test_parse_decision_returns_failure_without_silently_creating_a_command() -> None:
    result = parse_decision('{"operation": "invent"}', MaintenanceDecision)

    assert isinstance(result, DecisionParseResult)
    assert result.decision is None
    assert result.raw_output == '{"operation": "invent"}'
    assert result.repaired_output is None
    assert result.error


def test_parse_decision_repairs_once_and_returns_the_repaired_output() -> None:
    calls: list[str] = []

    def repair(raw_output: str, error: str) -> str:
        calls.append(f"{raw_output}:{error}")
        return '{"action": "answer"}'

    result = parse_decision("not json", ActionDecision, repair=repair)

    assert result.decision == ActionDecision(action="answer")
    assert result.raw_output == "not json"
    assert result.repaired_output == '{"action": "answer"}'
    assert result.error is None
    assert len(calls) == 1


def test_parse_decision_stops_after_one_failed_repair() -> None:
    calls = 0

    def repair(_raw_output: str, _error: str) -> str:
        nonlocal calls
        calls += 1
        return '{"action": " "}'

    result = parse_decision("not json", ActionDecision, repair=repair)

    assert result.decision is None
    assert result.repaired_output == '{"action": " "}'
    assert result.error
    assert calls == 1
