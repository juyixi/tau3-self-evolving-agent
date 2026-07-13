from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tau3_retail_evolver.fast_loop.decisions import (
    ActionDecision,
    DecisionParseResult,
    MaintenanceDecision,
    MemoryWrite,
    SelectionDecision,
    WriteDecision,
    parse_decision,
)
from tau3_retail_evolver.memory.operations import DeleteCommand, LookupCommand, MergeCommand
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
        validator=lambda decision: decision.validate_candidates({"memory-1"}),
    )

    assert result.decision is None
    assert "not a candidate" in (result.error or "")


def test_action_and_write_decisions_validate_text_and_json_metadata() -> None:
    assert ActionDecision(action="  reply  ").action == "reply"
    with pytest.raises(ValidationError, match="blank"):
        ActionDecision(action=" ")

    write = MemoryWrite(
        tier=MemoryTier.TIP,
        content="  Confirm the order before changing it. ",
        retrieval_text=None,
        metadata={"source": ["public"]},
    )
    assert WriteDecision(memories=(write,)).memories == (write,)
    with pytest.raises(ValidationError, match="JSON"):
        MemoryWrite(
            tier=MemoryTier.TIP,
            content="Valid content",
            retrieval_text="Valid retrieval text",
            metadata={"score": math.nan},
        )


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
