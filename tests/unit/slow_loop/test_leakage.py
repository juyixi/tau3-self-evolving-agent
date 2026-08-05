from __future__ import annotations

import pytest

from tau3_evolver.slow_loop.leakage import (
    audit_artifact_payload,
    audit_public_input,
    normalized_key,
)


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": {"memoryValue": 0.4}},
        {"diagnostics": [{"last-used": "2026-01-01"}]},
        {"evaluation_criteria": {"actions": []}},
        {"rubric": {"nlAssertions": ["private"]}},
        {"url": "https://user:secret@example.com/path"},
        {"nested": {"apiToken": "secret"}},
        {"note": "Authorization: Bearer sk-secret-value"},
        {"note": "api_key=sk-secret-value"},
    ],
)
def test_public_input_rejects_privileged_or_secret_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        audit_public_input("sel", payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"context": {"selectedMemories": ["mem-tip-a"]}},
        {"context": {"notes": "Use mem_tip_abc123"}},
        {"memory_context": None},
    ],
)
def test_action_public_input_rejects_memory_even_under_alias(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="action public input contains memory"):
        audit_public_input("act", payload)


@pytest.mark.parametrize(
    "text",
    [
        "The pet bed uses memory foam.",
        "Based on the memories and policy guidance, confirm before writing.",
    ],
)
def test_action_public_input_allows_business_or_conversation_memory_words(
    text: str,
) -> None:
    audit_public_input("act", {"history": [{"next_observation": text}]})


@pytest.mark.parametrize(
    "value",
    [
        "history/evaluations/run-a",
        "data/retail/test/tasks.json",
        "split=test",
    ],
)
def test_all_artifact_payloads_reject_evaluation_or_test_paths(value: str) -> None:
    with pytest.raises(ValueError, match="test|evaluation"):
        audit_artifact_payload({"source": value})


def test_selection_public_input_allows_candidate_join_ids_and_content() -> None:
    audit_public_input(
        "sel",
        {
            "candidates": [
                {
                    "memory_id": "mem_tip_public",
                    "tier": "tip",
                    "content": "Public candidate content.",
                }
            ]
        },
    )


def test_normalized_key_handles_case_and_separators() -> None:
    assert normalized_key("Last.Used-At") == "lastusedat"
    assert normalized_key("memoryValue") == "memoryvalue"
