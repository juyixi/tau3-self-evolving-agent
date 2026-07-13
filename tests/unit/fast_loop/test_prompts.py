from __future__ import annotations

import json
import math

import pytest

from tau3_retail_evolver.fast_loop.prompts import (
    LifecyclePrompt,
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
        }


def test_write_prompt_allows_terminal_public_evaluation_but_rejects_hidden_criteria() -> None:
    prompt = build_write_prompt(
        **_public_context(),
        trajectory=[{"role": "assistant", "content": "Address updated."}],
        terminal_evaluation={"reward": 1.0, "official_result": "success"},
    )

    assert prompt.payload["trajectory"] == [{"role": "assistant", "content": "Address updated."}]
    assert prompt.payload["terminal_evaluation"] == {"official_result": "success", "reward": 1.0}
    with pytest.raises(ValueError, match="hidden"):
        build_write_prompt(
            **_public_context(),
            trajectory=[],
            terminal_evaluation={"evaluation_criteria": "private rubric"},
        )


def test_maintenance_prompt_keeps_only_caller_diagnostics_and_command_schemas() -> None:
    prompt = build_maintenance_prompt(
        diagnostics={"stale_count": 2, "tier": "tip"},
    )

    assert prompt.payload == {"diagnostics": {"stale_count": 2, "tier": "tip"}}
    assert {schema["operation"] for schema in prompt.command_schemas} == {
        "lookup",
        "merge",
        "delete",
    }


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


def test_public_prompt_rejects_forbidden_evaluation_and_metadata_fields() -> None:
    context = _public_context()
    context["policy"] = {"official": "public", "evaluation_criteria": "private"}

    with pytest.raises(ValueError, match="forbidden"):
        build_selection_prompt(**context, candidates=())
