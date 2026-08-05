from __future__ import annotations

from tau3_evolver.agent.tool_schemas import build_baseline_prompt


class FakeTool:
    def __init__(self) -> None:
        self.openai_schema = {
            "type": "function",
            "function": {"name": "lookup_order", "parameters": {"type": "object"}},
        }


def test_builds_a_prompt_from_the_official_tau2_reset_policy_and_tools() -> None:
    prompt = build_baseline_prompt(
        observation="How can I return this item?",
        reset_info={
            "policy": "Follow the retail policy.",
            "tools": [{"type": "function", "function": {"name": "lookup_order"}}],
        },
    )

    assert prompt.messages == (
        {"role": "system", "content": "Follow the retail policy."},
        {"role": "user", "content": "How can I return this item?"},
    )
    assert prompt.tools == ({"type": "function", "function": {"name": "lookup_order"}},)


def test_includes_public_prior_messages_before_the_current_observation() -> None:
    prompt = build_baseline_prompt(
        observation="Can you check the status now?",
        reset_info={
            "policy": "Follow the retail policy.",
            "tools": [{"type": "function", "function": {"name": "lookup_order"}}],
        },
        history=(
            {"role": "user", "content": "My order is 123."},
            {"role": "assistant", "content": "I will check it."},
        ),
    )

    assert prompt.messages == (
        {"role": "system", "content": "Follow the retail policy."},
        {"role": "user", "content": "My order is 123."},
        {"role": "assistant", "content": "I will check it."},
        {"role": "user", "content": "Can you check the status now?"},
    )


def test_normalizes_and_copies_a_tool_object_openai_schema_without_tau2_imports() -> None:
    tool = FakeTool()

    prompt = build_baseline_prompt(
        observation="Where is my order?",
        reset_info={"policy": "Follow policy.", "tools": [tool]},
    )
    tool.openai_schema["function"]["name"] = "mutated"

    assert prompt.tools == (
        {
            "type": "function",
            "function": {"name": "lookup_order", "parameters": {"type": "object"}},
        },
    )
