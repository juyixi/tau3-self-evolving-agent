from __future__ import annotations

from tau3_retail_evolver.fast_loop.baseline_prompt import build_baseline_prompt


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
