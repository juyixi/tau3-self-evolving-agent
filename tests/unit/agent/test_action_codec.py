from __future__ import annotations

import pytest

from tau3_evolver.agent.action_codec import Tau2ActionCodec


TOOLS = {"find_order", "get_order_details"}


def test_decodes_ordinary_user_text() -> None:
    assert Tau2ActionCodec.decode("I can help with that.", TOOLS) == "I can help with that."


def test_preserves_a_valid_json_tool_call() -> None:
    action = '{"name": "find_order", "arguments": {"order_id": "123"}}'

    assert Tau2ActionCodec.decode(action, TOOLS) == action


def test_preserves_a_valid_function_style_tool_call() -> None:
    action = 'find_order(order_id="123")'

    assert Tau2ActionCodec.decode(action, TOOLS) == action


def test_rejects_an_unknown_tool_call() -> None:
    with pytest.raises(ValueError, match="not available"):
        Tau2ActionCodec.decode('cancel_order(order_id="123")', TOOLS)


@pytest.mark.parametrize(
    "action",
    (
        '{"name": "find_order", "arguments": "order_id=123"}',
        "find_order(order_id=)",
    ),
)
def test_rejects_malformed_tool_arguments(action: str) -> None:
    with pytest.raises(ValueError, match="arguments"):
        Tau2ActionCodec.decode(action, TOOLS)


def test_preserves_the_stop_action() -> None:
    assert Tau2ActionCodec.decode("stop", TOOLS) == "###STOP###"


def test_preserves_the_official_tau2_stop_token() -> None:
    assert Tau2ActionCodec.decode("###STOP###", TOOLS) == "###STOP###"


def test_strips_thinking_blocks_without_losing_the_final_answer() -> None:
    model_output = "<think>Check the order state first.</think>I found your order."

    assert Tau2ActionCodec.decode(model_output, TOOLS) == "I found your order."


def test_rejects_an_unterminated_thinking_block() -> None:
    with pytest.raises(ValueError, match="unterminated thinking"):
        Tau2ActionCodec.decode("<think>I need to inspect the order", TOOLS)
