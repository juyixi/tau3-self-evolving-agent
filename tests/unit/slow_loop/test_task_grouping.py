import pytest

from tau3_evolver.slow_loop.task_grouping import (
    canonicalize_task_group,
    is_supported_task_group,
)


@pytest.mark.parametrize("value", ("retail", "airline", "retail.refund-v1"))
def test_accepts_benchmark_owned_groups(value: str) -> None:
    assert canonicalize_task_group(value.upper()) == value
    assert is_supported_task_group(value)


@pytest.mark.parametrize("value", ("", "../retail", "retail group", None))
def test_rejects_unsafe_groups(value: object) -> None:
    with pytest.raises(ValueError):
        canonicalize_task_group(value)
    assert not is_supported_task_group(value)
