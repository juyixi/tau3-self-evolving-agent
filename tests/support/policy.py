from __future__ import annotations

from collections.abc import Sequence

from tau3_retail_evolver.models.policy import DecisionRequest, DecisionResponse, Policy


class ScriptedPolicy(Policy):
    """Deterministic policy used by rollout tests."""

    def __init__(self, responses: Sequence[DecisionResponse]) -> None:
        self._responses = iter(responses)
        self._requests: list[DecisionRequest] = []

    @property
    def requests(self) -> tuple[DecisionRequest, ...]:
        return tuple(self._requests)

    def generate(self, request: DecisionRequest) -> DecisionResponse:
        try:
            response = next(self._responses)
        except StopIteration as error:
            raise RuntimeError("scripted policy has no remaining responses") from error
        self._requests.append(request)
        return response
