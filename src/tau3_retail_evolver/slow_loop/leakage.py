from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any, Literal
import unicodedata
from urllib.parse import urlsplit

from tau3_retail_evolver.credential_policy import is_credential_key


ExampleKind = Literal["sel", "act", "write", "maint"]

_ARTIFACT_FORBIDDEN_KEYS = frozenset(
    {
        "apikeyenv",
        "baseurl",
        "evaluationcriteria",
        "evaluatormetadata",
        "evaluator",
        "goldenaction",
        "goldenactions",
        "goldenarguments",
        "nlassertion",
        "nlassertions",
        "privaterubric",
        "rewardbasis",
        "rubric",
        "testdatapath",
        "testtaskid",
        "testtaskids",
    }
)

_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "attribution",
        "attributionscore",
        "candidatevalues",
        "confidence",
        "gamma",
        "kappa",
        "lastused",
        "lastusedat",
        "memoryscore",
        "memoryvalue",
        "privilegedhindsight",
        "redundancy",
        "score",
        "successcount",
        "terminalevaluation",
        "usage",
        "usagecount",
        "value",
    }
)

_PRIVILEGED_KEY_MARKERS = (
    "attribution",
    "lastused",
    "redundancy",
)

_ACTION_MEMORY_VALUE = re.compile(
    r"(?:^|[^a-z0-9])(?:memory|memories|mem[-_][a-z0-9]+)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


def normalized_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]", "", normalized)


def audit_artifact_payload(payload: Any) -> None:
    """Reject credentials, evaluator data, and quarantined test references."""
    _walk_artifact(payload, path="artifact")


def audit_public_input(kind: ExampleKind, payload: Mapping[str, Any]) -> None:
    if kind not in {"sel", "act", "write", "maint"}:
        raise ValueError(f"unknown OPD example kind: {kind!r}")
    if not isinstance(payload, Mapping):
        raise TypeError("public input must be a mapping")
    _walk_public(kind, payload, path="public_input")


def _walk_artifact(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = normalized_key(key)
            if is_credential_key(key) or normalized in _ARTIFACT_FORBIDDEN_KEYS:
                raise ValueError(f"artifact contains credential or evaluator field at {path}.{key}")
            _walk_artifact(nested, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _walk_artifact(nested, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _credential_bearing_url(value):
            raise ValueError(f"artifact contains credential-bearing URL at {path}")
        if _is_test_or_evaluation_path(value):
            raise ValueError(f"artifact contains test or evaluation path at {path}")


def _walk_public(kind: ExampleKind, value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = normalized_key(key)
            if is_credential_key(key) or normalized in _ARTIFACT_FORBIDDEN_KEYS:
                raise ValueError(
                    f"public input contains credential or evaluator field at {path}.{key}"
                )
            if _is_privileged_public_key(normalized):
                raise ValueError(f"public input contains privileged field at {path}.{key}")
            if kind == "act" and (
                "memory" in normalized or "memories" in normalized
            ):
                raise ValueError(f"action public input contains memory at {path}.{key}")
            _walk_public(kind, nested, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _walk_public(kind, nested, path=f"{path}[{index}]")
        return
    _walk_artifact(value, path=path)
    if kind == "act" and isinstance(value, str) and _ACTION_MEMORY_VALUE.search(value):
        raise ValueError(f"action public input contains memory at {path}")


def _is_privileged_public_key(normalized: str) -> bool:
    if normalized in _PUBLIC_FORBIDDEN_KEYS:
        return True
    return any(marker in normalized for marker in _PRIVILEGED_KEY_MARKERS)


def _credential_bearing_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(
        parsed.username or parsed.password or parsed.query or parsed.fragment
    )


def _is_test_or_evaluation_path(value: str) -> bool:
    normalized = value.replace("\\", "/").casefold()
    return any(
        marker in normalized
        for marker in (
            "history/evaluations/",
            "/test/",
            "split=test",
            "test_tasks",
            "tasks_test",
        )
    )
