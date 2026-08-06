from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from tau3_evolver.credential_policy import is_credential_key


_REDACTED = "[REDACTED]"


def redact_public_data(value: Any) -> Any:
    """Recursively preserve public metadata while removing credential values."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                _REDACTED if is_credential_key(key) else redact_public_data(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_public_data(item) for item in value]
    if isinstance(value, str) and _is_credential_bearing_url(value):
        return _REDACTED
    return value


def _is_credential_bearing_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(
        parsed.username or parsed.password or parsed.query or parsed.fragment
    )


__all__ = ["redact_public_data"]
