from __future__ import annotations

import re
from typing import Any


_CREDENTIAL_TERMINAL_WORDS = {
    "token",
    "authorization",
    "secret",
    "secrets",
    "password",
    "passwords",
    "credential",
    "credentials",
}
_CREDENTIAL_KEY_SUFFIXES = (("api", "key"), ("private", "key"), ("access", "key"))
_COMPACT_CREDENTIAL_KEYS = {
    "apikey",
    "apitoken",
    "accesstoken",
    "authtoken",
    "clientsecret",
    "privatekey",
    "accesskey",
}


def is_credential_key(key: Any) -> bool:
    """Return whether a field name conventionally carries a credential value."""
    key_text = str(key)
    compact = "".join(re.findall(r"[a-z0-9]+", key_text.casefold()))
    if compact in _COMPACT_CREDENTIAL_KEYS:
        return True
    separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key_text)
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", separated)
    words = tuple(re.findall(r"[a-z0-9]+", separated.casefold()))
    return bool(words) and (
        words[-1] in _CREDENTIAL_TERMINAL_WORDS
        or any(words[-len(suffix) :] == suffix for suffix in _CREDENTIAL_KEY_SUFFIXES)
    )
