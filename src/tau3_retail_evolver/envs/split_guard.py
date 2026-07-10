from __future__ import annotations


def require_learning_split(split: str) -> None:
    """Reject splits that must never be used for learning."""
    if split != "train":
        raise ValueError(f"learning split must be 'train', received {split!r}")
