from enum import StrEnum


class ExecutionMode(StrEnum):
    TRAIN = "train"
    TEST = "test"


__all__ = ["ExecutionMode"]
