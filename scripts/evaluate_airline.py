from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

from scripts.evaluate_retail import main as _evaluate


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(
        value == "--config" or value.startswith("--config=")
        for value in arguments
    ):
        arguments[:0] = ["--config", str(Path("configs") / "airline.yaml")]
    return _evaluate(arguments, expected_domain="airline")


if __name__ == "__main__":
    raise SystemExit(main())
