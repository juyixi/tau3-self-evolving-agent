from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from tau3_retail_evolver.slow_loop.audit import audit_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit a published Stage 5 OPD dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_dataset(args.dataset_dir)
    sys.stdout.write(
        json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
