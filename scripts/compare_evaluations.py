from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any

from tau3_retail_evolver.eval.metrics import (
    compare_evaluation_reports,
    read_evaluation_json,
    write_evaluation_json,
)


_LABEL = re.compile(r"^[a-z0-9_-]+$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare controlled Tau3 Retail evaluation reports."
    )
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reports = _load_labeled_reports(args.report)
    comparison = compare_evaluation_reports(
        reports,
        baseline_label=args.baseline_label,
    )
    write_evaluation_json(args.output, comparison)
    sys.stdout.write(
        json.dumps(
            {
                "baseline_label": args.baseline_label,
                "output": str(args.output),
                "report_count": len(reports),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


def _load_labeled_reports(
    specifications: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    reports: dict[str, Mapping[str, Any]] = {}
    for specification in specifications:
        label, separator, raw_path = specification.partition("=")
        if (
            not separator
            or not _LABEL.fullmatch(label)
            or not raw_path.strip()
        ):
            raise ValueError(
                "report must use a safe lowercase label and path: label=path"
            )
        if label in reports:
            raise ValueError(f"duplicate report label: {label}")
        reports[label] = read_evaluation_json(Path(raw_path))
    return reports


if __name__ == "__main__":
    raise SystemExit(main())
