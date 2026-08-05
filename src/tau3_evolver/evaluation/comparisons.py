from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compare_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    baseline_index: int = 0,
) -> list[dict[str, Any]]:
    if not reports:
        return []
    baseline = reports[baseline_index]
    baseline_metrics = baseline["summary"]["metrics"]
    rows: list[dict[str, Any]] = []
    for report in reports:
        execution = report["execution"]
        memory = report["memory"]
        metrics = report["summary"]["metrics"]
        rows.append(
            {
                "execution_benchmark": execution["benchmark"],
                "memory_source_namespace": memory["source_namespace"],
                "memory_snapshot_id": memory["input_snapshot_id"],
                "cross_domain_memory": memory["cross_domain"],
                "mean_reward": metrics["mean_reward"],
                "pass_rate": metrics["pass_rate"],
                "mean_reward_delta": round(
                    metrics["mean_reward"] - baseline_metrics["mean_reward"], 12
                ),
                "pass_rate_delta": round(
                    metrics["pass_rate"] - baseline_metrics["pass_rate"], 12
                ),
            }
        )
    return rows
