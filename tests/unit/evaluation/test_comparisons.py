from tau3_evolver.evaluation.comparisons import compare_reports


def _run(
    *,
    source: str | None,
    snapshot: str | None,
    cross_domain: bool,
    mean_reward: float,
) -> dict:
    return {
        "execution": {"benchmark": "airline"},
        "memory": {
            "source_namespace": source,
            "input_snapshot_id": snapshot,
            "cross_domain": cross_domain,
        },
        "summary": {
            "metrics": {"mean_reward": mean_reward, "pass_rate": mean_reward}
        },
    }


def test_comparison_reads_dimensions_and_metrics_from_run_record() -> None:
    baseline = _run(
        source=None,
        snapshot=None,
        cross_domain=False,
        mean_reward=0.4,
    )
    transfer = _run(
        source="retail",
        snapshot="s1",
        cross_domain=True,
        mean_reward=0.6,
    )

    rows = compare_reports((baseline, transfer))

    assert len(rows) == 2
    assert rows[1]["memory_source_namespace"] == "retail"
    assert rows[1]["mean_reward_delta"] == 0.2
