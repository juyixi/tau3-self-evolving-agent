from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import math
from pathlib import Path
from typing import Any

from tau3_retail_evolver.eval.experiment import (
    BASE_NO_MEMORY,
    BASE_WITH_MEMORY,
    EXPERIMENT_ORDER,
    OPD_NO_MEMORY,
    OPD_WITH_MEMORY,
)
from tau3_retail_evolver.memory.json_store import write_bytes_atomic


_CELL_LABELS = {
    BASE_NO_MEMORY: "A Base / No Memory",
    BASE_WITH_MEMORY: "B Base / Frozen Memory",
    OPD_WITH_MEMORY: "C OPD / Frozen Memory",
    OPD_NO_MEMORY: "D OPD / No Memory",
}
_COLORS = ("#157f5b", "#2f6feb", "#b7791f", "#c2413b")
_OPD_KINDS = ("sel", "act", "write", "maint")
_OPD_KIND_COLORS = ("#2f6feb", "#157f5b", "#b7791f", "#c2413b")


def write_stage8_dashboard(
    path: Path,
    report: Mapping[str, Any],
) -> None:
    html = render_stage8_dashboard(report)
    write_bytes_atomic(path, html.encode("utf-8"))


def render_stage8_dashboard(report: Mapping[str, Any]) -> str:
    domain = report.get("design", {}).get("domain", "retail")
    if report.get("report_type") != f"tau3-{domain}-stage8-experiment":
        raise ValueError("dashboard requires a Stage 8 experiment report")
    cells = report["evaluation"]["cells"]
    pass_values = [_number(cells[label]["pass_at_1"]) for label in EXPERIMENT_ORDER]
    token_values = [
        _optional_number(cells[label]["mean_agent_tokens"])
        for label in EXPERIMENT_ORDER
    ]
    memory_labels = (BASE_WITH_MEMORY, OPD_WITH_MEMORY)
    reuse_values = [
        _optional_number(cells[label]["memory_reuse_coverage"])
        for label in memory_labels
    ]
    train_passes = report["fast_loop"]["passes"]
    growth_values = [
        sum(row["output_memory_counts"].values())
        for row in train_passes
    ]
    kind_counts = report["opd_dataset"]["kind_counts"]
    kind_order = [
        kind for kind in _OPD_KINDS if kind in kind_counts
    ] + sorted(set(kind_counts) - set(_OPD_KINDS))
    kl_curve = report["opd_training"]["forward_kl_curve"]

    charts = [
        _bar_chart(
            "Test pass@1",
            [_CELL_LABELS[label] for label in EXPERIMENT_ORDER],
            pass_values,
            colors=_COLORS,
            percent=True,
        ),
        _bar_chart(
            "Average Agent tokens per task",
            [_CELL_LABELS[label] for label in EXPERIMENT_ORDER],
            token_values,
            colors=_COLORS,
        ),
        _bar_chart(
            "Frozen Memory reuse coverage",
            [_CELL_LABELS[label] for label in memory_labels],
            reuse_values,
            colors=(_COLORS[1], _COLORS[2]),
            percent=True,
        ),
        _line_chart(
            "Memory growth across train passes",
            [f"Pass {row['pass_index']}" for row in train_passes],
            growth_values,
            color="#157f5b",
        ),
        _bar_chart(
            "OPD examples by kind",
            kind_order,
            [_number(kind_counts[kind]) for kind in kind_order],
            colors=_OPD_KIND_COLORS,
        ),
        _line_chart(
            "Forward KL during OPD training",
            [str(index + 1) for index in range(len(kl_curve))],
            [_number(row["forward_kl"]) for row in kl_curve],
            color="#c2413b",
            sparse_labels=True,
        ),
    ]
    return _document(
        experiment_id=str(report["experiment_id"]),
        domain=str(domain),
        charts=charts,
        contrast_table=_contrast_table(report["evaluation"]["contrasts"]),
        provenance_table=_provenance_table(report),
    )


def _document(
    *,
    experiment_id: str,
    domain: str,
    charts: Sequence[str],
    contrast_table: str,
    provenance_table: str,
) -> str:
    chart_markup = "\n".join(
        f'<section class="chart">{chart}</section>' for chart in charts
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 8 {escape(domain.title())} experiment report</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  color: #17212b;
  background: #f4f6f8;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
header {{
  background: #ffffff;
  border-bottom: 1px solid #d8dee4;
  padding: 28px max(24px, calc((100vw - 1180px) / 2));
}}
h1 {{ margin: 0 0 7px; font-size: 28px; letter-spacing: 0; }}
header p {{ margin: 0; color: #57606a; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}}
.chart {{
  min-width: 0;
  background: #ffffff;
  border: 1px solid #d8dee4;
  border-radius: 6px;
  padding: 16px;
  overflow-x: auto;
}}
.chart h2, .band h2 {{
  margin: 0 0 12px;
  font-size: 17px;
  letter-spacing: 0;
}}
svg {{ display: block; width: 100%; height: auto; }}
.band {{
  margin-top: 16px;
  background: #ffffff;
  border-top: 1px solid #d8dee4;
  border-bottom: 1px solid #d8dee4;
  padding: 18px;
  overflow-x: auto;
}}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{
  padding: 9px 10px;
  border-bottom: 1px solid #eaeef2;
  text-align: right;
  white-space: nowrap;
}}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #57606a; font-weight: 600; }}
.muted {{ color: #6e7781; }}
@media (max-width: 760px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .chart svg {{ min-width: 620px; }}
  main {{ padding: 14px; }}
  header {{ padding: 22px 18px; }}
}}
</style>
</head>
<body>
<header>
  <h1>Stage 8 {escape(domain.title())} experiment</h1>
  <p>{escape(experiment_id)} | Base/OPD x No Memory/Frozen Memory</p>
</header>
<main>
  <div class="grid">{chart_markup}</div>
  <section class="band"><h2>Paired contrasts</h2>{contrast_table}</section>
  <section class="band"><h2>Experiment provenance</h2>{provenance_table}</section>
</main>
</body>
</html>
"""


def _bar_chart(
    title: str,
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    colors: Sequence[str],
    percent: bool = False,
) -> str:
    width, height = 760, 330
    left, right, top, bottom = 62, 18, 28, 88
    plot_width = width - left - right
    plot_height = height - top - bottom
    numeric = [value for value in values if value is not None]
    maximum = max(numeric, default=1.0)
    if percent:
        maximum = max(1.0, maximum)
    else:
        maximum = _nice_max(maximum)
    slot = plot_width / max(1, len(values))
    bar_width = min(105.0, slot * 0.58)
    parts = [_axes(width, height, left, top, plot_width, plot_height, maximum, percent)]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        center = left + slot * (index + 0.5)
        color = colors[index % len(colors)]
        if value is not None:
            bar_height = plot_height * value / maximum if maximum else 0
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" '
                f'width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{color}" rx="2"/>'
            )
            parts.append(
                _svg_text(
                    center,
                    max(15.0, y - 7),
                    _format_value(value, percent=percent),
                    anchor="middle",
                    weight="600",
                )
            )
        else:
            parts.append(
                _svg_text(
                    center,
                    top + plot_height - 8,
                    "n/a",
                    anchor="middle",
                    fill="#6e7781",
                )
            )
        parts.extend(_wrapped_label(center, height - 55, label))
    return _chart_svg(title, width, height, parts)


def _line_chart(
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    color: str,
    sparse_labels: bool = False,
) -> str:
    width, height = 760, 330
    left, right, top, bottom = 62, 18, 28, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = _nice_max(max(values, default=1.0))
    count = len(values)
    points = []
    for index, value in enumerate(values):
        x = left + (plot_width * index / max(1, count - 1))
        y = top + plot_height - plot_height * value / maximum
        points.append((x, y, value))
    parts = [_axes(width, height, left, top, plot_width, plot_height, maximum, False)]
    if points:
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        marker_stride = max(1, len(points) // 24)
        for index, (x, y, value) in enumerate(points):
            if index % marker_stride == 0 or index == len(points) - 1:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>'
                )
    if labels and not sparse_labels:
        for index, label in enumerate(labels):
            x = left + plot_width * index / max(1, len(labels) - 1)
            parts.append(_svg_text(x, height - 32, label, anchor="middle"))
    elif labels:
        parts.append(_svg_text(left, height - 32, "start", anchor="start"))
        parts.append(
            _svg_text(left + plot_width, height - 32, "end", anchor="end")
        )
    return _chart_svg(title, width, height, parts)


def _axes(
    width: int,
    height: int,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    maximum: float,
    percent: bool,
) -> str:
    parts = []
    for index in range(5):
        ratio = index / 4
        y = top + plot_height * (1 - ratio)
        value = maximum * ratio
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
            f'y2="{y:.1f}" stroke="#eaeef2" stroke-width="1"/>'
        )
        parts.append(
            _svg_text(
                left - 9,
                y + 4,
                _format_value(value, percent=percent),
                anchor="end",
                fill="#6e7781",
            )
        )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}" stroke="#8c959f"/>'
    )
    return "".join(parts)


def _chart_svg(
    title: str,
    width: int,
    height: int,
    parts: Sequence[str],
) -> str:
    return (
        f"<h2>{escape(title)}</h2>"
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}">'
        + "".join(parts)
        + "</svg>"
    )


def _wrapped_label(x: float, y: float, label: str) -> list[str]:
    words = label.split()
    midpoint = max(1, len(words) // 2)
    if len(label) <= 18:
        lines = (label,)
    elif " / " in label:
        lines = tuple(label.split(" / ", maxsplit=1))
    else:
        lines = (" ".join(words[:midpoint]), " ".join(words[midpoint:]))
    return [
        _svg_text(x, y + index * 16, line, anchor="middle")
        for index, line in enumerate(lines)
    ]


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    anchor: str,
    fill: str = "#24292f",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'fill="{fill}" font-size="12" font-weight="{weight}">'
        f"{escape(value)}</text>"
    )


def _contrast_table(contrasts: Mapping[str, Mapping[str, Any]]) -> str:
    rows = []
    for name, value in contrasts.items():
        interval = value["pass_at_1_ci95"]
        rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{_format_value(value['pass_at_1_delta'], percent=True)}</td>"
            f"<td>{_format_value(interval[0], percent=True)} to "
            f"{_format_value(interval[1], percent=True)}</td>"
            f"<td>{_format_optional(value['mean_agent_tokens_delta'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Contrast</th><th>pass@1 delta</th>"
        "<th>95% paired CI</th><th>Token delta</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _provenance_table(report: Mapping[str, Any]) -> str:
    design = report["design"]
    fast = report["fast_loop"]
    dataset = report["opd_dataset"]
    training = report["opd_training"]
    values = (
        ("Train passes", design["train_passes"]),
        ("Train episodes", fast["episode_count"]),
        ("Final Memory items", fast["final_memory_item_count"]),
        ("Train Memory reuse", fast["memory_reuse_coverage"]),
        ("OPD examples", dataset["example_count"]),
        ("Optimizer steps", training["optimizer_steps"]),
        ("Mean forward KL", training["forward_kl_mean"]),
        ("Checkpoint", training["latest_checkpoint"]),
    )
    rows = "".join(
        f"<tr><td>{escape(str(label))}</td><td>{escape(_display(value))}</td></tr>"
        for label, value in values
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{_number(value):,.1f}"


def _format_value(value: float, *, percent: bool) -> str:
    return f"{value * 100:.1f}%" if percent else f"{value:,.1f}"


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    step = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    return step * magnitude


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"chart value must be numeric: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("chart value must be finite")
    return numeric


def _optional_number(value: Any) -> float | None:
    return None if value is None else _number(value)
