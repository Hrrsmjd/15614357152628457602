"""Build an ECMWF-style inferred AIFS v1.1 50r1-versus-49r1 scorecard.

The two public ECMWF scorecards each expose AIFS v1.1 as their control:

* AIFS v1.1 initialised from 49r1 in the first scorecard.
* AIFS v1.1 initialised from 50r1 in the second scorecard.

This module joins those aggregate control means and renders a descriptive
comparison. It must not be interpreted as a direct paired ECMWF verification:
the pages expose no date-level scores, their populations differ, and their
analysis sections use different analysis cycles as truth.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from public_score_analysis import (
    FIGURES,
    OUTPUTS,
    PROCESSED,
    SCORECARDS,
    derive_v11_cycle_comparison,
    extract_subpages_data,
    scorecard_to_frame,
)


LEADS = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]
COLOR_LIMIT_PERCENT = 25.0
METRICS = {
    "rmsef": {
        "label": "RMSE",
        "formula": "100 × (RMSE50 / RMSE49 − 1)",
        "unit": "%",
        "limit": 25.0,
        "cmap": "RdBu_r",
        "reverse_legend": True,
        "negative": "50r1 lower RMSE (better)",
        "positive": "50r1 higher RMSE (worse)",
    },
    "ccaf": {
        "label": "Anomaly correlation (ACC)",
        "formula": "ACC50 − ACC49",
        "unit": "percentage points",
        "limit": 5.0,
        "cmap": "RdBu",
        "reverse_legend": False,
        "negative": "50r1 lower ACC (worse)",
        "positive": "50r1 higher ACC (better)",
    },
    "sdaf": {
        "label": "Standard deviation of forecast anomaly (activity)",
        "formula": "100 × (SDAF50 / SDAF49 − 1)",
        "unit": "%",
        "limit": 5.0,
        "cmap": "RdBu",
        "reverse_legend": False,
        "negative": "50r1 less active (not a skill judgement)",
        "positive": "50r1 more active (not a skill judgement)",
    },
    "seeps": {
        "label": "SEEPS precipitation score",
        "formula": "100 × (SEEPS50 / SEEPS49 − 1)",
        "unit": "%",
        "limit": 10.0,
        "cmap": "RdBu_r",
        "reverse_legend": True,
        "negative": "50r1 lower SEEPS (better)",
        "positive": "50r1 higher SEEPS (worse)",
    },
}

REGIONS = [
    ("n.hem", "N. Hem", "Northern extratropics: 20°N–90°N"),
    ("s.hem", "S. Hem", "Southern extratropics: 90°S–20°S"),
    ("tropics", "Tropics", "Tropics: 20°S–20°N"),
    ("europe", "Europe", "Europe: 35°N–75°N, 12.5°W–42.5°E"),
    ("n.amer", "N. America", "North America: 25°N–60°N, 120°W–75°W"),
    ("e.asia", "E. Asia", "East Asia: 25°N–60°N, 102.5°E–150°E"),
    ("austnz", "Aust/NZ", "Australia and New Zealand"),
    ("n.atl", "N. Atlantic", "North Atlantic"),
    ("n.pac", "N. Pacific", "North Pacific"),
    ("arctic", "Arctic", "Arctic: 60°N–90°N"),
    ("antarctic", "Antarctic", "Antarctic: 90°S–60°S"),
]

OFFICIAL_FIELD_ORDER = [
    ("z", "50", "Geopotential", "50"),
    ("z", "100", "Geopotential", "100"),
    ("z", "250", "Geopotential", "250"),
    ("z", "500", "Geopotential", "500"),
    ("z", "850", "Geopotential", "850"),
    ("msl", "0", "Mean sea-level pressure", ""),
    ("t", "50", "Temperature", "50"),
    ("t", "100", "Temperature", "100"),
    ("t", "250", "Temperature", "250"),
    ("t", "500", "Temperature", "500"),
    ("t", "850", "Temperature", "850"),
    ("2t", "0", "2 metre temperature", ""),
    ("ff", "50", "Wind speed", "50"),
    ("ff", "100", "Wind speed", "100"),
    ("ff", "250", "Wind speed", "250"),
    ("ff", "500", "Wind speed", "500"),
    ("ff", "850", "Wind speed", "850"),
    ("10ff", "0", "10 metre wind speed", ""),
    ("2d", "0", "2 metre dew point", ""),
    ("tp", "0", "Total precipitation", ""),
]

ANALYSIS_FIELDS = [
    field
    for field in OFFICIAL_FIELD_ORDER
    if field[0] not in {"2d", "tp"}
]

OBSERVATION_FIELDS = [
    field
    for field in OFFICIAL_FIELD_ORDER
    if field[0] != "msl"
]


def load_derived_scores() -> pd.DataFrame:
    """Parse both source pages and derive the joined v1.1 aggregates."""
    frames = []
    for source_name, metadata in SCORECARDS.items():
        frames.append(
            scorecard_to_frame(
                source_name,
                metadata,
                extract_subpages_data(metadata["path"]),
            )
        )
    return derive_v11_cycle_comparison(pd.concat(frames, ignore_index=True))


def inferred_scorecard_frame(derived: pd.DataFrame) -> pd.DataFrame:
    """Return every published cell with transparent descriptive differences."""
    out = derived[derived["metric"].isin(METRICS)].copy()
    denominator = out["score_49r1"]
    ratio_change = np.where(
        denominator.abs() > 1e-12,
        100.0 * (out["score_50r1"] / denominator - 1.0),
        np.nan,
    )
    out["descriptive_difference"] = np.where(
        out["metric"].eq("ccaf"),
        out["score_50r1"] - out["score_49r1"],
        ratio_change,
    )
    out["difference_formula"] = out["metric"].map(
        {metric: metadata["formula"] for metric, metadata in METRICS.items()}
    )
    out["difference_units"] = out["metric"].map(
        {metric: metadata["unit"] for metric, metadata in METRICS.items()}
    )
    out["sample_count_gap_50r1_minus_49r1"] = (
        out["sample_count_50r1_scorecard"]
        - out["sample_count_49r1_scorecard"]
    )
    out["severe_sample_mismatch"] = (
        out["sample_count_gap_50r1_minus_49r1"].abs() > 1
    )
    out["comparison_quality"] = np.select(
        [
            out["severe_sample_mismatch"],
            out["reference_type"].eq("ob"),
        ],
        [
            "severe_sample_mismatch_descriptive_only",
            "common_observation_type_one_case_population_gap",
        ],
        default="different_analysis_truth_one_case_population_gap",
    )
    out["negative_interpretation"] = out["metric"].map(
        {metric: metadata["negative"] for metric, metadata in METRICS.items()}
    )
    out["positive_interpretation"] = out["metric"].map(
        {metric: metadata["positive"] for metric, metadata in METRICS.items()}
    )
    order = {
        (reference, parameter, level): rank
        for reference, fields in (
            ("ob", OBSERVATION_FIELDS),
            ("an", ANALYSIS_FIELDS),
        )
        for rank, (parameter, level, _, _) in enumerate(fields)
    }
    reference_order = {"ob": 0, "an": 1}
    region_order = {region: rank for rank, (region, _, _) in enumerate(REGIONS)}
    out["_reference_order"] = (
        out["reference_type"].map(reference_order).fillna(10_000)
    )
    out["_field_order"] = [
        order.get((reference, parameter, str(level)), 10_000)
        for reference, parameter, level in zip(
            out["reference_type"],
            out["parameter"],
            out["level_hpa"],
            strict=True,
        )
    ]
    out["_region_order"] = out["region"].map(region_order).fillna(10_000)
    return (
        out.sort_values(
            [
                "metric",
                "_reference_order",
                "_field_order",
                "_region_order",
                "lead_hours",
            ]
        )
        .drop(columns=["_reference_order", "_field_order", "_region_order"])
        .reset_index(drop=True)
    )


def inferred_rmse_frame(derived: pd.DataFrame) -> pd.DataFrame:
    """Return the RMSE-only view requested as the primary scorecard."""
    out = inferred_scorecard_frame(derived)
    out = out[out["metric"].eq("rmsef")].copy()
    out["rmse_relative_change_percent"] = out["descriptive_difference"]
    out["colour_interpretation"] = (
        "positive/red: 50r1 aggregate RMSE is larger (worse); "
        "negative/blue: 50r1 aggregate RMSE is smaller (better)"
    )
    return out.reset_index(drop=True)


def _colour(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "#e8e8e8"
    metadata = METRICS[metric]
    limit = float(metadata["limit"])
    norm = Normalize(-limit, limit, clip=True)
    return to_hex(
        colormaps[str(metadata["cmap"])](norm(value)),
        keep_alpha=False,
    )


def _field_label(name: str, level: str) -> str:
    return f"{name} {level} hPa" if level else name


def _lookup(
    frame: pd.DataFrame,
) -> dict[tuple[str, str, str, str, str, int], pd.Series]:
    return {
        (
            str(row["metric"]),
            str(row["reference_type"]),
            str(row["parameter"]),
            str(row["level_hpa"]),
            str(row["region"]),
            int(row["lead_hours"]),
        ): row
        for _, row in frame.iterrows()
    }


def _score_cell(
    lookup: dict[tuple[str, str, str, str, str, int], pd.Series],
    metric: str,
    reference: str,
    parameter: str,
    level: str,
    region: str,
) -> str:
    boxes = []
    for lead in LEADS:
        row = lookup.get((metric, reference, parameter, level, region, lead))
        if row is None:
            boxes.append(
                '<span class="lead-box missing" '
                f'title="T+{lead}: not published">&nbsp;</span>'
            )
            continue
        value = float(row["descriptive_difference"])
        score49 = float(row["score_49r1"])
        score50 = float(row["score_50r1"])
        n49 = int(row["sample_count_49r1_scorecard"])
        n50 = int(row["sample_count_50r1_scorecard"])
        units = html.escape(str(row["units"]))
        truth49 = html.escape(str(row["truth_49r1_scorecard"]))
        truth50 = html.escape(str(row["truth_50r1_scorecard"]))
        severe = bool(row["severe_sample_mismatch"])
        css_class = "lead-box severe" if severe else "lead-box"
        warning = " WARNING: severe sample-count mismatch." if severe else ""
        difference_units = str(row["difference_units"])
        tooltip = html.escape(
            (
                f"T+{lead}: {value:+.2f} {difference_units} | "
                f"49r1={score49:.6g} {units} (n={n49}); "
                f"50r1={score50:.6g} {units} (n={n50}). "
                f"Truth: 49 page={truth49}; 50 page={truth50}.{warning}"
            ),
            quote=True,
        )
        boxes.append(
            f'<span class="{css_class}" '
            f'style="--cell-colour:{_colour(metric, value)}" '
            f'title="{tooltip}">&nbsp;</span>'
        )
    return '<div class="lead-strip">' + "".join(boxes) + "</div>"


def _section_rows(
    lookup: dict[tuple[str, str, str, str, str, int], pd.Series],
    metric: str,
    reference: str,
    fields: list[tuple[str, str, str, str]],
) -> list[str]:
    rows = []
    previous_parameter: str | None = None
    for parameter, level, name, level_label in fields:
        show_name = parameter != previous_parameter
        previous_parameter = parameter
        row = ["<tr>"]
        row.append(
            f'<td class="reference">{"an" if reference == "an" else "obs"}</td>'
        )
        row.append(
            f'<td class="parameter">{html.escape(name) if show_name else ""}</td>'
        )
        row.append(f'<td class="level">{html.escape(level_label)}</td>')
        for region, _, _ in REGIONS:
            row.append(
                '<td class="score-cell">'
                + _score_cell(
                    lookup,
                    metric,
                    reference,
                    parameter,
                    level,
                    region,
                )
                + "</td>"
            )
        row.append("</tr>")
        rows.append("".join(row))
    return rows


def _available_fields(
    frame: pd.DataFrame,
    metric: str,
    reference: str,
    candidates: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    available = {
        (str(parameter), str(level))
        for parameter, level in frame[
            frame["metric"].eq(metric) & frame["reference_type"].eq(reference)
        ][["parameter", "level_hpa"]].itertuples(index=False, name=None)
    }
    return [
        field
        for field in candidates
        if (field[0], field[1]) in available
    ]


def _metric_legend(metric: str) -> str:
    metadata = METRICS[metric]
    limit = float(metadata["limit"])
    reverse = bool(metadata["reverse_legend"])
    left_value = limit if reverse else -limit
    right_value = -left_value
    left = _colour(metric, left_value)
    middle = _colour(metric, 0.0)
    right = _colour(metric, right_value)
    left_interpretation = (
        metadata["positive"] if left_value > 0 else metadata["negative"]
    )
    right_interpretation = (
        metadata["positive"] if right_value > 0 else metadata["negative"]
    )
    return f"""
<div class="legend" aria-label="{html.escape(str(metadata["label"]), quote=True)} colour scale">
  <span class="legend-label legend-label-left">
    {html.escape(str(left_interpretation))}
    <strong>{left_value:+g} {html.escape(str(metadata["unit"]))}</strong>
  </span>
  <div class="gradient" style="background:linear-gradient(
    90deg, {left}, {middle}, {right});"></div>
  <span class="legend-label">
    {html.escape(str(right_interpretation))}
    <strong>{right_value:+g} {html.escape(str(metadata["unit"]))}</strong>
  </span>
</div>
<div class="subtitle">
  Formula: <code>{html.escape(str(metadata["formula"]))}</code>.
  Colours saturate at ±{limit:g} {html.escape(str(metadata["unit"]))}.
  Red is on the left and blue is on the right. Diagonal hatching marks a
  sample-count difference greater than one case.
</div>
"""


def _metric_table(
    frame: pd.DataFrame,
    lookup: dict[tuple[str, str, str, str, str, int], pd.Series],
    metric: str,
    region_headers: str,
) -> str:
    metadata = METRICS[metric]
    analysis_fields = _available_fields(
        frame, metric, "an", ANALYSIS_FIELDS
    )
    observation_fields = _available_fields(
        frame, metric, "ob", OBSERVATION_FIELDS
    )
    rows: list[str] = []
    if observation_fields:
        rows.append(
            '<tr class="section-divider observation-divider"><td colspan="14">'
            "<strong>Against observations</strong>"
            "<span>Same observation type on both pages; populations generally "
            "differ by one case.</span></td></tr>"
        )
        rows.extend(
            _section_rows(
                lookup,
                metric,
                "ob",
                observation_fields,
            )
        )
    if analysis_fields:
        rows.append(
            '<tr class="section-divider analysis-divider"><td colspan="14">'
            "<strong>Against analysis</strong>"
            "<span>Different analysis-cycle truth on the two pages; "
            "descriptive only.</span></td></tr>"
        )
        rows.extend(
            _section_rows(
                lookup,
                metric,
                "an",
                analysis_fields,
            )
        )
    return f"""
<section id="{metric}">
<h2>{html.escape(str(metadata["label"]))}</h2>
{_metric_legend(metric)}
<div class="table-wrap">
<table>
<thead>
<tr><th>ref</th><th>field</th><th>hPa</th>{region_headers}</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</div>
</section>
"""


def render_html(frame: pd.DataFrame, destination: Path) -> None:
    """Write a self-contained, hoverable ECMWF-style scorecard."""
    lookup = _lookup(frame)
    region_headers = "".join(
        f'<th title="{html.escape(description, quote=True)}">{label}</th>'
        for _, label, description in REGIONS
    )
    lead_labels = " ".join(str(lead) for lead in LEADS)
    metric_tables = "".join(
        _metric_table(frame, lookup, metric, region_headers)
        for metric in ("rmsef", "ccaf", "sdaf", "seeps")
    )
    template = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inferred AIFS v1.1 50r1 versus 49r1 scorecard</title>
<style>
  :root {{
    --bg: #F8F2EA;
    --ink: #1C1C1C;
    --blue: #0000FF;
    --blue-hover: #0000CC;
    --red: #D94A3A;
    --muted: #1C1C1C;
    --line: #D8D0C6;
    --soft: #EEE7DE;
    font-size: 15px;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    color: var(--ink);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
      "Liberation Mono", "Courier New", monospace;
    line-height: 1.5;
    margin: 0;
    background: var(--bg);
  }}
  a {{ color: var(--ink); }}
  a:hover {{ color: var(--blue-hover); }}
  .site-header {{
    background: var(--ink);
    color: var(--bg);
    min-height: 52px;
    padding: 14px max(16px, calc((100% - 1120px) / 2));
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid var(--ink);
  }}
  .site-header a {{
    color: var(--bg);
    text-decoration: none;
    font-size: inherit;
  }}
  .site-header .path {{ color: var(--bg); font-size: inherit; }}
  main {{ width: 100%; padding: 28px 0 72px; }}
  .hero {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 0 auto;
    border-bottom: 1px solid var(--ink);
    padding-bottom: 18px;
  }}
  .eyebrow {{
    color: var(--ink);
    font-size: inherit;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  h1 {{
    font-size: inherit;
    font-weight: 400;
    letter-spacing: 0;
    line-height: 1.5;
    margin: 0 0 10px;
  }}
  h2 {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 34px auto 6px;
    font-size: inherit;
    font-weight: 400;
    padding-top: 20px;
    border-top: 1px solid var(--ink);
  }}
  .subtitle {{ font-size: inherit; line-height: 1.5; color: var(--ink); }}
  main > section > .subtitle {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin-left: auto;
    margin-right: auto;
  }}
  .evidence-order {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 1px;
    background: var(--ink);
    border: 1px solid var(--ink);
    margin: 20px 0 0;
  }}
  .evidence-order div {{ background: var(--bg); padding: 12px 14px; }}
  .evidence-order strong {{ display: block; font-weight: 400; margin-bottom: 3px; }}
  .evidence-order .primary strong,
  .evidence-order .secondary strong {{ color: var(--ink); }}
  .evidence-order span {{ color: var(--ink); font-size: inherit; }}
  .warning {{
    width: calc(100% - 32px);
    max-width: 1120px;
    border-left: 3px solid var(--red);
    padding: 6px 14px;
    margin: 20px auto;
    font-size: inherit;
  }}
  .warning strong {{ font-weight: 400; color: var(--ink); }}
  .legend {{
    display: grid;
    grid-template-columns: minmax(190px, 1fr) minmax(260px, 420px) minmax(190px, 1fr);
    align-items: center;
    gap: 12px;
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 14px auto 7px;
    font-size: inherit;
  }}
  .legend-label {{ color: var(--ink); }}
  .legend-label strong {{ color: var(--ink); font-weight: 400; display: block; }}
  .legend-label-left {{ text-align: right; }}
  .gradient {{
    width: 100%; height: 14px;
    border: 1px solid var(--ink);
  }}
  .table-wrap {{
    width: calc(100% - 32px);
    overflow-x: hidden;
    border: 1px solid var(--ink);
    max-width: none;
    margin: 14px auto 0;
    background: var(--bg);
  }}
  table {{
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 12px;
    white-space: nowrap;
    background: var(--bg);
  }}
  th, td {{ border: 1px solid var(--line); padding: 2px 3px; }}
  th {{
    background: var(--ink);
    color: var(--bg);
    font-weight: 400;
    text-align: center;
    position: sticky;
    top: 0;
    z-index: 3;
  }}
  th:nth-child(1) {{ width: 34px; }}
  th:nth-child(2) {{ width: 172px; }}
  th:nth-child(3) {{ width: 42px; }}
  td.reference {{ background: var(--soft); font-weight: 400; text-align: center; color: var(--muted); }}
  td.parameter {{ background: #F3ECDF; width: 172px; min-width: 0; overflow: hidden; }}
  td.level {{ background: var(--soft); width: 42px; min-width: 0; text-align: right; color: var(--muted); }}
  td.score-cell {{ width: auto; min-width: 0; padding: 1px; }}
  .lead-strip {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 1px; }}
  .lead-box {{
    display: block; height: 17px; min-width: 10px;
    background: var(--cell-colour); border: 1px solid var(--bg);
    box-sizing: border-box; cursor: help;
  }}
  .lead-box:hover {{ outline: 2px solid var(--ink); position: relative; z-index: 2; }}
  .lead-box.missing {{ background: #E6DFD6; }}
  .lead-box.severe {{
    background:
      repeating-linear-gradient(135deg, transparent 0 3px, rgba(30,30,30,.45) 3px 4px),
      var(--cell-colour);
  }}
  .section-divider td {{ padding: 8px 10px; text-align: left; border-top: 1px solid var(--ink); }}
  .section-divider strong {{ font-weight: 400; display: inline-block; min-width: 180px; }}
  .section-divider span {{ color: var(--ink); font-size: inherit; }}
  .observation-divider td {{ background: #ECEBFF; border-left: 3px solid var(--blue); }}
  .observation-divider strong {{ color: var(--ink); }}
  .analysis-divider td {{ background: #F5E5DF; border-left: 3px solid var(--red); }}
  .analysis-divider strong {{ color: var(--ink); }}
  .lead-key {{
    width: calc(100% - 32px);
    max-width: 1120px;
    color: var(--ink);
    font-size: inherit;
    margin: 8px auto 12px;
    letter-spacing: .02em;
  }}
  .notes {{
    width: calc(100% - 32px);
    max-width: 1120px;
    font-size: inherit;
    line-height: 1.5;
    margin: 34px auto 0;
    padding-top: 18px;
    border-top: 1px solid var(--ink);
    color: var(--muted);
  }}
  .notes strong {{ color: var(--ink); font-weight: 400; }}
  nav {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 20px auto 4px;
  }}
  nav a {{
    display: inline-block; padding: 5px 9px; margin: 0 4px 6px 0;
    border: 1px solid var(--ink); background: var(--bg); color: var(--ink);
    text-decoration: none;
  }}
  nav a:hover {{ background: var(--ink); color: var(--bg); }}
  code {{ background: #EFE8DF; color: var(--ink); padding: 1px 4px; }}
  @media (max-width: 760px) {{
    .evidence-order {{ grid-template-columns: 1fr; }}
    .legend {{ grid-template-columns: 1fr; }}
    .legend-label, .legend-label-left {{ text-align: left; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 1050px; }}
  }}
</style>
</head>
<body>
<header class="site-header">
  <a href="https://hrrs.ai/">hrrs.ai</a>
  <span class="path">/ atmospheric model diagnostics / scorecard</span>
</header>
<main>
<div class="hero">
  <div class="eyebrow">AIFS v1.1 / IFS cycle sensitivity</div>
  <h1>50r1 versus 49r1 initialisation</h1>
  <div class="subtitle">
    Inferred AIFS Single v1.1 comparison from public ECMWF aggregate control
    means. Each regional cell contains ten lead boxes. Hover a box for its
    lead, both means, units, verification truth and
    <em>n</em><sub>49</sub>/<em>n</em><sub>50</sub>.
  </div>
  <div class="evidence-order">
    <div class="primary">
      <strong>01 / against observations</strong>
      <span>Shown first. Same observation type; aggregate populations generally differ by one case.</span>
    </div>
    <div class="secondary">
      <strong>02 / against analysis</strong>
      <span>Shown second. The 49r1 and 50r1 pages use different analysis-cycle truth.</span>
    </div>
  </div>
</div>
<div class="warning">
  <strong>Inferred descriptive comparison—not a direct ECMWF scorecard.</strong>
  No significance frames are shown because date-level paired scores and their
  covariance are unavailable. Analysis rows also use 49r1 truth on the 49r1
  page and 50r1 truth on the 50r1 page.
</div>
<nav>
  <a href="#rmsef">RMSE</a>
  <a href="#ccaf">ACC</a>
  <a href="#sdaf">Forecast activity</a>
  <a href="#seeps">SEEPS precipitation</a>
</nav>
<div class="lead-key">lead hours within every regional cell: {lead_labels}</div>
{metric_tables}
<div class="notes">
  <strong>Provenance.</strong>
  The 49r1 values are the <code>control.mean</code> arrays from ECMWF’s
  “AIFS v2/50r1 vs AIFS v1.1/49r1” page. The 50r1 values are the
  <code>control.mean</code> arrays from “AIFS v2/50r1 vs AIFS v1.1/50r1”.
  Observation rows share the same observation type and are more defensible,
  but their populations still differ by one case. Exact date membership and
  paired-bootstrap significance cannot be reconstructed from the HTML.
  Forecast activity uses the same red-to-blue scale for visual consistency,
  but “more active” and “less active” are not themselves better/worse skill
  judgements.
</div>
</main>
</body>
</html>
"""
    destination.write_text(template, encoding="utf-8")


def _matrix_for(
    frame: pd.DataFrame,
    reference: str,
    region: str,
    fields: list[tuple[str, str, str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(fields), len(LEADS)), np.nan)
    severe = np.zeros(matrix.shape, dtype=bool)
    subset = frame[
        frame["reference_type"].eq(reference) & frame["region"].eq(region)
    ]
    field_index = {
        (parameter, level): index
        for index, (parameter, level, _, _) in enumerate(fields)
    }
    lead_index = {lead: index for index, lead in enumerate(LEADS)}
    for _, row in subset.iterrows():
        key = (str(row["parameter"]), str(row["level_hpa"]))
        lead = int(row["lead_hours"])
        if key not in field_index or lead not in lead_index:
            continue
        i, j = field_index[key], lead_index[lead]
        matrix[i, j] = row["rmse_relative_change_percent"]
        severe[i, j] = bool(row["severe_sample_mismatch"])
    return matrix, severe


def plot_static_scorecard(
    frame: pd.DataFrame,
    reference: str,
    fields: list[tuple[str, str, str, str]],
    destination: Path,
) -> None:
    """Render a readable all-region static companion to the HTML scorecard."""
    labels = [_field_label(name, level) for _, _, name, level in fields]
    fig, axes = plt.subplots(4, 3, figsize=(20, 21), constrained_layout=True)
    axes_flat = axes.ravel()
    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("#e8e8e8")
    image = None
    for ax, (region, region_label, _) in zip(axes_flat, REGIONS, strict=False):
        matrix, severe = _matrix_for(frame, reference, region, fields)
        image = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=-COLOR_LIMIT_PERCENT,
            vmax=COLOR_LIMIT_PERCENT,
            aspect="auto",
        )
        for i, j in zip(*np.where(severe), strict=True):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="none",
                    edgecolor=(0.1, 0.1, 0.1, 0.65),
                    hatch="///",
                    linewidth=0,
                )
            )
        ax.set_title(region_label, fontsize=11)
        ax.set_xticks(range(len(LEADS)), LEADS, rotation=45, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Lead (h)", fontsize=8)
        ax.tick_params(length=0)
    for ax in axes_flat[len(REGIONS) :]:
        ax.axis("off")
    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=axes_flat.tolist(),
        orientation="horizontal",
        shrink=0.72,
        pad=0.02,
        aspect=45,
    )
    colorbar.set_label(
        "100 × (aggregate RMSE50 / aggregate RMSE49 − 1) (%)  "
        "blue/negative: 50r1 better; red/positive: 50r1 worse; "
        "colour saturated beyond ±25%"
    )
    fig.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="0.25",
                hatch="///",
                label="sample-count difference > 1",
            )
        ],
        loc="lower right",
        bbox_to_anchor=(0.96, 0.015),
        frameon=False,
        fontsize=8,
    )
    truth_note = (
        "49r1 and 50r1 page values use different analysis-cycle truth"
        if reference == "an"
        else "common observation type; scorecard populations differ by one case"
    )
    fig.suptitle(
        "Inferred AIFS v1.1 50r1 versus 49r1 aggregate RMSE scorecard\n"
        f"{'Analysis' if reference == 'an' else 'Observation'} verification — "
        f"{truth_note}; no reconstructed significance",
        fontsize=15,
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    derived = load_derived_scores()
    inferred_all = inferred_scorecard_frame(derived)
    inferred = inferred_rmse_frame(derived)
    all_csv_path = PROCESSED / "inferred_v11_50r1_vs_49r1_all_metrics.csv"
    all_parquet_path = (
        PROCESSED / "inferred_v11_50r1_vs_49r1_all_metrics.parquet"
    )
    csv_path = PROCESSED / "inferred_v11_50r1_vs_49r1_rmse.csv"
    parquet_path = PROCESSED / "inferred_v11_50r1_vs_49r1_rmse.parquet"
    html_path = OUTPUTS / "inferred_v11_50r1_vs_49r1_scorecard.html"
    inferred_all.to_csv(all_csv_path, index=False)
    inferred_all.to_parquet(all_parquet_path, index=False)
    inferred.to_csv(csv_path, index=False)
    inferred.to_parquet(parquet_path, index=False)
    render_html(inferred_all, html_path)
    plot_static_scorecard(
        inferred,
        "an",
        ANALYSIS_FIELDS,
        FIGURES / "inferred_v11_50r1_vs_49r1_rmse_analysis.png",
    )
    plot_static_scorecard(
        inferred,
        "ob",
        OBSERVATION_FIELDS,
        FIGURES / "inferred_v11_50r1_vs_49r1_rmse_observations.png",
    )
    summary: dict[str, Any] = {
        "title": "Inferred AIFS v1.1 50r1 versus 49r1 scorecard",
        "formulas": {
            metric: metadata["formula"]
            for metric, metadata in METRICS.items()
        },
        "rows": int(len(inferred_all)),
        "metric_rows": {
            metric: int(inferred_all["metric"].eq(metric).sum())
            for metric in METRICS
        },
        "rmse_rows": int(len(inferred)),
        "analysis_rows": int(inferred["reference_type"].eq("an").sum()),
        "observation_rows": int(inferred["reference_type"].eq("ob").sum()),
        "severe_sample_mismatch_rows": int(
            inferred_all["severe_sample_mismatch"].sum()
        ),
        "significance_reconstructed": False,
        "outputs": {
            "html": str(html_path.relative_to(OUTPUTS.parent)),
            "all_metrics_csv": str(all_csv_path.relative_to(OUTPUTS.parent)),
            "all_metrics_parquet": str(
                all_parquet_path.relative_to(OUTPUTS.parent)
            ),
            "csv": str(csv_path.relative_to(OUTPUTS.parent)),
            "parquet": str(parquet_path.relative_to(OUTPUTS.parent)),
            "analysis_png": str(
                (
                    FIGURES
                    / "inferred_v11_50r1_vs_49r1_rmse_analysis.png"
                ).relative_to(OUTPUTS.parent)
            ),
            "observation_png": str(
                (
                    FIGURES
                    / "inferred_v11_50r1_vs_49r1_rmse_observations.png"
                ).relative_to(OUTPUTS.parent)
            ),
        },
        "limitations": [
            "aggregate ratio of means, not a mean of paired case differences",
            "date-level membership and covariance are unavailable",
            "analysis rows use different analysis-cycle truth",
            "scorecard populations differ",
        ],
    }
    summary_path = OUTPUTS / "inferred_v11_scorecard_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
