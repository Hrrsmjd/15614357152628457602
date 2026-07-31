"""Render separate self-contained forecast scorecards for the two experiments."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aifs_forecast_experiment import (
    EVALUATION_VARIABLES,
    HIGHLIGHT_VARIABLES,
    LEADS,
    PROCESSED,
    OUTPUTS,
    REGIONS,
    ROOT,
)
from extract_aifs_v11_metadata import NORMALIZATION_JSON


SCORES_PATH = PROCESSED / "aifs_v11_forecast_cohort_scores.parquet"
COMPARISONS_PATH = PROCESSED / "aifs_v11_forecast_cycle_comparisons.parquet"
METHODS_PATH = ROOT / "forecast_experiment_methods.md"

SCORECARDS = {
    "same_season": {
        "title": "Same season, different year",
        "subtitle": "AIFS Single v1.1 · 49r1 May 13–20, 2025 vs 50r1 May 13–20, 2026",
        "filename": "aifs_v11_same_season_forecast_scorecard.html",
        "cohort49": "13–20 May 2025 · final ERA5 reference",
        "cohort50": "13–20 May 2026 · ERA5T reference",
        "question": (
            "How do forecasts initialized in the same May window compare after "
            "each is verified against ERA5/ERA5T?"
        ),
        "reference_note": (
            "49r1 is verified against final ERA5 (2025); 50r1 is verified "
            "against ERA5T (2026). This comparison includes interannual weather "
            "and reference-release differences."
        ),
    },
    "cutover": {
        "title": "Within-2026 cutover",
        "subtitle": "AIFS Single v1.1 · 49r1 May 4–11 vs 50r1 May 13–20, 2026",
        "filename": "aifs_v11_cutover_forecast_scorecard.html",
        "cohort49": "4–11 May 2026 · ERA5T reference",
        "cohort50": "13–20 May 2026 · ERA5T reference",
        "question": (
            "How do forecasts initialized immediately before and after the "
            "50r1 cutover compare against the same ERA5T reference?"
        ),
        "reference_note": (
            "Both cohorts are verified against ERA5T. The cases are adjacent "
            "weather periods, not paired forecasts of identical atmospheric states."
        ),
    },
}

REGION_LABELS = {
    "global": "Global",
    "tropics": "Tropics",
    "northern_extratropics": "N. extratropics",
    "southern_extratropics": "S. extratropics",
    "global_land": "Global land",
    "global_ocean": "Global ocean",
}


def _fmt(value: float, *, signed: bool = False, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "—"
    prefix = "+" if signed and value > 0 else ""
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 0.01 or magnitude >= 10_000):
        return f"{prefix}{value:.2e}"
    return f"{prefix}{value:.{digits}f}"


def _color(value: float, limit: float, *, reverse: bool = False) -> str:
    if not np.isfinite(value):
        return "#e6dfd6"
    scaled = float(np.clip(value / limit, -1.0, 1.0))
    if reverse:
        scaled = -scaled
    neutral = np.array([247, 246, 246], dtype=float)
    positive = np.array([103, 0, 31], dtype=float)
    negative = np.array([5, 48, 97], dtype=float)
    target = positive if scaled > 0 else negative
    rgb = neutral * (1 - abs(scaled)) + target * abs(scaled)
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in rgb)


def _text_color(value: float, limit: float) -> str:
    return "#f8f2ea" if np.isfinite(value) and abs(value) / limit > 0.58 else "#1c1c1c"


def _variable_groups() -> list[tuple[str, list[str]]]:
    groups = []
    for parameter, label in (
        ("z", "Geopotential"),
        ("t", "Temperature"),
        ("u", "U wind"),
        ("v", "V wind"),
        ("w", "Vertical velocity"),
        ("q", "Specific humidity"),
    ):
        groups.append(
            (
                label,
                [
                    variable
                    for variable in EVALUATION_VARIABLES
                    if variable.startswith(f"{parameter}_")
                ],
            )
        )
    groups.append(
        (
            "Surface and soil",
            [
                variable
                for variable in EVALUATION_VARIABLES
                if "_" not in variable
                or variable.startswith(("stl", "swvl"))
            ],
        )
    )
    return groups


def _heatmap(
    data: pd.DataFrame,
    *,
    value_column: str,
    low_column: str,
    high_column: str,
    significant_column: str,
    limit: float,
    units: str,
    reverse: bool = False,
) -> str:
    lookup = data.set_index(["variable", "lead_hours"])
    rows = []
    for group_label, variables in _variable_groups():
        rows.append(
            f'<tr class="group-row"><th colspan="{len(LEADS)+1}">'
            f"{html.escape(group_label)}</th></tr>"
        )
        for variable in variables:
            highlight = " highlight-row" if variable in HIGHLIGHT_VARIABLES else ""
            cells = [
                f'<th class="field{highlight}">{html.escape(variable.replace("_", ""))}</th>'
            ]
            for lead in LEADS:
                row = lookup.loc[(variable, lead)]
                value = float(row[value_column])
                low = float(row[low_column])
                high = float(row[high_column])
                significant = bool(row[significant_column])
                classes = ["heat-cell"]
                if significant:
                    classes.append("significant")
                if variable in HIGHLIGHT_VARIABLES:
                    classes.append("highlight-cell")
                title = (
                    f"{variable}, day {lead//24}: {_fmt(value, signed=True)} {units}; "
                    f"95% CI [{_fmt(low, signed=True)}, {_fmt(high, signed=True)}]"
                )
                cells.append(
                    f'<td class="{" ".join(classes)}" '
                    f'style="background:{_color(value, limit, reverse=reverse)};'
                    f'color:{_text_color(value, limit)}" '
                    f'title="{html.escape(title)}">'
                    f"{_fmt(value, signed=True, digits=1)}</td>"
                )
            rows.append(f'<tr class="{highlight.strip()}">{"".join(cells)}</tr>')
    headers = "".join(f"<th>D{lead//24}</th>" for lead in LEADS)
    return (
        '<div class="heatmap-scroll"><table class="heatmap">'
        f"<thead><tr><th>Field</th>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _q_cards(scores: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    def estimate_cell(
        value: float,
        low: float,
        high: float,
        *,
        signed: bool = False,
        percent: bool = False,
    ) -> str:
        suffix = "%" if percent else ""
        digits = 1 if percent else 3
        title = (
            f"95% CI [{_fmt(low, signed=signed, digits=digits)}, "
            f"{_fmt(high, signed=signed, digits=digits)}]{suffix}"
        )
        return (
            f'<td title="{html.escape(title)}">'
            f"{_fmt(value, signed=signed, digits=digits)}{suffix}</td>"
        )

    cards = []
    for variable in HIGHLIGHT_VARIABLES:
        variable_scores = scores[
            (scores["variable"] == variable) & (scores["region"] == "global")
        ]
        variable_comparisons = comparisons[
            (comparisons["variable"] == variable)
            & (comparisons["region"] == "global")
        ].set_index("lead_hours")
        score_lookup = variable_scores.set_index(["cohort", "lead_hours"])
        rows = []
        for lead in LEADS:
            score49 = score_lookup.loc[("49r1", lead)]
            score50 = score_lookup.loc[("50r1", lead)]
            difference = variable_comparisons.loc[lead]
            rows.append(
                "<tr>"
                f"<td>D{lead//24}</td>"
                + estimate_cell(
                    float(score49["rmse"]),
                    float(score49["rmse_ci95_low"]),
                    float(score49["rmse_ci95_high"]),
                )
                + estimate_cell(
                    float(score50["rmse"]),
                    float(score50["rmse_ci95_low"]),
                    float(score50["rmse_ci95_high"]),
                )
                + estimate_cell(
                    float(difference["relative_rmse_change_percent"]),
                    float(difference["relative_rmse_change_ci95_low"]),
                    float(difference["relative_rmse_change_ci95_high"]),
                    signed=True,
                    percent=True,
                )
                + estimate_cell(
                    float(score49["bias"]),
                    float(score49["bias_ci95_low"]),
                    float(score49["bias_ci95_high"]),
                    signed=True,
                )
                + estimate_cell(
                    float(score50["bias"]),
                    float(score50["bias_ci95_low"]),
                    float(score50["bias_ci95_high"]),
                    signed=True,
                )
                + estimate_cell(
                    float(difference["bias_difference_50r1_minus_49r1"]),
                    float(difference["bias_difference_ci95_low"]),
                    float(difference["bias_difference_ci95_high"]),
                    signed=True,
                )
                + "</tr>"
            )
        units = html.escape(str(variable_scores.iloc[0]["units"]))
        cards.append(
            '<section class="q-card">'
            f'<div class="q-title"><span>{html.escape(variable.replace("_", ""))}</span>'
            f"<small>{units} · global · n=32/cohort</small></div>"
            '<div class="table-scroll"><table class="detail-table">'
            "<thead><tr><th>Lead</th><th>49r1 RMSE</th><th>50r1 RMSE</th>"
            "<th>ΔRMSE</th><th>49r1 bias</th><th>50r1 bias</th><th>Δbias</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            "</section>"
        )
    return "".join(cards)


def _interactive_payload(
    scores: pd.DataFrame, comparisons: pd.DataFrame
) -> dict[str, Any]:
    def safe(value: Any) -> float | None:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    score_lookup = scores.set_index(["cohort", "variable", "region", "lead_hours"])
    comparison_lookup = comparisons.set_index(["variable", "region", "lead_hours"])
    payload: dict[str, Any] = {}
    for variable in EVALUATION_VARIABLES:
        payload[variable] = {}
        for region in REGIONS:
            rows = []
            for lead in LEADS:
                row49 = score_lookup.loc[("49r1", variable, region, lead)]
                row50 = score_lookup.loc[("50r1", variable, region, lead)]
                difference = comparison_lookup.loc[(variable, region, lead)]
                rows.append(
                    {
                        "lead": lead,
                        "units": str(row49["units"]),
                        "rmse49": safe(row49["rmse"]),
                        "rmse49lo": safe(row49["rmse_ci95_low"]),
                        "rmse49hi": safe(row49["rmse_ci95_high"]),
                        "rmse50": safe(row50["rmse"]),
                        "rmse50lo": safe(row50["rmse_ci95_low"]),
                        "rmse50hi": safe(row50["rmse_ci95_high"]),
                        "drmse": safe(difference["relative_rmse_change_percent"]),
                        "drmselo": safe(
                            difference["relative_rmse_change_ci95_low"]
                        ),
                        "drmsehi": safe(
                            difference["relative_rmse_change_ci95_high"]
                        ),
                        "bias49": safe(row49["bias"]),
                        "bias49lo": safe(row49["bias_ci95_low"]),
                        "bias49hi": safe(row49["bias_ci95_high"]),
                        "bias50": safe(row50["bias"]),
                        "bias50lo": safe(row50["bias_ci95_low"]),
                        "bias50hi": safe(row50["bias_ci95_high"]),
                        "dbias": safe(
                            difference["bias_difference_50r1_minus_49r1"]
                        ),
                        "dbiaslo": safe(difference["bias_difference_ci95_low"]),
                        "dbiashi": safe(difference["bias_difference_ci95_high"]),
                        "n49": int(row49["forecast_case_count"]),
                        "n50": int(row50["forecast_case_count"]),
                        "area49": safe(row49["mean_valid_area_fraction"]),
                        "area50": safe(row50["mean_valid_area_fraction"]),
                    }
                )
            payload[variable][region] = rows
    return payload


def _normalization_scales() -> dict[str, float]:
    payload = json.loads(NORMALIZATION_JSON.read_text(encoding="utf-8"))
    return {
        variable: float(record["normalization_scale"])
        for variable, record in payload["variables"].items()
    }


def render_scorecard(
    comparison: str,
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    metadata = SCORECARDS[comparison]
    current_scores = scores[scores["comparison"] == comparison].copy()
    current_comparisons = comparisons[
        comparisons["comparison"] == comparison
    ].copy()
    scales = _normalization_scales()
    current_comparisons["normalized_bias_difference"] = [
        value / scales[variable]
        for variable, value in zip(
            current_comparisons["variable"],
            current_comparisons["bias_difference_50r1_minus_49r1"],
            strict=True,
        )
    ]
    current_comparisons["normalized_bias_ci_low"] = [
        value / scales[variable]
        for variable, value in zip(
            current_comparisons["variable"],
            current_comparisons["bias_difference_ci95_low"],
            strict=True,
        )
    ]
    current_comparisons["normalized_bias_ci_high"] = [
        value / scales[variable]
        for variable, value in zip(
            current_comparisons["variable"],
            current_comparisons["bias_difference_ci95_high"],
            strict=True,
        )
    ]
    global_rows = current_comparisons[
        current_comparisons["region"] == "global"
    ]
    rmse_map = _heatmap(
        global_rows,
        value_column="relative_rmse_change_percent",
        low_column="relative_rmse_change_ci95_low",
        high_column="relative_rmse_change_ci95_high",
        significant_column="relative_rmse_change_significant_95",
        limit=20.0,
        units="%",
    )
    bias_map = _heatmap(
        global_rows,
        value_column="normalized_bias_difference",
        low_column="normalized_bias_ci_low",
        high_column="normalized_bias_ci_high",
        significant_column="bias_difference_significant_95",
        limit=0.5,
        units="training σ",
    )
    q_cards = _q_cards(current_scores, current_comparisons)
    payload = _interactive_payload(current_scores, current_comparisons)
    variable_options = "".join(
        f'<option value="{html.escape(variable)}"'
        f'{" selected" if variable == "q_50" else ""}>'
        f'{html.escape(variable.replace("_", ""))}</option>'
        for variable in EVALUATION_VARIABLES
    )
    region_options = "".join(
        f'<option value="{region}">{html.escape(REGION_LABELS[region])}</option>'
        for region in REGIONS
    )
    methods = METHODS_PATH.read_text(encoding="utf-8")
    methods_excerpt = html.escape(methods)
    other_comparison = "cutover" if comparison == "same_season" else "same_season"
    other_metadata = SCORECARDS[other_comparison]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(metadata["title"])} · AIFS v1.1 forecast scorecard</title>
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
  .section-subtitle {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 0 auto;
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
  nav {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 20px auto 4px;
  }}
  nav a {{
    display: inline-block;
    padding: 5px 9px;
    margin: 0 4px 6px 0;
    border: 1px solid var(--ink);
    background: var(--bg);
    color: var(--ink);
    text-decoration: none;
  }}
  nav a:hover {{ background: var(--ink); color: var(--bg); }}
  .lead-key {{
    width: calc(100% - 32px);
    max-width: 1120px;
    color: var(--ink);
    font-size: inherit;
    margin: 8px auto 12px;
    letter-spacing: .02em;
  }}
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
  .gradient {{ width: 100%; height: 14px; border: 1px solid var(--ink); }}
  .table-wrap, .q-card, .explorer-panel {{
    width: calc(100% - 32px);
    overflow-x: auto;
    border: 1px solid var(--ink);
    margin: 14px auto 0;
    background: var(--bg);
  }}
  .table-wrap {{ max-width: 1120px; }}
  .q-grid {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 0 auto;
  }}
  .q-grid .q-card {{
    width: 100%;
    max-width: none;
  }}
  .explorer-panel {{ max-width: 1120px; }}
  .q-card {{ max-width: 1120px; border-left: 3px solid var(--blue); }}
  .q-title {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 10px;
    background: #ECEBFF;
    border-bottom: 1px solid var(--ink);
  }}
  .q-title span, .q-title small {{ font-size: inherit; font-weight: 400; color: var(--ink); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    white-space: nowrap;
    background: var(--bg);
  }}
  th, td {{ border: 1px solid var(--line); padding: 3px 5px; }}
  th {{
    background: var(--ink);
    color: var(--bg);
    font-weight: 400;
    text-align: center;
    position: sticky;
    top: 0;
    z-index: 3;
  }}
  .detail-table th, .detail-table td {{
    padding: 5px 7px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .detail-table th:first-child, .detail-table td:first-child {{ text-align: left; }}
  .heat-intro {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 8px auto 0;
    font-size: inherit;
  }}
  .heatmap {{ min-width: 790px; table-layout: fixed; }}
  .heatmap th, .heatmap td {{
    height: 24px;
    min-width: 52px;
    padding: 2px 4px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }}
  .heatmap .field {{
    position: sticky;
    left: 0;
    background: #F3ECDF;
    color: var(--ink);
    text-align: left;
    width: 92px;
    min-width: 92px;
    z-index: 2;
    font-weight: 400;
  }}
  .group-row th {{
    text-align: left;
    background: var(--soft);
    color: var(--ink);
    height: 26px;
    border-top: 1px solid var(--ink);
  }}
  .heat-cell {{ cursor: help; }}
  .heat-cell:hover {{ outline: 2px solid var(--ink); position: relative; z-index: 2; }}
  .heat-cell.significant {{ box-shadow: inset 0 0 0 2px var(--ink); }}
  .highlight-cell {{ outline: 2px solid var(--blue); outline-offset: -2px; }}
  .highlight-row .field {{ background: #ECEBFF; border-left: 3px solid var(--blue); }}
  .explorer-panel {{ padding: 12px 14px 14px; overflow: visible; }}
  .explorer-controls {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
  select {{
    border: 1px solid var(--ink);
    border-radius: 0;
    background: var(--bg);
    padding: 5px 26px 5px 7px;
    color: var(--ink);
    font: inherit;
  }}
  .table-scroll, .heatmap-scroll {{ overflow-x: auto; }}
  .ci {{ color: var(--muted); font-size: 11px; }}
  .notes {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 18px auto 0;
    padding-top: 18px;
    border-top: 1px solid var(--ink);
    color: var(--muted);
  }}
  details {{ border: 1px solid var(--ink); padding: 10px 12px; margin: 0 0 10px; }}
  summary {{ font-weight: 400; cursor: pointer; }}
  pre {{
    white-space: pre-wrap;
    font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--ink);
    max-height: 420px;
    overflow: auto;
  }}
  .footer {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 34px auto 0;
    padding-top: 18px;
    border-top: 1px solid var(--ink);
    color: var(--muted);
    font-size: inherit;
  }}
  code {{ background: #EFE8DF; color: var(--ink); padding: 1px 4px; }}
  @media (max-width: 760px) {{
    .evidence-order {{ grid-template-columns: 1fr; }}
    .legend {{ grid-template-columns: 1fr; }}
    .legend-label, .legend-label-left {{ text-align: left; }}
    .table-wrap {{ overflow-x: auto; }}
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
  <div class="eyebrow">AIFS v1.1 / direct forecast verification</div>
  <h1>{html.escape(metadata["title"])}</h1>
  <div class="subtitle">
    {html.escape(metadata["subtitle"])}. Direct ten-day AIFS forecasts verified
    every 24 hours. Each cohort has 32 initializations; every score cell has a
    case-bootstrap 95% interval.
  </div>
  <div class="evidence-order">
    <div>
      <strong>01 / 49r1 initialization</strong>
      <span>{html.escape(metadata["cohort49"])}</span>
    </div>
    <div>
      <strong>02 / 50r1 initialization</strong>
      <span>{html.escape(metadata["cohort50"])}</span>
    </div>
  </div>
</div>
<div class="warning">
  <strong>{html.escape(metadata["question"])}</strong>
  {html.escape(metadata["reference_note"])}
</div>
<nav>
  <a href="#humidity">q50 / q100</a>
  <a href="#rmse">RMSE</a>
  <a href="#bias">Bias</a>
  <a href="#explorer">Raw score explorer</a>
  <a href="#methods">Methods</a>
  <a href="{html.escape(other_metadata["filename"])}">{html.escape(other_metadata["title"])}</a>
</nav>
<div class="lead-key">lead hours: 24 48 72 96 120 144 168 192 216 240 · 96 instantaneous fields · global + 5 regional/land-ocean domains</div>

<section id="humidity">
<h2>Specific humidity / q50 and q100</h2>
<div class="subtitle section-subtitle">
  Area-weighted global RMSE and bias. Bias is forecast − ERA5/ERA5T;
  Δ values are 50r1 − 49r1. These two fields are highlighted throughout.
</div>
<div class="q-grid">{q_cards}</div>
</section>

<section id="rmse">
<h2>Global RMSE change</h2>
<div class="legend" aria-label="RMSE colour scale">
  <span class="legend-label legend-label-left">
    50r1 lower RMSE (better)
    <strong>−20 %</strong>
  </span>
  <div class="gradient" style="background:linear-gradient(90deg,#053061,#f7f6f6,#67001f);"></div>
  <span class="legend-label">
    50r1 higher RMSE (worse)
    <strong>+20 %</strong>
  </span>
</div>
<div class="subtitle section-subtitle">
  Formula: <code>100 × (RMSE50 / RMSE49 − 1)</code>. Colours saturate at
  ±20%; an inset black frame means the 95% interval excludes zero.
</div>
<div class="heat-intro">All 96 matched instantaneous outputs; hover a cell for the estimate and interval.</div>
<div class="table-wrap">{rmse_map}</div>
</section>

<section id="bias">
<h2>Global bias shift</h2>
<div class="legend" aria-label="Bias colour scale">
  <span class="legend-label legend-label-left">
    Negative bias shift
    <strong>−0.5 training σ</strong>
  </span>
  <div class="gradient" style="background:linear-gradient(90deg,#053061,#f7f6f6,#67001f);"></div>
  <span class="legend-label">
    Positive bias shift
    <strong>+0.5 training σ</strong>
  </span>
</div>
<div class="subtitle section-subtitle">
  Formula: <code>(bias50 − bias49) / AIFS training normalization scale</code>.
  Raw bias and intervals remain available below.
</div>
<div class="heat-intro">All 96 matched instantaneous outputs; an inset black frame means the 95% interval excludes zero.</div>
<div class="table-wrap">{bias_map}</div>
</section>

<section id="explorer">
<h2>Raw score explorer</h2>
<div class="subtitle section-subtitle">
  Select any field and domain for raw RMSE, bias, case-bootstrap 95%
  intervals, sample counts and valid-area coverage.
</div>
<div class="explorer-panel"><div class="explorer-controls"><label>Field <select id="field-select">{variable_options}</select></label><label>Domain <select id="region-select">{region_options}</select></label></div><div class="table-scroll"><table class="detail-table"><thead><tr><th>Lead</th><th>49r1 RMSE</th><th>50r1 RMSE</th><th>ΔRMSE</th><th>49r1 bias</th><th>50r1 bias</th><th>Δbias</th><th>n</th><th>valid area</th></tr></thead><tbody id="explorer-body"></tbody></table></div></div>
</section>

<section id="methods">
<h2>Reading this scorecard</h2>
<div class="notes"><details open><summary>Scope and uncertainty</summary><p>Every complete cell contains 32 forecast cases per cohort. Cohort RMSE is pooled from per-case area-weighted mean squared errors; bias is the mean per-case area-weighted error. Intervals use 4,000 fixed-seed case-bootstrap draws. The cohorts are resampled independently and are not treated as causal pairs.</p><p>Soil scores over ocean are retained for inventory completeness but have limited physical interpretation. The explorer reports valid-area coverage for every field and domain.</p></details>
<details><summary>Pre-run audit and methods</summary><pre>{methods_excerpt}</pre></details></div>
</section>

<div class="footer">Machine-readable results: <a href="../data/processed/aifs_v11_forecast_cohort_scores.csv">cohort scores</a> · <a href="../data/processed/aifs_v11_forecast_cycle_comparisons.csv">cycle comparisons</a> · <a href="../data/processed/aifs_v11_forecast_case_metrics.csv">per-case metrics</a>. Model: ecmwf/aifs-single-1.1, revision {html.escape("049b9ab1ccac3382b6332870ae550fd20a432faf")}.</div>
</main>
<script>
const SCORE_DATA={json.dumps(payload, separators=(",", ":"), allow_nan=False)};
const fmt=(x,d=3)=>Number.isFinite(x)?(Math.abs(x)>0&&Math.abs(x)<0.01?x.toExponential(2):x.toFixed(d)):"—";
function interval(x,lo,hi,percent=false){{const unit=percent?"%":"";return `${{fmt(x,percent?1:3)}}${{unit}} <span class="ci">[${{fmt(lo,percent?1:3)}}, ${{fmt(hi,percent?1:3)}}]${{unit}}</span>`}}
function renderExplorer(){{
 const variable=document.getElementById("field-select").value;
 const region=document.getElementById("region-select").value;
 document.getElementById("explorer-body").innerHTML=SCORE_DATA[variable][region].map(r=>`<tr><td>D${{r.lead/24}} <span class="ci">${{r.units}}</span></td><td>${{interval(r.rmse49,r.rmse49lo,r.rmse49hi)}}</td><td>${{interval(r.rmse50,r.rmse50lo,r.rmse50hi)}}</td><td>${{interval(r.drmse,r.drmselo,r.drmsehi,true)}}</td><td>${{interval(r.bias49,r.bias49lo,r.bias49hi)}}</td><td>${{interval(r.bias50,r.bias50lo,r.bias50hi)}}</td><td>${{interval(r.dbias,r.dbiaslo,r.dbiashi)}}</td><td>${{r.n49}} + ${{r.n50}}</td><td>${{(100*Math.min(r.area49,r.area50)).toFixed(1)}}%</td></tr>`).join("");
}}
document.getElementById("field-select").addEventListener("change",renderExplorer);
document.getElementById("region-select").addEventListener("change",renderExplorer);
renderExplorer();
</script>
</body></html>"""


def build_scorecards() -> dict[str, Any]:
    scores = pd.read_parquet(SCORES_PATH)
    comparisons = pd.read_parquet(COMPARISONS_PATH)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for comparison, metadata in SCORECARDS.items():
        target = OUTPUTS / metadata["filename"]
        target.write_text(
            render_scorecard(comparison, scores, comparisons),
            encoding="utf-8",
        )
        outputs[comparison] = {
            "path": str(target),
            "bytes": target.stat().st_size,
        }
    return outputs


if __name__ == "__main__":
    print(json.dumps(build_scorecards(), indent=2))
