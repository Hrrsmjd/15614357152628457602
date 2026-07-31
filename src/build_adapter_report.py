"""Render a self-contained comparison of paired frozen-adapter experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aifs_forecast_experiment as base
import aifs_adapter_experiment as experiment
import build_forecast_scorecards as baseline_renderer
from aifs_forecast_experiment import LEADS, OUTPUTS, PROCESSED, ROOT


METHODS_PATH = ROOT / "forecast_adapter_methods.md"
REPORT_PATH = OUTPUTS / "aifs_v11_frozen_adapter_comparison.html"
FIGURE_PATH = OUTPUTS / "figures" / "aifs_v11_frozen_adapter_comparison.png"
SUMMARY_PATH = PROCESSED / "aifs_v11_frozen_adapter_summary.csv"
COMPARISON_PATH = (
    PROCESSED / "aifs_v11_frozen_adapter_paired_comparisons.parquet"
)
SELECTED_ADAPTER = "hybrid_q50_affine_extratropics_q100_q150"
FINAL_SCORECARD_PATH = (
    OUTPUTS / "aifs_v11_corrected_vs_uncorrected_forecast_scorecard.html"
)
FINAL_SCORES_PATH = (
    PROCESSED / "aifs_v11_corrected_vs_uncorrected_scores.parquet"
)
FINAL_COMPARISONS_PATH = (
    PROCESSED / "aifs_v11_corrected_vs_uncorrected_comparisons.parquet"
)

ADAPTER_LABELS = {
    "residual_all_no_q50": "All-field residual / no q50",
    "residual_all_no_q50_half": "Half-strength all-field / no q50",
    "residual_q100_q150": "Targeted q100 + q150",
    "hybrid_q50_affine_q100_q150": "Hybrid q50 affine + q100/q150",
    "hybrid_q50_affine_extratropics_q100_q150": (
        "Hybrid extratropical q50 + q100/q150"
    ),
}

FINAL_SCORECARD_METADATA = {
    "title": "Corrected versus uncorrected 50r1",
    "subtitle": (
        "AIFS Single v1.1 · identical 50r1 initializations from "
        "17–20 May 2026"
    ),
    "filename": FINAL_SCORECARD_PATH.name,
    "cohort49": (
        "Uncorrected 50r1 inputs · 16 initializations · ERA5T reference"
    ),
    "cohort50": (
        "Frozen extratropical-q50 + q100/q150 adapter · the same 16 "
        "initializations · ERA5T reference"
    ),
    "question": (
        "Does the selected frozen input adapter improve forecasts from the "
        "same 50r1 atmospheric states?"
    ),
    "reference_note": (
        "This is the primary treatment comparison: every corrected forecast "
        "is paired with its uncorrected counterpart and verified against the "
        "same ERA5T fields."
    ),
}


def _fmt(value: float, *, signed: bool = False, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    prefix = "+" if signed and value > 0 else ""
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 0.01 or magnitude >= 10_000):
        return f"{prefix}{value:.2e}"
    return f"{prefix}{value:.{digits}f}"


def _color(value: float, limit: float = 10.0) -> str:
    if not np.isfinite(value):
        return "#e6dfd6"
    scaled = float(np.clip(value / limit, -1.0, 1.0))
    neutral = np.array([247, 246, 246], dtype=float)
    worse = np.array([103, 0, 31], dtype=float)
    better = np.array([5, 48, 97], dtype=float)
    target = worse if scaled > 0 else better
    rgb = neutral * (1 - abs(scaled)) + target * abs(scaled)
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in rgb)


def _text_color(value: float, limit: float = 10.0) -> str:
    return "#f8f2ea" if abs(value) / limit > 0.58 else "#1c1c1c"


def _load_results(
    adapter_names: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    summaries = {}
    comparisons = {}
    for adapter in adapter_names:
        directory = experiment.adapter_directory(adapter)
        summary_path = directory / "summary.json"
        comparison_path = directory / "paired_comparisons.parquet"
        if not summary_path.exists() or not comparison_path.exists():
            raise FileNotFoundError(
                f"Incomplete adapter result for {adapter}: "
                f"{summary_path}, {comparison_path}"
            )
        summaries[adapter] = json.loads(summary_path.read_text(encoding="utf-8"))
        comparisons[adapter] = pd.read_parquet(comparison_path)
    return summaries, comparisons


def _summary_table(
    adapter_names: list[str], summaries: dict[str, dict[str, Any]], best: str
) -> str:
    rows = []
    for adapter in adapter_names:
        current = summaries[adapter]["summary"]
        marker = " ✓ selected" if adapter == best else ""
        rows.append(
            "<tr>"
            f"<th>{html.escape(ADAPTER_LABELS.get(adapter, adapter))}"
            f"{html.escape(marker)}</th>"
            f"<td>{_fmt(current['early_global_q100_q150_mean_percent'], signed=True)}</td>"
            f"<td>{_fmt(current['early_global_q50_mean_percent'], signed=True)}</td>"
            f"<td>{_fmt(current['guardrail_mean_percent'], signed=True)}</td>"
            f"<td>{_fmt(current['guardrail_worst_percent'], signed=True)}</td>"
            f"<td>{_fmt(current['global_all_cell_mean_percent'], signed=True)}</td>"
            f"<td>{current['significant_cells_improved']}</td>"
            f"<td>{current['significant_cells_worsened']}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table class="summary-table"><thead><tr>'
        "<th>Adapter</th><th>q100/q150<br>early mean</th>"
        "<th>q50<br>early mean</th><th>guardrail<br>mean</th>"
        "<th>guardrail<br>worst cell</th><th>all global<br>mean</th>"
        "<th>significant<br>improvements</th>"
        "<th>significant<br>harms</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _q_tables(
    adapter_names: list[str], comparisons: dict[str, pd.DataFrame]
) -> str:
    sections = []
    for variable in experiment.Q_TARGETS:
        header = "".join(f"<th>D{lead // 24}</th>" for lead in LEADS)
        rows = []
        for adapter in adapter_names:
            data = comparisons[adapter]
            data = data[
                data["variable"].eq(variable) & data["region"].eq("global")
            ].set_index("lead_hours")
            cells = []
            for lead in LEADS:
                row = data.loc[lead]
                value = float(row["relative_rmse_change_percent"])
                low = float(row["relative_rmse_change_ci95_low"])
                high = float(row["relative_rmse_change_ci95_high"])
                significant = bool(
                    row["relative_rmse_change_significant_95"]
                )
                title = (
                    f"{variable}, day {lead // 24}: "
                    f"{value:+.2f}% [{low:+.2f}, {high:+.2f}]"
                )
                classes = "heat significant" if significant else "heat"
                cells.append(
                    f'<td class="{classes}" title="{html.escape(title)}" '
                    f'style="background:{_color(value, 20)};'
                    f'color:{_text_color(value, 20)}">'
                    f"{_fmt(value, signed=True, digits=1)}</td>"
                )
            rows.append(
                f"<tr><th>{html.escape(ADAPTER_LABELS.get(adapter, adapter))}"
                f"</th>{''.join(cells)}</tr>"
            )
        sections.append(
            f"<section><h3>{html.escape(variable.replace('_', ''))}</h3>"
            '<div class="table-scroll"><table class="heat-table adapter-table">'
            f"<thead><tr><th>Adapter</th>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></section>"
        )
    return "".join(sections)


def _guardrail_table(
    adapter_names: list[str], comparisons: dict[str, pd.DataFrame]
) -> str:
    header = "".join(
        f"<th>{html.escape(variable.replace('_', ''))}</th>"
        for variable in experiment.GUARDRAIL_VARIABLES
    )
    rows = []
    for adapter in adapter_names:
        data = comparisons[adapter]
        data = data[
            data["region"].isin(("global", "northern_extratropics"))
            & data["lead_hours"].le(120)
            & data["variable"].isin(experiment.GUARDRAIL_VARIABLES)
        ]
        means = data.groupby("variable")[
            "relative_rmse_change_percent"
        ].mean()
        cells = []
        for variable in experiment.GUARDRAIL_VARIABLES:
            value = float(means.loc[variable])
            cells.append(
                f'<td style="background:{_color(value, 5)};'
                f'color:{_text_color(value, 5)}">'
                f"{_fmt(value, signed=True)}</td>"
            )
        rows.append(
            f"<tr><th>{html.escape(ADAPTER_LABELS.get(adapter, adapter))}</th>"
            f"{''.join(cells)}</tr>"
        )
    return (
        '<div class="table-scroll"><table class="heat-table adapter-table"><thead><tr>'
        f"<th>Adapter</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _full_heatmap(data: pd.DataFrame) -> str:
    global_rows = data[data["region"].eq("global")].set_index(
        ["variable", "lead_hours"]
    )
    rows = []
    for group, variables in baseline_renderer._variable_groups():
        rows.append(
            f'<tr class="group"><th colspan="{len(LEADS) + 1}">'
            f"{html.escape(group)}</th></tr>"
        )
        for variable in variables:
            cells = [f"<th>{html.escape(variable.replace('_', ''))}</th>"]
            for lead in LEADS:
                current = global_rows.loc[(variable, lead)]
                value = float(current["relative_rmse_change_percent"])
                low = float(current["relative_rmse_change_ci95_low"])
                high = float(current["relative_rmse_change_ci95_high"])
                significant = bool(
                    current["relative_rmse_change_significant_95"]
                )
                classes = "heat significant" if significant else "heat"
                title = (
                    f"{variable}, D{lead // 24}: {value:+.2f}% "
                    f"[{low:+.2f}, {high:+.2f}]"
                )
                cells.append(
                    f'<td class="{classes}" title="{html.escape(title)}" '
                    f'style="background:{_color(value, 10)};'
                    f'color:{_text_color(value, 10)}">'
                    f"{_fmt(value, signed=True, digits=1)}</td>"
                )
            rows.append(f"<tr>{''.join(cells)}</tr>")
    header = "".join(f"<th>D{lead // 24}</th>" for lead in LEADS)
    return (
        '<div class="table-scroll full"><table class="heat-table field-table">'
        f"<thead><tr><th>Field</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _plot_comparison(
    adapter_names: list[str],
    comparisons: dict[str, pd.DataFrame],
) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.2), constrained_layout=True
    )
    for ax, variable in zip(
        axes.flat[:3], experiment.Q_TARGETS, strict=True
    ):
        for adapter in adapter_names:
            data = comparisons[adapter]
            data = data[
                data["variable"].eq(variable) & data["region"].eq("global")
            ].sort_values("lead_hours")
            ax.plot(
                data["lead_hours"] / 24,
                data["relative_rmse_change_percent"],
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                label=ADAPTER_LABELS.get(adapter, adapter),
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(variable.replace("_", "").upper())
        ax.set_xlabel("Lead (days)")
        ax.set_ylabel("Paired RMSE change (%)")
        ax.grid(alpha=0.25)
    guardrail_ax = axes.flat[3]
    positions = np.arange(len(adapter_names))
    values = []
    for adapter in adapter_names:
        data = comparisons[adapter]
        data = data[
            data["region"].isin(("global", "northern_extratropics"))
            & data["lead_hours"].le(120)
            & data["variable"].isin(experiment.GUARDRAIL_VARIABLES)
        ]
        values.append(float(data["relative_rmse_change_percent"].mean()))
    guardrail_ax.barh(
        positions,
        values,
        color=["#053061" if value < 0 else "#67001f" for value in values],
    )
    guardrail_ax.axvline(0, color="black", linewidth=0.8)
    guardrail_ax.set_yticks(
        positions,
        [ADAPTER_LABELS.get(adapter, adapter) for adapter in adapter_names],
    )
    guardrail_ax.set_xlabel("Mean paired RMSE change (%)")
    guardrail_ax.set_title("Operational guardrails · D1–D5")
    guardrail_ax.grid(axis="x", alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Frozen IFS 50r1 adapters versus identical uncorrected AIFS v1.1 cases"
    )
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def _write_consolidated_results(
    adapter_names: list[str],
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, pd.DataFrame],
    *,
    best: str,
) -> None:
    summary_rows = []
    comparison_frames = []
    for adapter in adapter_names:
        summary_rows.append(
            {
                "adapter": adapter,
                "adapter_label": ADAPTER_LABELS.get(adapter, adapter),
                "selected": adapter == best,
                "description": summaries[adapter]["description"],
                **summaries[adapter]["summary"],
            }
        )
        current = comparisons[adapter].copy()
        current["adapter_label"] = ADAPTER_LABELS.get(adapter, adapter)
        current["selected"] = adapter == best
        comparison_frames.append(current)

    summary_frame = pd.DataFrame(summary_rows)
    comparison_frame = pd.concat(comparison_frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    base._write_frame_atomic(summary_frame, SUMMARY_PATH)
    base._write_frame_atomic(comparison_frame, COMPARISON_PATH)


def _final_case_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    adapter_path = (
        experiment.adapter_directory(SELECTED_ADAPTER)
        / "case_metrics.parquet"
    )
    baseline_path = PROCESSED / "aifs_v11_forecast_case_metrics.parquet"
    adapter_cases = pd.read_parquet(adapter_path)
    evaluation_ids = set(adapter_cases["initialization_id"])
    baseline_cases = pd.read_parquet(baseline_path)
    baseline_cases = baseline_cases[
        baseline_cases["initialization_id"].isin(evaluation_ids)
    ].copy()

    expected_rows = (
        16
        * len(base.EVALUATION_VARIABLES)
        * len(base.REGIONS)
        * len(base.LEADS)
    )
    if len(baseline_cases) != expected_rows or len(adapter_cases) != expected_rows:
        raise RuntimeError(
            "Unexpected final scorecard case rows: "
            f"{len(baseline_cases)}, {len(adapter_cases)}"
        )
    keys = ["initialization_id", "variable", "region", "lead_hours"]
    matched = baseline_cases[keys].merge(
        adapter_cases[keys],
        on=keys,
        validate="one_to_one",
    )
    if len(matched) != expected_rows:
        raise RuntimeError(
            f"Only {len(matched)} of {expected_rows} final rows matched"
        )
    return baseline_cases, adapter_cases


def _final_score_rows(
    baseline_cases: pd.DataFrame,
    adapter_cases: pd.DataFrame,
    *,
    bootstrap_draws: int,
) -> pd.DataFrame:
    cohorts = []
    for cohort, cases in (
        ("49r1", baseline_cases),
        ("50r1", adapter_cases),
    ):
        current = cases.copy()
        current["comparison"] = "cutover"
        current["cohort"] = cohort
        cohorts.append(current)
    membership_cases = pd.concat(cohorts, ignore_index=True)

    rows: list[dict[str, Any]] = []
    keys = [
        "comparison",
        "cohort",
        "variable",
        "units",
        "region",
        "lead_hours",
    ]
    for key, group in membership_cases.groupby(
        keys, sort=True, dropna=False
    ):
        comparison, cohort, variable, units, region, lead = key
        bias_values = group["bias"].to_numpy(dtype=float)
        mse_values = group["mean_squared_error"].to_numpy(dtype=float)
        finite_bias = bias_values[np.isfinite(bias_values)]
        finite_mse = mse_values[np.isfinite(mse_values)]
        bias_low, bias_high = base._bootstrap_interval(
            bias_values,
            transform="identity",
            seed_parts=("final-scorecard", *key, "bias"),
            draws=bootstrap_draws,
        )
        rmse_low, rmse_high = base._bootstrap_interval(
            mse_values,
            transform="sqrt",
            seed_parts=("final-scorecard", *key, "rmse"),
            draws=bootstrap_draws,
        )
        rows.append(
            {
                "comparison": comparison,
                "cohort": cohort,
                "variable": variable,
                "units": units,
                "region": region,
                "lead_hours": int(lead),
                "bias": float(np.mean(finite_bias)),
                "bias_ci95_low": bias_low,
                "bias_ci95_high": bias_high,
                "rmse": float(np.sqrt(np.mean(finite_mse))),
                "rmse_ci95_low": rmse_low,
                "rmse_ci95_high": rmse_high,
                "forecast_case_count": int(
                    group["initialization_id"].nunique()
                ),
                "valid_gridpoint_count": int(
                    group["valid_gridpoint_count"].sum()
                ),
                "mean_valid_area_fraction": float(
                    group["valid_area_fraction"].mean()
                ),
                "reference_products": ";".join(
                    sorted(group["reference_product"].unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def _final_comparison_rows() -> pd.DataFrame:
    paired = pd.read_parquet(
        experiment.adapter_directory(SELECTED_ADAPTER)
        / "paired_comparisons.parquet"
    ).copy()
    paired["comparison"] = "cutover"
    paired["rmse_49r1"] = paired["rmse_baseline"]
    paired["rmse_50r1"] = paired["rmse_adapter"]
    paired["bias_49r1"] = paired["bias_baseline"]
    paired["bias_50r1"] = paired["bias_adapter"]
    paired["bias_difference_50r1_minus_49r1"] = paired[
        "bias_difference_adapter_minus_baseline"
    ]
    paired["forecast_case_count_49r1"] = paired["forecast_case_count"]
    paired["forecast_case_count_50r1"] = paired["forecast_case_count"]
    paired["reference_products_49r1"] = "ERA5T"
    paired["reference_products_50r1"] = "ERA5T"
    columns = [
        "comparison",
        "variable",
        "units",
        "region",
        "lead_hours",
        "rmse_49r1",
        "rmse_50r1",
        "relative_rmse_change_percent",
        "relative_rmse_change_ci95_low",
        "relative_rmse_change_ci95_high",
        "relative_rmse_change_significant_95",
        "bias_49r1",
        "bias_50r1",
        "bias_difference_50r1_minus_49r1",
        "bias_difference_ci95_low",
        "bias_difference_ci95_high",
        "bias_difference_significant_95",
        "forecast_case_count_49r1",
        "forecast_case_count_50r1",
        "reference_products_49r1",
        "reference_products_50r1",
    ]
    return paired[columns]


def _final_scorecard_labels(page: str) -> str:
    replacements = {
        "Each cohort has 32 initializations; every score cell has a\n"
        "    case-bootstrap 95% interval.": (
            "The same 16 initializations are evaluated with uncorrected and "
            "corrected inputs; every score cell has a paired-bootstrap 95% "
            "interval."
        ),
        "<strong>01 / 49r1 initialization</strong>": (
            "<strong>01 / uncorrected 50r1 initialization</strong>"
        ),
        "<strong>02 / 50r1 initialization</strong>": (
            "<strong>02 / corrected 50r1 initialization</strong>"
        ),
        "q50 / q100": "q50 / q100 / q150",
        "Specific humidity / q50 and q100": (
            "Specific humidity / q50, q100 and q150"
        ),
        "Δ values are 50r1 − 49r1.": (
            "Δ values are corrected 50r1 − uncorrected 50r1."
        ),
        "These two fields are highlighted throughout.": (
            "These three fields are highlighted throughout."
        ),
        "n=32/cohort": "n=16 paired cases",
        "<th>49r1 RMSE</th>": "<th>uncorrected RMSE</th>",
        "<th>50r1 RMSE</th>": "<th>corrected RMSE</th>",
        "<th>49r1 bias</th>": "<th>uncorrected bias</th>",
        "<th>50r1 bias</th>": "<th>corrected bias</th>",
        "50r1 lower RMSE (better)": "corrected lower RMSE (better)",
        "50r1 higher RMSE (worse)": "corrected higher RMSE (worse)",
        "100 × (RMSE50 / RMSE49 − 1)": (
            "100 × (RMSEcorrected / RMSEuncorrected − 1)"
        ),
        "(bias50 − bias49) / AIFS training normalization scale": (
            "(biascorrected − biasuncorrected) / AIFS training "
            "normalization scale"
        ),
        (
            "Every complete cell contains 32 forecast cases per cohort. "
            "Cohort RMSE is pooled from per-case area-weighted mean squared "
            "errors; bias is the mean per-case area-weighted error. Intervals "
            "use 4,000 fixed-seed case-bootstrap draws. The cohorts are "
            "resampled independently and are not treated as causal pairs."
        ): (
            "Every complete cell contains 16 matched initialization pairs. "
            "RMSE is pooled from per-case area-weighted mean squared errors; "
            "bias is the mean per-case area-weighted error. Intervals use "
            "4,000 fixed-seed paired-bootstrap draws, applying the same "
            "resampled initialization indices to corrected and uncorrected "
            "errors."
        ),
        "../data/processed/aifs_v11_forecast_cohort_scores.csv": (
            "../data/processed/"
            "aifs_v11_corrected_vs_uncorrected_scores.csv"
        ),
        "../data/processed/aifs_v11_forecast_cycle_comparisons.csv": (
            "../data/processed/"
            "aifs_v11_corrected_vs_uncorrected_comparisons.csv"
        ),
        "../data/processed/aifs_v11_forecast_case_metrics.csv": (
            "../data/forecast_experiment/adapter_experiments/"
            f"{SELECTED_ADAPTER}/case_metrics.parquet"
        ),
    }
    for old, new in replacements.items():
        page = page.replace(old, new)
    page = page.replace(
        ">cycle comparisons</a>", ">paired comparisons</a>"
    )
    return page


def build_final_scorecard(*, bootstrap_draws: int = 4000) -> dict[str, Any]:
    baseline_cases, adapter_cases = _final_case_metrics()
    scores = _final_score_rows(
        baseline_cases,
        adapter_cases,
        bootstrap_draws=bootstrap_draws,
    )
    comparisons = _final_comparison_rows()
    base._write_frame_atomic(scores, FINAL_SCORES_PATH)
    base._write_frame_atomic(scores, FINAL_SCORES_PATH.with_suffix(".csv"))
    base._write_frame_atomic(comparisons, FINAL_COMPARISONS_PATH)
    base._write_frame_atomic(
        comparisons, FINAL_COMPARISONS_PATH.with_suffix(".csv")
    )

    final_scorecards = {
        "cutover": FINAL_SCORECARD_METADATA,
        "same_season": {
            "title": "Five-candidate ablation comparison",
            "filename": REPORT_PATH.name,
        },
    }
    original_scorecards = baseline_renderer.SCORECARDS
    original_methods = baseline_renderer.METHODS_PATH
    original_highlights = baseline_renderer.HIGHLIGHT_VARIABLES
    try:
        baseline_renderer.SCORECARDS = final_scorecards
        baseline_renderer.METHODS_PATH = METHODS_PATH
        baseline_renderer.HIGHLIGHT_VARIABLES = experiment.Q_TARGETS
        page = baseline_renderer.render_scorecard(
            "cutover", scores, comparisons
        )
    finally:
        baseline_renderer.SCORECARDS = original_scorecards
        baseline_renderer.METHODS_PATH = original_methods
        baseline_renderer.HIGHLIGHT_VARIABLES = original_highlights

    page = _final_scorecard_labels(page)
    FINAL_SCORECARD_PATH.write_text(page, encoding="utf-8")
    return {
        "scorecard": str(FINAL_SCORECARD_PATH),
        "scorecard_bytes": FINAL_SCORECARD_PATH.stat().st_size,
        "scores": str(FINAL_SCORES_PATH),
        "comparisons": str(FINAL_COMPARISONS_PATH),
        "adapter": SELECTED_ADAPTER,
        "paired_cases": 16,
        "bootstrap_draws": bootstrap_draws,
    }


def build_report(adapter_names: list[str], *, best: str) -> dict[str, Any]:
    if best not in adapter_names:
        raise ValueError("The selected adapter must be included in the report")
    summaries, comparisons = _load_results(adapter_names)
    _plot_comparison(adapter_names, comparisons)
    _write_consolidated_results(
        adapter_names, summaries, comparisons, best=best
    )
    methods = html.escape(METHODS_PATH.read_text(encoding="utf-8"))
    best_label = ADAPTER_LABELS.get(best, best)
    summary_table = _summary_table(adapter_names, summaries, best)
    q_tables = _q_tables(adapter_names, comparisons)
    guardrails = _guardrail_table(adapter_names, comparisons)
    full_heatmap = _full_heatmap(comparisons[best])
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIFS v1.1 frozen 50r1 adapter comparison</title>
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
  h1, h2, h3 {{ font: inherit; font-weight: 400; }}
  h1 {{ margin: 0 0 10px; }}
  h2 {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 34px auto 6px;
    padding-top: 20px;
    border-top: 1px solid var(--ink);
  }}
  h3 {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 22px auto 6px;
  }}
  .subtitle, .section-subtitle {{
    font-size: inherit;
    line-height: 1.5;
    color: var(--ink);
  }}
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
  .evidence-order strong {{
    display: block;
    font-weight: 400;
    margin-bottom: 3px;
  }}
  .selection {{
    width: calc(100% - 32px);
    max-width: 1120px;
    border-left: 3px solid var(--blue);
    padding: 6px 14px;
    margin: 20px auto;
  }}
  .selection strong {{ font-weight: 400; }}
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
    margin: 8px auto 12px;
    letter-spacing: .02em;
  }}
  .legend {{
    display: grid;
    grid-template-columns:
      minmax(190px, 1fr) minmax(260px, 420px) minmax(190px, 1fr);
    align-items: center;
    gap: 12px;
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 14px auto 7px;
  }}
  .legend-label strong {{
    display: block;
    color: var(--ink);
    font-weight: 400;
  }}
  .legend-label-left {{ text-align: right; }}
  .gradient {{
    width: 100%;
    height: 14px;
    border: 1px solid var(--ink);
    background: linear-gradient(90deg, #053061, #f7f6f6, #67001f);
  }}
  .table-scroll {{
    width: calc(100% - 32px);
    max-width: 1120px;
    overflow-x: auto;
    margin: 14px auto 18px;
    border: 1px solid var(--ink);
    background: var(--bg);
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    white-space: nowrap;
    font-size: 12px;
    background: var(--bg);
  }}
  th, td {{
    border: 1px solid var(--line);
    padding: 5px 7px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  th {{
    background: var(--ink);
    color: var(--bg);
    font-weight: 400;
  }}
  th:first-child {{ text-align: left; }}
  .summary-table {{ min-width: 860px; }}
  .heat-table {{ min-width: 790px; table-layout: fixed; }}
  .adapter-table {{ min-width: 1040px; }}
  .adapter-table th:first-child {{ width: 300px; }}
  .field-table th:first-child {{ width: 92px; }}
  .heat {{ text-align: center; min-width: 54px; }}
  .significant {{ box-shadow: inset 0 0 0 2px var(--ink); }}
  .group th {{
    background: var(--soft);
    color: var(--ink);
    text-align: left;
  }}
  .full {{ max-height: 760px; overflow: auto; }}
  .full thead th {{ position: sticky; top: 0; z-index: 2; }}
  .full tr th:first-child {{
    position: sticky;
    left: 0;
    z-index: 1;
  }}
  .notes {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 18px auto 0;
  }}
  details {{ border: 1px solid var(--ink); padding: 10px 12px; }}
  summary {{ cursor: pointer; }}
  pre {{
    white-space: pre-wrap;
    max-height: 520px;
    overflow: auto;
    font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .footer {{
    width: calc(100% - 32px);
    max-width: 1120px;
    margin: 34px auto 0;
    padding-top: 18px;
    border-top: 1px solid var(--ink);
    color: var(--muted);
  }}
  code {{ background: #EFE8DF; padding: 1px 4px; }}
  @media (max-width: 760px) {{
    .evidence-order {{ grid-template-columns: 1fr; }}
    .legend {{ grid-template-columns: 1fr; }}
    .legend-label, .legend-label-left {{ text-align: left; }}
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
  <div class="eyebrow">AIFS v1.1 / frozen input adaptation</div>
  <h1>Five-candidate 50r1 adapter comparison</h1>
  <div class="subtitle">
    Sixteen 17–20 May 2026 initializations. Every adapter forecast is paired
    with the unmodified forecast from the identical 50r1 state and verified
    against the same ERA5T field. Negative RMSE change is better.
  </div>
  <div class="evidence-order">
    <div>
      <strong>01 / fixed forecast model</strong>
      <span>The same AIFS Single v1.1 checkpoint is used throughout.</span>
    </div>
    <div>
      <strong>02 / frozen input adapters</strong>
      <span>Only the two 50r1 initialization states are transformed.</span>
    </div>
  </div>
</div>
<div class="selection"><strong>Selected adapter: {html.escape(best_label)}.</strong>
Selection considers upper-atmosphere humidity improvement, operational
guardrails, physical clipping, and breadth of significant harms—not the
largest improvement in any single score cell.</div>
<nav>
  <a href="#summary">Candidate summary</a>
  <a href="#humidity">Humidity</a>
  <a href="#guardrails">Guardrails</a>
  <a href="#selected">Selected adapter</a>
  <a href="#methods">Methods</a>
  <a href="{html.escape(FINAL_SCORECARD_PATH.name)}">Final paired scorecard</a>
</nav>
<div class="lead-key">lead hours: 24 48 72 96 120 144 168 192 216 240 · 96 instantaneous fields · global + 5 regional/land-ocean domains</div>

<section id="summary">
<h2>Candidate summary</h2>
<div class="section-subtitle">All values except counts are paired RMSE
changes in percent. “Early” is D1–D5; guardrails are Z500, T850, 2t, 10u,
10v and MSLP over global and northern-extratropical domains.</div>
{summary_table}
</section>

<section id="humidity">
<h2>Upper-atmosphere humidity</h2>
<div class="section-subtitle">Black inset borders indicate paired 95%
intervals excluding zero.</div>
{q_tables}
</section>

<section id="guardrails">
<h2>Operational guardrails</h2>
<div class="section-subtitle">Mean paired RMSE change across D1–D5 in global
and northern-extratropical domains.</div>
{guardrails}
</section>

<section id="selected">
<h2>Selected adapter · all global instantaneous fields</h2>
<div class="legend" aria-label="RMSE colour scale">
  <span class="legend-label legend-label-left">
    corrected lower RMSE (better)<strong>−10 %</strong>
  </span>
  <div class="gradient"></div>
  <span class="legend-label">
    corrected higher RMSE (worse)<strong>+10 %</strong>
  </span>
</div>
<div class="section-subtitle"><code>100 × (RMSEadapter / RMSEbaseline − 1)</code>;
colours saturate at ±10%. Hover for the paired 95% interval.</div>
{full_heatmap}
</section>

<section id="methods">
<h2>Methods and limitations</h2>
<div class="notes"><details open><summary>Frozen-adapter protocol</summary>
<pre>{methods}</pre></details></div>
</section>
<div class="footer">Machine-readable results:
<a href="../data/processed/aifs_v11_frozen_adapter_summary.csv">candidate summary</a>
· <a href="../data/processed/aifs_v11_frozen_adapter_paired_comparisons.parquet">paired comparisons</a>
· <a href="{html.escape(FINAL_SCORECARD_PATH.name)}">selected-adapter scorecard</a>.
</div>
</main></body></html>"""
    REPORT_PATH.write_text(page, encoding="utf-8")
    return {
        "report": str(REPORT_PATH),
        "report_bytes": REPORT_PATH.stat().st_size,
        "figure": str(FIGURE_PATH),
        "summary": str(SUMMARY_PATH),
        "comparisons": str(COMPARISON_PATH),
        "best": best,
        "adapters": adapter_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=list(ADAPTER_LABELS),
        choices=sorted(experiment.ADAPTERS),
    )
    parser.add_argument(
        "--best", required=True, choices=sorted(experiment.ADAPTERS)
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "candidate_comparison": build_report(
                    args.adapters, best=args.best
                ),
                "final_scorecard": build_final_scorecard(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
