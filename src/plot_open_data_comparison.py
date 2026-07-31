"""Create hrrs.ai-themed design and result figures for the cohort comparison."""

from __future__ import annotations

import argparse
import base64
from html import escape
from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_CACHE = PROJECT_ROOT / "data" / "cache" / "matplotlib"
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(PLOT_CACHE)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, SymLogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

from analyze_open_data_inputs import COMPARISONS_CSV, MAPS
from open_data_inputs import (
    COHORTS,
    IFS_50R1_CUTOVER,
    PLAN_SUMMARY,
    PRESSURE_LEVELS,
    ROOT,
)
from variable_order import (
    DISPLAY_GROUPS,
    DISPLAY_LABEL,
    DISPLAY_VARIABLE_ORDER,
    display_order_frame,
)


FIGURES = ROOT / "outputs" / "figures"
REPORT = ROOT / "outputs" / "open_data_input_comparison_report.html"
DISPLAY_ORDER_CSV = (
    ROOT / "data" / "processed" / "open_data_variable_display_order.csv"
)

BACKGROUND = "#F8F2EA"
INK = "#1C1C1C"
BLUE = "#0000FF"
RED = "#D94A3A"
MUTED = INK
GRID = "#D8D0C6"
PALE_BLUE = "#D8D8FF"
PLOT_FONT_SIZE = 9.0
COMPACT_PLOT_FONT_SIZE = 14.0
PLOT_SIZE = (16.0, 7.2)
PLOT_MARGINS = {
    "left": 0.17,
    "right": 0.94,
    "bottom": 0.18,
    "top": 0.78,
}

MATRIX_ROWS = (
    ("global", "all", "all", "global · all UTC"),
    ("tropics", "all", "all", "tropics · all UTC"),
    ("extratropics", "all", "all", "extratropics · all UTC"),
    ("global", "land", "all", "global land · all UTC"),
    ("global", "ocean", "all", "global ocean · all UTC"),
    ("global", "all", "00", "global · 00 UTC"),
    ("global", "all", "06", "global · 06 UTC"),
    ("global", "all", "12", "global · 12 UTC"),
    ("global", "all", "18", "global · 18 UTC"),
)

COMPLETE_METRICS = {
    "mean": {
        "column": "normalized_mean_shift",
        "transform": None,
        "title": "normalized mean shift",
        "colorbar": "50r1 − 49r1 / AIFS v1.1 normalization scale",
        "signed": True,
    },
    "variance": {
        "column": "variance_ratio_50r1_over_49r1",
        "transform": np.log2,
        "title": "variance change",
        "colorbar": "log₂(variance 50r1 / variance 49r1)",
        "signed": True,
    },
    "wasserstein": {
        "column": "normalized_wasserstein_distance",
        "transform": None,
        "title": "distribution distance",
        "colorbar": "Wasserstein distance / AIFS v1.1 normalization scale",
        "signed": False,
    },
}


def set_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "monospace",
            "font.size": PLOT_FONT_SIZE,
            "font.monospace": [
                "SFMono-Regular",
                "Menlo",
                "Monaco",
                "Consolas",
                "Liberation Mono",
                "DejaVu Sans Mono",
            ],
            "axes.titlelocation": "left",
            "axes.titleweight": "normal",
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "axes.labelpad": 10,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "figure.titlesize": PLOT_FONT_SIZE,
            "axes.grid": False,
        }
    )


def _finish_figure(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{name}.png"
    svg = FIGURES / f"{name}.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def plot_cohort_design() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=PLOT_SIZE)
    fig.subplots_adjust(**PLOT_MARGINS, wspace=0.65)
    panel_definitions = [
        (
            axes[0],
            "same_season",
            "same season / calendar-aligned",
            {
                "49r1": "49r1 · 13–20 May 2025",
                "50r1": "50r1 · 13–20 May 2026",
            },
            (12.5, 20.5),
        ),
        (
            axes[1],
            "cutover",
            "cutover / adjacent windows",
            {
                "49r1": "49r1 · 4–11 May 2026",
                "50r1": "50r1 · 13–20 May 2026",
            },
            (3.5, 20.5),
        ),
    ]
    for ax, comparison, title, labels, limits in panel_definitions:
        for y, cycle in ((1, "49r1"), (0, "50r1")):
            cohort = next(
                item
                for item in COHORTS
                if item.comparison == comparison and item.cohort == cycle
            )
            start = cohort.start.day + cohort.start.hour / 24
            end = cohort.end.day + cohort.end.hour / 24
            color = BLUE if cycle == "50r1" else INK
            ax.plot(
                [start, end],
                [y, y],
                color=color,
                linewidth=8,
                solid_capstyle="butt",
            )
            ax.scatter(
                [start, end],
                [y, y],
                s=34,
                facecolor=BACKGROUND,
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )
            ax.text(
                start,
                y + 0.18,
                f"{len(cohort.pair_times)} t₀",
                color=INK,
                fontsize=COMPACT_PLOT_FONT_SIZE,
            )
        if comparison == "cutover":
            cutover_day = IFS_50R1_CUTOVER.day + IFS_50R1_CUTOVER.hour / 24
            ax.axvline(
                cutover_day,
                color=RED,
                linestyle=(0, (3, 3)),
                linewidth=1.4,
            )
            ax.text(
                cutover_day,
                1.53,
                "50r1 · 12 May 06 UTC",
                color=INK,
                fontsize=COMPACT_PLOT_FONT_SIZE,
                ha="center",
            )
        ax.set_xlim(*limits)
        ax.set_ylim(-0.48, 1.68)
        ax.set_yticks([1, 0], [labels["49r1"], labels["50r1"]])
        ax.set_xticks(
            np.arange(np.ceil(limits[0]), np.floor(limits[1]) + 1, 2)
        )
        ax.set_xlabel("day of May", fontsize=COMPACT_PLOT_FONT_SIZE)
        ax.set_title(title, pad=10, fontsize=COMPACT_PLOT_FONT_SIZE)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(
            axis="y",
            length=0,
            pad=8,
            labelsize=COMPACT_PLOT_FONT_SIZE,
        )
        ax.tick_params(axis="x", pad=8, labelsize=COMPACT_PLOT_FONT_SIZE)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    fig.suptitle(
        "Experiment design / two complementary unpaired comparisons",
        x=0.02,
        y=0.95,
        ha="left",
        fontsize=COMPACT_PLOT_FONT_SIZE,
    )
    fig.text(
        0.02,
        0.90,
        "Each t₀ uses its immediately preceding six-hour state; 12 May is not a t₀ sample.",
        color=MUTED,
        fontsize=COMPACT_PLOT_FONT_SIZE,
    )
    png, _ = _finish_figure(fig, "open_data_cohort_design")
    return png


def plot_field_inventory() -> Path:
    categories = pd.DataFrame(
        {
            "category": [
                "pressure-level dynamic",
                "surface dynamic",
                "soil dynamic",
                "static / per cycle",
            ],
            "messages": [6 * 13, 8, 2 * 2, 4],
            "retention": ["per state", "per state", "per state", "once / cycle"],
        }
    )
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    fig.subplots_adjust(**PLOT_MARGINS)
    y = np.arange(len(categories))
    colors = [BLUE, INK, RED, MUTED]
    ax.barh(y, categories["messages"], color=colors, height=0.58)
    for index, row in categories.iterrows():
        ax.text(
            row["messages"] + 1.2,
            index,
            f"{row['messages']} messages · {row['retention']}",
            va="center",
            fontsize=COMPACT_PLOT_FONT_SIZE,
        )
    ax.set_yticks(y, categories["category"])
    ax.invert_yaxis()
    ax.set_xlim(0, 96)
    ax.set_xlabel("GRIB messages", fontsize=COMPACT_PLOT_FONT_SIZE)
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(
        axis="y",
        length=0,
        pad=8,
        labelsize=COMPACT_PLOT_FONT_SIZE,
    )
    ax.tick_params(axis="x", pad=8, labelsize=COMPACT_PLOT_FONT_SIZE)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.suptitle(
        "AIFS v1.1 input inventory / 90 dynamic + 4 static",
        x=0.02,
        y=0.95,
        ha="left",
        fontsize=COMPACT_PLOT_FONT_SIZE,
    )
    fig.text(
        0.02,
        0.90,
        "Dynamic fields are retained for every state; static fields are retained once per cycle.",
        fontsize=COMPACT_PLOT_FONT_SIZE,
    )
    png, _ = _finish_figure(fig, "open_data_field_inventory")
    return png


def _complete_metric_arrays(
    comparisons: pd.DataFrame,
    *,
    kind: str,
    metric: str,
) -> list[np.ndarray]:
    specification = COMPLETE_METRICS[metric]
    arrays: list[np.ndarray] = []
    for comparison in ("same_season", "cutover"):
        values = np.full(
            (len(MATRIX_ROWS), len(DISPLAY_VARIABLE_ORDER)),
            np.nan,
            dtype=np.float64,
        )
        for row, (region, surface, utc, _) in enumerate(MATRIX_ROWS):
            selected = comparisons[
                (comparisons["comparison"] == comparison)
                & (comparisons["kind"] == kind)
                & (comparisons["region"] == region)
                & (comparisons["surface"] == surface)
                & (comparisons["utc"] == utc)
            ].set_index("variable")[specification["column"]]
            values[row] = selected.reindex(DISPLAY_VARIABLE_ORDER).to_numpy()
        transform = specification["transform"]
        if transform is not None:
            values = transform(values)
        arrays.append(values)
    return arrays


def _matrix_norm(
    arrays: list[np.ndarray],
    *,
    signed: bool,
) -> tuple[mpl.colors.Normalize, str]:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if signed:
        vmax = max(float(np.max(np.abs(finite))), 1e-12)
        nonzero = np.abs(finite[np.abs(finite) > 0])
        if len(nonzero):
            linear_threshold = max(
                float(np.percentile(nonzero, 20)),
                vmax / 300,
                1e-12,
            )
        else:
            linear_threshold = vmax / 100
        return (
            SymLogNorm(
                linthresh=linear_threshold,
                linscale=0.7,
                vmin=-vmax,
                vmax=vmax,
                base=10,
            ),
            "coolwarm",
        )
    vmax = max(float(np.max(finite)), 1e-12)
    return PowerNorm(gamma=0.45, vmin=0, vmax=vmax), "Blues"


def _draw_variable_groups(ax: plt.Axes, *, show_labels: bool) -> None:
    position = 0
    for group in DISPLAY_GROUPS:
        start = position
        end = position + len(group.variables)
        if start:
            ax.axvline(start - 0.5, color=INK, linewidth=0.55)
        if show_labels:
            ax.text(
                (start + end - 1) / 2,
                1.15,
                group.label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=PLOT_FONT_SIZE,
                color=INK,
            )
        position = end
    ax.axhline(4.5, color=INK, linewidth=0.7)


def plot_complete_metric(
    comparisons: pd.DataFrame,
    *,
    kind: str,
    metric: str,
) -> Path:
    specification = COMPLETE_METRICS[metric]
    arrays = _complete_metric_arrays(comparisons, kind=kind, metric=metric)
    norm, cmap = _matrix_norm(arrays, signed=bool(specification["signed"]))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=PLOT_SIZE,
        sharex=True,
    )
    fig.subplots_adjust(**PLOT_MARGINS, hspace=0.40)
    row_labels = [row[3] for row in MATRIX_ROWS]
    for index, (ax, comparison, array) in enumerate(
        zip(axes, ("same season", "cutover"), arrays, strict=True)
    ):
        image = ax.imshow(
            array,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_title(comparison, pad=10)
        ax.tick_params(axis="y", length=0, pad=8)
        _draw_variable_groups(ax, show_labels=index == 0)
    axes[-1].set_xticks(
        np.arange(len(DISPLAY_VARIABLE_ORDER)),
        [DISPLAY_LABEL[variable] for variable in DISPLAY_VARIABLE_ORDER],
        rotation=90,
        fontsize=PLOT_FONT_SIZE,
    )
    axes[-1].tick_params(axis="x", length=0, pad=8)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label(str(specification["colorbar"]))
    colorbar.ax.tick_params(labelsize=PLOT_FONT_SIZE, pad=8)
    kind_title = "direct states" if kind == "state" else "six-hour tendencies"
    fig.suptitle(
        f"Complete input scorecard / {kind_title} / {specification['title']}",
        x=0.02,
        y=0.95,
        ha="left",
        fontsize=PLOT_FONT_SIZE,
    )
    fig.text(
        0.02,
        0.90,
        (
            "All 90 dynamic fields · scorecard order · pressure levels run "
            "50 → 1000 hPa · symmetric-log colour for signed metrics"
            if specification["signed"]
            else (
                "All 90 dynamic fields · scorecard order · pressure levels run "
                "50 → 1000 hPa · square-root-like colour scaling"
            )
        ),
        color=MUTED,
        fontsize=PLOT_FONT_SIZE,
    )
    filename = f"open_data_all_{kind}_{metric}"
    png, _ = _finish_figure(fig, filename)
    return png


def consistent_ranking(comparisons: pd.DataFrame, kind: str = "state") -> pd.DataFrame:
    selected = comparisons[
        (comparisons["kind"] == kind)
        & (comparisons["region"] == "global")
        & (comparisons["surface"] == "all")
        & (comparisons["utc"] == "all")
    ]
    pivot = selected.pivot(
        index="variable",
        columns="comparison",
        values="normalized_mean_shift",
    ).dropna()
    pivot["consistent_direction"] = (
        np.sign(pivot["same_season"]) == np.sign(pivot["cutover"])
    )
    pivot["consistent_magnitude"] = np.minimum(
        np.abs(pivot["same_season"]), np.abs(pivot["cutover"])
    )
    pivot["max_magnitude"] = np.maximum(
        np.abs(pivot["same_season"]), np.abs(pivot["cutover"])
    )
    return pivot.sort_values(
        ["consistent_direction", "consistent_magnitude", "max_magnitude"],
        ascending=False,
    ).reset_index()


def plot_consistent_shifts(
    comparisons: pd.DataFrame, *, kind: str = "state"
) -> Path:
    ranking = consistent_ranking(comparisons, kind=kind).head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 7.4), constrained_layout=True)
    y = np.arange(len(ranking))
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.scatter(
        ranking["same_season"],
        y - 0.13,
        s=34,
        color=INK,
        label="same season",
    )
    ax.scatter(
        ranking["cutover"],
        y + 0.13,
        s=34,
        color=BLUE,
        label="cutover",
    )
    for index, row in ranking.reset_index(drop=True).iterrows():
        color = GRID if row["consistent_direction"] else RED
        ax.plot(
            [row["same_season"], row["cutover"]],
            [index - 0.13, index + 0.13],
            color=color,
            linewidth=0.8,
            zorder=0,
        )
    ax.set_yticks(y, ranking["variable"])
    ax.set_xlabel("mean shift / AIFS v1.1 normalization scale")
    title_kind = "state" if kind == "state" else "six-hour tendency"
    ax.set_title(
        f"Most repeatable {title_kind} shifts / global · all surface · all UTC",
        pad=16,
    )
    ax.text(
        0,
        1.01,
        "Ranked by the smaller absolute shift across both comparisons.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
        va="bottom",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    png, _ = _finish_figure(
        fig,
        (
            "open_data_consistent_mean_shifts"
            if kind == "state"
            else "open_data_consistent_tendency_shifts"
        ),
    )
    return png


def plot_variance_ratios(comparisons: pd.DataFrame) -> Path:
    selected = comparisons[
        (comparisons["kind"] == "state")
        & (comparisons["region"] == "global")
        & (comparisons["surface"] == "all")
        & (comparisons["utc"] == "all")
    ].copy()
    selected["log2_variance_ratio"] = np.log2(
        selected["variance_ratio_50r1_over_49r1"]
    )
    pivot = selected.pivot(
        index="variable",
        columns="comparison",
        values="log2_variance_ratio",
    ).dropna()
    pivot["consistent"] = np.sign(pivot["same_season"]) == np.sign(pivot["cutover"])
    pivot["rank"] = np.minimum(
        np.abs(pivot["same_season"]), np.abs(pivot["cutover"])
    )
    ranking = (
        pivot.sort_values(["consistent", "rank"], ascending=False)
        .head(18)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(10.5, 7.4), constrained_layout=True)
    y = np.arange(len(ranking))
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.scatter(ranking["same_season"], y - 0.13, s=34, color=INK, label="same season")
    ax.scatter(ranking["cutover"], y + 0.13, s=34, color=BLUE, label="cutover")
    for index, row in enumerate(ranking.itertuples()):
        ax.plot(
            [row.same_season, row.cutover],
            [index - 0.13, index + 0.13],
            color=GRID if row.consistent else RED,
            linewidth=0.8,
            zorder=0,
        )
    ax.set_yticks(y, ranking.index)
    ax.set_xlabel("log₂(variance 50r1 / variance 49r1)")
    ax.set_title(
        "Most repeatable variance changes / global · all surface · all UTC",
        pad=16,
    )
    ax.text(
        0,
        1.01,
        "Positive values indicate more variance in the 50r1 cohort.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
        va="bottom",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    png, _ = _finish_figure(fig, "open_data_consistent_variance_ratios")
    return png


def plot_pressure_heatmap(comparisons: pd.DataFrame) -> Path:
    selected = comparisons[
        (comparisons["kind"] == "state")
        & (comparisons["region"] == "global")
        & (comparisons["surface"] == "all")
        & (comparisons["utc"] == "all")
        & (comparisons["level_type"] == "pl")
    ].copy()
    selected["base_variable"] = selected["variable"].str.split("_").str[0]
    variables = ["z", "t", "u", "v", "w", "q"]
    levels = list(reversed(PRESSURE_LEVELS))
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 7.8), sharey=True, constrained_layout=True
    )
    arrays = []
    for comparison in ("same_season", "cutover"):
        current = selected[selected["comparison"] == comparison]
        table = current.pivot(
            index="level",
            columns="base_variable",
            values="normalized_mean_shift",
        ).reindex(index=levels, columns=variables)
        arrays.append(table.to_numpy())
    vmax = np.nanpercentile(np.abs(np.concatenate([array.ravel() for array in arrays])), 98)
    vmax = max(float(vmax), 1e-12)
    for ax, comparison, array in zip(
        axes, ("same season", "cutover"), arrays, strict=True
    ):
        image = ax.imshow(
            array,
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        )
        ax.set_xticks(np.arange(len(variables)), variables)
        ax.set_yticks(np.arange(len(levels)), levels)
        ax.set_title(comparison)
        ax.set_xlabel("variable")
        ax.tick_params(length=0)
    axes[0].set_ylabel("pressure level / hPa")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025)
    colorbar.set_label("mean shift / normalization scale")
    fig.suptitle("Vertical structure of the input-distribution shift", x=0.06, ha="left")
    png, _ = _finish_figure(fig, "open_data_pressure_level_heatmap")
    return png


def _annotated_shift_heatmaps(
    arrays: list[np.ndarray],
    row_labels: list[str],
    column_labels: list[str],
    *,
    title: str,
    filename: str,
) -> Path:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    vmax = max(float(np.max(np.abs(finite))), 1e-12)
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 6.7), sharey=True, constrained_layout=True
    )
    for ax, comparison, array in zip(
        axes, ("same season", "cutover"), arrays, strict=True
    ):
        image = ax.imshow(
            array,
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        )
        ax.set_xticks(np.arange(len(column_labels)), column_labels)
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_title(comparison)
        ax.tick_params(length=0)
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                if np.isfinite(value):
                    text_color = BACKGROUND if abs(value) > 0.55 * vmax else INK
                    ax.text(
                        column,
                        row,
                        f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=text_color,
                    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025)
    colorbar.set_label("mean shift / normalization scale")
    fig.suptitle(title, x=0.06, ha="left")
    png, _ = _finish_figure(fig, filename)
    return png


def plot_regional_surface_shifts(comparisons: pd.DataFrame) -> Path:
    variables = consistent_ranking(comparisons).head(10)["variable"].tolist()
    strata = [
        ("global", "land", "global · land"),
        ("global", "ocean", "global · ocean"),
        ("tropics", "all", "tropics · all"),
        ("extratropics", "all", "extratropics · all"),
    ]
    arrays: list[np.ndarray] = []
    for comparison in ("same_season", "cutover"):
        values = np.full((len(variables), len(strata)), np.nan)
        for column, (region, surface, _) in enumerate(strata):
            selected = comparisons[
                (comparisons["comparison"] == comparison)
                & (comparisons["kind"] == "state")
                & (comparisons["region"] == region)
                & (comparisons["surface"] == surface)
                & (comparisons["utc"] == "all")
            ].set_index("variable")["normalized_mean_shift"]
            values[:, column] = selected.reindex(variables).to_numpy()
        arrays.append(values)
    return _annotated_shift_heatmaps(
        arrays,
        variables,
        [label for _, _, label in strata],
        title="Where the repeatable state shifts occur",
        filename="open_data_regional_surface_shifts",
    )


def plot_utc_shifts(comparisons: pd.DataFrame) -> Path:
    variables = consistent_ranking(comparisons).head(10)["variable"].tolist()
    utc_runs = ["00", "06", "12", "18"]
    arrays: list[np.ndarray] = []
    for comparison in ("same_season", "cutover"):
        selected = comparisons[
            (comparisons["comparison"] == comparison)
            & (comparisons["kind"] == "state")
            & (comparisons["region"] == "global")
            & (comparisons["surface"] == "all")
            & (comparisons["utc"].isin(utc_runs))
        ]
        table = selected.pivot(
            index="variable",
            columns="utc",
            values="normalized_mean_shift",
        ).reindex(index=variables, columns=utc_runs)
        arrays.append(table.to_numpy())
    return _annotated_shift_heatmaps(
        arrays,
        variables,
        [f"{utc} UTC" for utc in utc_runs],
        title="UTC-run sensitivity of the repeatable state shifts",
        filename="open_data_utc_shifts",
    )


def _binned_map(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    resolution: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_edges = np.arange(-90, 90 + resolution, resolution)
    lon_edges = np.arange(0, 360 + resolution, resolution)
    count, _, _ = np.histogram2d(latitude, longitude, bins=(lat_edges, lon_edges))
    total, _, _ = np.histogram2d(
        latitude, longitude, bins=(lat_edges, lon_edges), weights=values
    )
    grid = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    )
    return grid, lat_edges, lon_edges


def _best_mapped_field(comparisons: pd.DataFrame) -> str | None:
    ranking = consistent_ranking(comparisons)
    available = {
        path.stem.removeprefix("same_season_")
        for path in MAPS.glob("same_season_*.npz")
    }
    for variable in ranking["variable"]:
        if variable in available:
            return str(variable)
    return None


def plot_change_maps(comparisons: pd.DataFrame) -> Path | None:
    variable = _best_mapped_field(comparisons)
    if variable is None:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.3), constrained_layout=True)
    map_data = {}
    for comparison in ("same_season", "cutover"):
        with np.load(MAPS / f"{comparison}_{variable}.npz") as data:
            map_data[comparison] = {key: np.asarray(data[key]) for key in data.files}
    mean_grids = []
    ratio_grids = []
    for comparison in ("same_season", "cutover"):
        data = map_data[comparison]
        mean_grid, lat_edges, lon_edges = _binned_map(
            data["latitude"],
            data["longitude"],
            data["normalized_mean_shift"],
        )
        ratio_grid, _, _ = _binned_map(
            data["latitude"],
            data["longitude"],
            np.log2(data["variance_ratio_50r1_over_49r1"]),
        )
        mean_grids.append(mean_grid)
        ratio_grids.append(ratio_grid)
    mean_vmax = max(
        float(np.nanpercentile(np.abs(np.concatenate([x.ravel() for x in mean_grids])), 98)),
        1e-12,
    )
    ratio_vmax = max(
        float(np.nanpercentile(np.abs(np.concatenate([x.ravel() for x in ratio_grids])), 98)),
        1e-12,
    )
    for column, comparison in enumerate(("same_season", "cutover")):
        first = axes[0, column].pcolormesh(
            lon_edges,
            lat_edges,
            mean_grids[column],
            cmap="coolwarm",
            vmin=-mean_vmax,
            vmax=mean_vmax,
            shading="flat",
        )
        second = axes[1, column].pcolormesh(
            lon_edges,
            lat_edges,
            ratio_grids[column],
            cmap="coolwarm",
            vmin=-ratio_vmax,
            vmax=ratio_vmax,
            shading="flat",
        )
        axes[0, column].set_title(comparison.replace("_", " "))
        axes[1, column].set_xlabel("longitude")
        for row in (0, 1):
            axes[row, column].set_xlim(0, 360)
            axes[row, column].set_ylim(-90, 90)
            axes[row, column].set_facecolor(BACKGROUND)
    axes[0, 0].set_ylabel("latitude\nmean shift")
    axes[1, 0].set_ylabel("latitude\nvariance ratio")
    fig.colorbar(first, ax=axes[0], fraction=0.025, label="normalized mean shift")
    fig.colorbar(
        second,
        ax=axes[1],
        fraction=0.025,
        label="log₂(variance 50r1 / 49r1)",
    )
    fig.suptitle(f"Spatial structure / {variable}", x=0.06, ha="left")
    png, _ = _finish_figure(fig, f"open_data_maps_{variable}")
    return png


def _embedded_image_source(path: Path) -> str:
    mime_type = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
    }.get(path.suffix.lower(), "application/octet-stream")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_report(figures: list[Path], comparisons: pd.DataFrame | None) -> None:
    plan_summary = (
        json.loads(PLAN_SUMMARY.read_text(encoding="utf-8"))
        if PLAN_SUMMARY.exists()
        else {}
    )
    estimate = plan_summary.get("download_estimate", {})
    captions = {
        "open_data_cohort_design": "Both unpaired designs contain 32 t₀ samples per cohort plus the immediately preceding six-hour state.",
        "open_data_field_inventory": "All 90 time-varying inputs and all four static inputs are retained.",
        "open_data_all_state_mean": "Area-weighted mean-state difference, shown for every input field, region/surface stratum and UTC run.",
        "open_data_all_state_variance": "State variance ratio for the same complete field and row inventory.",
        "open_data_all_state_wasserstein": "One-dimensional state-distribution distance; values are non-negative.",
        "open_data_all_tendency_mean": "Mean change in x(t₀)−x(t₀−6 h), retaining all fields and strata.",
        "open_data_all_tendency_variance": "Variance ratio of the six-hour input-pair tendencies.",
        "open_data_all_tendency_wasserstein": "Distribution distance between the six-hour tendencies.",
    }
    section_starts = {
        "open_data_cohort_design": "Experiment and inventory",
        "open_data_all_state_mean": "Complete direct-state scorecards",
        "open_data_all_tendency_mean": "Complete six-hour-tendency scorecards",
    }
    cards = []
    for figure in figures:
        if figure.stem in section_starts:
            cards.append(f"<h2>{escape(section_starts[figure.stem])}</h2>")
        caption = captions.get(figure.stem, "Complete comparison matrix.")
        image = figure.with_suffix(".svg")
        if not image.exists():
            image = figure
        is_matrix = figure.stem.startswith("open_data_all_")
        css_class = ' class="matrix"' if is_matrix else ""
        cards.append(
            f"<figure{css_class}>"
            f'<img src="{_embedded_image_source(image)}" '
            f'alt="{escape(figure.stem)}">'
            f"<figcaption>{escape(caption)}</figcaption></figure>"
        )
    if comparisons is None:
        results = """
        <section class="status">
          <span class="eyebrow">status / pipeline ready</span>
          <p>Distribution-result panels appear after the full indexed download
          and analysis complete. The design panels below contain no synthetic
          meteorological results.</p>
        </section>
        """
    else:
        results = """
        <section class="status complete">
          <span class="eyebrow">status / complete</span>
          <p>No field ranking or top-variable selection is applied. Every
          scorecard below contains all 90 dynamic inputs in one fixed order.</p>
        </section>
        <section class="guide">
          <div><strong>x-axis</strong><p>The inferred-scorecard convention:
          parameter blocks, with pressure levels ordered 50 → 1000 hPa.
          Surface and soil fields occupy fixed positions between or after
          their related blocks.</p></div>
          <div><strong>y-axis</strong><p>Global, tropical, extratropical,
          land and ocean results, followed by the four global UTC runs. The
          horizontal rule separates spatial strata from UTC strata.</p></div>
          <div><strong>panels</strong><p>Same-season and cutover comparisons
          are always shown in that order. Signed panels use a symmetric-log
          colour scale so small fields remain visible beside large shifts.</p></div>
        </section>
        <p class="downloads">Machine-readable:
        <a href="../data/processed/open_data_cycle_comparisons.csv">all comparisons</a>
        · <a href="../data/processed/open_data_cohort_statistics.csv">all cohort statistics and quantiles</a>
        · <a href="../data/processed/open_data_variable_display_order.csv">canonical field order</a>
        </p>
        """
    selected_gib = estimate.get("estimated_selected_gib", 6.03)
    transferred_gib = estimate.get("estimated_transferred_gib", 6.54)
    unique_states = int(plan_summary.get("unique_state_count", 99))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIFS input shift / 49r1 → 50r1</title>
<style>
:root {{ --bg:{BACKGROUND}; --ink:{INK}; --blue:{BLUE}; --red:{RED}; --muted:{MUTED}; --grid:{GRID}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace; font-size:.95rem; line-height:1.5; }}
header {{ width:calc(100% - 32px); max-width:1120px; margin:auto; }}
header {{ padding:36px 0 20px; border-bottom:1px solid var(--ink); }}
h1 {{ font-size:inherit; font-weight:400; margin:.2rem 0; letter-spacing:0; }}
h2 {{ font-size:inherit; font-weight:400; margin:3rem 0 1rem; }}
.eyebrow {{ color:var(--ink); font-size:inherit; text-transform:uppercase; letter-spacing:.08em; }}
.lede {{ max-width:72ch; color:var(--ink); }}
.meta {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--ink); border:1px solid var(--ink); margin:28px 0 0; }}
.meta div {{ background:var(--bg); padding:14px; }}
.meta strong {{ display:block; font-weight:400; font-size:inherit; color:var(--ink); }}
main {{ width:100%; padding:32px 0 64px; }}
.status {{ border-left:3px solid var(--red); padding:4px 18px; margin-bottom:34px; }}
.status.complete {{ border-color:var(--blue); }}
.status p {{ max-width:78ch; margin:.35rem 0; }}
figure {{ margin:0 0 24px; border:1px solid var(--grid); padding:12px; }}
figure img {{ display:block; width:100%; max-width:100%; height:auto; }}
main > .status,
main > .guide,
main > .downloads,
main > .caveat,
main > h2,
main > figure:not(.matrix) {{ width:calc(100% - 32px); max-width:1120px; margin-left:auto; margin-right:auto; }}
figure.matrix {{ width:calc(100% - 32px); max-width:none; margin-left:auto; margin-right:auto; overflow:visible; }}
figure.matrix img {{ width:100%; max-width:100%; }}
figcaption {{ color:var(--ink); font-size:inherit; padding:12px 0 0; }}
.caveat {{ margin-top:44px; padding-top:20px; border-top:1px solid var(--ink); max-width:82ch; }}
.guide {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--grid); border:1px solid var(--grid); margin-bottom:16px; }}
.guide div {{ background:var(--bg); padding:14px; }}
.guide strong {{ color:var(--ink); font-weight:400; }}
.guide p {{ margin:.5rem 0 0; }}
.downloads {{ color:var(--ink); margin-bottom:34px; }}
a {{ color:var(--ink); }}
@media(max-width:700px) {{
  .meta,.guide {{ grid-template-columns:1fr; }}
  header,
  main > .status,
  main > .guide,
  main > .downloads,
  main > .caveat,
  main > h2,
  main > figure:not(.matrix),
  figure.matrix {{ width:calc(100% - 24px); }}
}}
</style>
</head>
<body>
<header>
  <span class="eyebrow">hrrs.ai / atmospheric model diagnostics</span>
  <h1>AIFS v1.1 input shift / IFS 49r1 → 50r1</h1>
  <p class="lede">A distribution-level comparison of public ECMWF step-zero
  input proxies, regridded to N320 and expressed in AIFS checkpoint units.</p>
  <div class="meta">
    <div><strong>{unique_states}</strong>unique six-hourly states</div>
    <div><strong>{selected_gib:.2f} GiB</strong>selected GRIB payload</div>
    <div><strong>{transferred_gib:.2f} GiB</strong>estimated HTTP transfer</div>
  </div>
</header>
<main>
  {results}
  {''.join(cards)}
  <section class="caveat">
    <span class="eyebrow">interpretation boundary</span>
    <p>The cohorts do not provide the same weather dates under both cycles.
    Cycle, weather, seasonal progression and interannual variability are not
    perfectly isolated. Agreement between the same-season and cutover
    comparisons is evidence of repeatability, not a causal estimate.</p>
    <p>Methods and reproducibility details:
    <a href="../public_input_comparison_methods.md">local methods note</a>.
    Official references:
    <a href="https://confluence.ecmwf.int/spaces/DAC/pages/272310539/ECMWF+open+data+real-time+forecasts+from+IFS+and+AIFS">ECMWF Open Data</a>,
    <a href="https://huggingface.co/ecmwf/aifs-single-1.1">AIFS v1.1 checkpoint</a>,
    and
    <a href="https://confluence.ecmwf.int/spaces/UDOC/pages/599165906/AIFS+How+To+Generate+a+forecast+with+the+AIFS">AIFS inference guidance</a>.
    </p>
  </section>
</main>
</body>
</html>
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", type=Path, default=COMPARISONS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_theme()
    DISPLAY_ORDER_CSV.parent.mkdir(parents=True, exist_ok=True)
    display_order_frame().to_csv(DISPLAY_ORDER_CSV, index=False)
    figures = [plot_cohort_design(), plot_field_inventory()]
    comparisons = None
    if args.comparisons.exists():
        comparisons = pd.read_csv(args.comparisons, dtype={"utc": str})
        for kind in ("state", "tendency"):
            for metric in ("mean", "variance", "wasserstein"):
                figures.append(
                    plot_complete_metric(comparisons, kind=kind, metric=metric)
                )
    build_report(figures, comparisons)
    print(
        json.dumps(
            {
                "figures": [str(path.relative_to(ROOT)) for path in figures],
                "report": str(REPORT.relative_to(ROOT)),
                "results_included": comparisons is not None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
