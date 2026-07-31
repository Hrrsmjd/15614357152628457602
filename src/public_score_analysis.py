"""Extract and compare the raw aggregates embedded in ECMWF AIFS scorecards.

This module deliberately distinguishes public aggregate evidence from the
requested controlled experiment. The two source scorecards have different
sample populations and, for analysis metrics, different verification truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

SCORECARDS = {
    "v2_50r1_vs_v11_49r1": {
        "path": SOURCES / "scorecard_v2_50r1_vs_v11_49r1.html",
        "control_system": "AIFS-Single-v1.1_from_49r1",
        "experiment_system": "AIFS-Single-v2_from_50r1",
        "control_analysis_truth": "IFS_49r1_analysis",
        "experiment_analysis_truth": "IFS_50r1_analysis_expver_0080",
        "definition": (
            "class(control=ai,experiment=rd); "
            "expver(control=0001,experiment=j0b1)"
        ),
    },
    "v2_50r1_vs_v11_50r1": {
        "path": SOURCES / "scorecard_v2_vs_v11_from_50r1.html",
        "control_system": "AIFS-Single-v1.1_from_50r1",
        "experiment_system": "AIFS-Single-v2_from_50r1",
        "control_analysis_truth": "IFS_50r1_analysis_expver_0080",
        "experiment_analysis_truth": "IFS_50r1_analysis_expver_0080",
        "definition": (
            "class=rd; expver(control=j4av,experiment=j0b1); "
            "refclass=od; refexpver=0080"
        ),
    },
}

VARIABLE_LABELS = {
    ("z", "500"): "Z500",
    ("t", "850"): "T850",
    ("2t", "0"): "2 m temperature",
    ("10ff", "0"): "10 m wind",
    ("msl", "0"): "MSLP",
}
REGION_LABELS = {
    "n.hem": "Northern extratropics",
    "tropics": "Tropics",
    "arctic": "Arctic",
}
FOCUS_STEPS = [24, 48, 72, 96, 120]


def extract_subpages_data(path: Path) -> dict[str, Any]:
    """Read the JSON object assigned to ``SubPagesData`` in scorecard HTML."""
    text = path.read_text(encoding="utf-8")
    marker = "SubPagesData = "
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"{path}: SubPagesData declaration not found")
    data, _ = json.JSONDecoder().raw_decode(text[start + len(marker) :])
    if "steps" not in data:
        raise ValueError(f"{path}: malformed SubPagesData object")
    return data


def _truth(reference: str, system_role: str, metadata: dict[str, str]) -> str:
    if reference == "ob":
        return "observations"
    return metadata[f"{system_role}_analysis_truth"]


def scorecard_to_frame(
    source_name: str, metadata: dict[str, Any], data: dict[str, Any]
) -> pd.DataFrame:
    """Convert one embedded scorecard object to tidy raw-score rows."""
    rows: list[dict[str, Any]] = []
    steps = data["steps"]
    for key, payload in data.items():
        if key in {"steps", "max_sample_size"}:
            continue
        parts = key.split("|")
        if len(parts) != 5:
            continue
        parameter, level, region, metric, reference = parts
        for role in ("control", "experiment"):
            values = payload[role]["mean"]
            if len(values) != len(steps):
                raise ValueError(f"{source_name}:{key}:{role}: step/value mismatch")
            for i, (step, value) in enumerate(zip(steps, values, strict=True)):
                rows.append(
                    {
                        "source_scorecard": source_name,
                        "source_definition": metadata["definition"],
                        "score_key": key,
                        "parameter": parameter,
                        "level_hpa": level,
                        "region": region,
                        "metric": metric,
                        "reference_type": reference,
                        "verification_truth": _truth(reference, role, metadata),
                        "system_role": role,
                        "system": metadata[f"{role}_system"],
                        "lead_hours": int(step),
                        "mean_score": float(value),
                        "units": payload["units"],
                        "comparison_population": int(payload["popul"][i]),
                    }
                )
    return pd.DataFrame(rows)


def derive_v11_cycle_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    """Subtract public v1.1 aggregate means with explicit validity flags.

    This is *not* asserted to be an exact controlled estimate. The scorecards
    expose aggregate means from samples differing by one initialization, not
    date-level values that can be re-matched.
    """
    keys = [
        "score_key",
        "parameter",
        "level_hpa",
        "region",
        "metric",
        "reference_type",
        "lead_hours",
        "units",
    ]
    f49 = raw[
        (raw["source_scorecard"] == "v2_50r1_vs_v11_49r1")
        & (raw["system_role"] == "control")
    ].copy()
    f50 = raw[
        (raw["source_scorecard"] == "v2_50r1_vs_v11_50r1")
        & (raw["system_role"] == "control")
    ].copy()
    f49 = f49[keys + ["mean_score", "comparison_population", "verification_truth"]]
    f50 = f50[keys + ["mean_score", "comparison_population", "verification_truth"]]
    out = f49.merge(f50, on=keys, how="inner", suffixes=("_49r1", "_50r1"))
    out = out.rename(
        columns={
            "mean_score_49r1": "score_49r1",
            "mean_score_50r1": "score_50r1",
            "comparison_population_49r1": "sample_count_49r1_scorecard",
            "comparison_population_50r1": "sample_count_50r1_scorecard",
            "verification_truth_49r1": "truth_49r1_scorecard",
            "verification_truth_50r1": "truth_50r1_scorecard",
        }
    )
    out["delta_50r1_minus_49r1"] = out["score_50r1"] - out["score_49r1"]
    denominator = out["score_49r1"].abs()
    out["normalized_delta_percent_of_49r1"] = np.where(
        denominator > 1e-12,
        100.0 * out["delta_50r1_minus_49r1"] / denominator,
        np.nan,
    )
    out["same_truth_type"] = out["reference_type"].eq("ob")
    out["sample_counts_equal"] = out["sample_count_49r1_scorecard"].eq(
        out["sample_count_50r1_scorecard"]
    )
    # The pages do not expose the date membership behind each aggregate.
    out["exact_date_membership_available"] = False
    out["is_requested_controlled_comparison"] = False
    analysis = out["reference_type"].eq("an")
    activity = out["metric"].eq("sdaf")
    out["validity"] = "observation_truth_but_unmatched_aggregate_samples"
    out.loc[analysis & ~activity, "validity"] = (
        "different_analysis_truth_and_unmatched_aggregate_samples"
    )
    out.loc[analysis & activity, "validity"] = (
        "forecast_activity_from_unmatched_aggregate_samples"
    )
    out["sign_convention"] = np.select(
        [
            out["metric"].eq("rmsef"),
            out["metric"].eq("ccaf"),
            out["metric"].eq("sdaf"),
        ],
        [
            "positive delta means 50r1 has larger RMSE (worse)",
            "positive delta means 50r1 has larger ACC (better)",
            "positive delta means 50r1 is more active",
        ],
        default="positive delta means 50r1 score is larger",
    )
    return out.sort_values(keys).reset_index(drop=True)


def _focus_mask(frame: pd.DataFrame) -> pd.Series:
    pairs = pd.Series(
        list(zip(frame["parameter"], frame["level_hpa"], strict=True)),
        index=frame.index,
    )
    return (
        pairs.isin(VARIABLE_LABELS)
        & frame["region"].isin(REGION_LABELS)
        & frame["lead_hours"].isin(FOCUS_STEPS)
    )


def write_summary_table(derived: pd.DataFrame) -> pd.DataFrame:
    focus = derived[_focus_mask(derived)].copy()
    surface = focus["parameter"].isin(["2t", "10ff"])
    focus = focus[
        (surface & focus["reference_type"].eq("ob"))
        | (~surface & focus["reference_type"].eq("an"))
    ]
    focus["variable"] = [
        VARIABLE_LABELS[(p, level)]
        for p, level in zip(focus["parameter"], focus["level_hpa"], strict=True)
    ]
    columns = [
        "variable",
        "region",
        "lead_hours",
        "metric",
        "score_49r1",
        "score_50r1",
        "delta_50r1_minus_49r1",
        "normalized_delta_percent_of_49r1",
        "units",
        "sample_count_49r1_scorecard",
        "sample_count_50r1_scorecard",
        "truth_49r1_scorecard",
        "truth_50r1_scorecard",
        "validity",
        "sign_convention",
    ]
    table = focus[columns].sort_values(["region", "variable", "lead_hours"])
    table.to_csv(OUTPUTS / "public_aggregate_summary_table.csv", index=False)
    return table


def plot_rmse_scorecard(derived: pd.DataFrame) -> None:
    focus = derived[_focus_mask(derived) & derived["metric"].eq("rmsef")].copy()
    surface = focus["parameter"].isin(["2t", "10ff"])
    focus = focus[
        (surface & focus["reference_type"].eq("ob"))
        | (~surface & focus["reference_type"].eq("an"))
    ]
    variables = list(VARIABLE_LABELS.values())
    values = focus["normalized_delta_percent_of_49r1"].to_numpy()
    vmax = max(5.0, float(np.nanpercentile(np.abs(values), 95)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    image = None
    for ax, (region, region_label) in zip(
        axes, REGION_LABELS.items(), strict=True
    ):
        matrix = np.full((len(variables), len(FOCUS_STEPS)), np.nan)
        notes = np.full(matrix.shape, "", dtype=object)
        subset = focus[focus["region"].eq(region)]
        for _, row in subset.iterrows():
            label = VARIABLE_LABELS[(row["parameter"], row["level_hpa"])]
            i = variables.index(label)
            j = FOCUS_STEPS.index(int(row["lead_hours"]))
            matrix[i, j] = row["normalized_delta_percent_of_49r1"]
            notes[i, j] = "†" if row["reference_type"] == "an" else "*"
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                color = "white" if abs(value) > 0.55 * vmax else "black"
                ax.text(
                    j,
                    i,
                    f"{value:+.1f}{notes[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )
        ax.set_title(region_label)
        ax.set_xticks(range(len(FOCUS_STEPS)), FOCUS_STEPS)
        ax.set_yticks(range(len(variables)), variables)
        ax.set_xlabel("Lead (h)")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.02)
    colorbar.set_label(
        "100 × (aggregate RMSE50 − aggregate RMSE49) / aggregate RMSE49 (%)\n"
        "red/positive: 50r1 worse; blue/negative: 50r1 better"
    )
    fig.suptitle(
        "Public aggregate scorecard — diagnostic only, NOT a controlled estimate\n"
        "* common observation type but samples differ by one case; "
        "† analysis truth differs and samples differ",
        fontsize=12,
    )
    fig.savefig(FIGURES / "public_aggregate_rmse_scorecard.png", dpi=180)
    plt.close(fig)


def plot_observation_lead_time(derived: pd.DataFrame) -> None:
    subset = derived[
        derived["parameter"].isin(["2t", "10ff"])
        & derived["reference_type"].eq("ob")
        & derived["metric"].eq("rmsef")
        & derived["region"].isin(REGION_LABELS)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, parameter in zip(axes, ["2t", "10ff"], strict=True):
        for region, region_label in REGION_LABELS.items():
            rows = subset[
                subset["parameter"].eq(parameter) & subset["region"].eq(region)
            ].sort_values("lead_hours")
            ax.plot(
                rows["lead_hours"],
                rows["delta_50r1_minus_49r1"],
                marker="o",
                linewidth=1.5,
                label=region_label,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(VARIABLE_LABELS[(parameter, "0")])
        ax.set_xlabel("Lead (h)")
        ax.set_ylabel(
            f"Aggregate RMSE50 − RMSE49 ({rows['units'].iloc[0]})\n"
            "positive: 50r1 worse"
        )
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Observation-verified public aggregates (samples differ by one case;\n"
        "not the exact matched-date controlled estimate)"
    )
    fig.savefig(FIGURES / "public_aggregate_observation_lead_time.png", dpi=180)
    plt.close(fig)


def plot_activity_scorecard(derived: pd.DataFrame) -> None:
    focus = derived[_focus_mask(derived) & derived["metric"].eq("sdaf")].copy()
    # SDAF is available under the analysis section; it describes forecast
    # anomaly activity, but the page aggregates still have different samples.
    focus = focus[focus["reference_type"].eq("an")]
    variables = list(VARIABLE_LABELS.values())
    values = focus["normalized_delta_percent_of_49r1"].to_numpy()
    vmax = max(2.0, float(np.nanpercentile(np.abs(values), 95)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    image = None
    for ax, (region, region_label) in zip(
        axes, REGION_LABELS.items(), strict=True
    ):
        matrix = np.full((len(variables), len(FOCUS_STEPS)), np.nan)
        subset = focus[focus["region"].eq(region)]
        for _, row in subset.iterrows():
            label = VARIABLE_LABELS[(row["parameter"], row["level_hpa"])]
            matrix[variables.index(label), FOCUS_STEPS.index(row["lead_hours"])] = (
                row["normalized_delta_percent_of_49r1"]
            )
        image = ax.imshow(matrix, cmap="PRGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(region_label)
        ax.set_xticks(range(len(FOCUS_STEPS)), FOCUS_STEPS)
        ax.set_yticks(range(len(variables)), variables)
        ax.set_xlabel("Lead (h)")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.02)
    colorbar.set_label(
        "100 × (aggregate SDAF50 − aggregate SDAF49) / aggregate SDAF49 (%)\n"
        "positive/purple: 50r1 more active; negative/green: less active"
    )
    fig.suptitle(
        "Forecast-activity diagnostic from public aggregates\n"
        "Samples differ by one case; not an exact matched-date estimate"
    )
    fig.savefig(FIGURES / "public_aggregate_activity_scorecard.png", dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = []
    source_summary = []
    for source_name, metadata in SCORECARDS.items():
        path = metadata["path"]
        data = extract_subpages_data(path)
        frames.append(scorecard_to_frame(source_name, metadata, data))
        source_summary.append(
            {
                "source": source_name,
                "local_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "steps": data["steps"],
                "max_sample_size": data["max_sample_size"],
            }
        )
    raw = pd.concat(frames, ignore_index=True)
    derived = derive_v11_cycle_comparison(raw)
    raw.to_csv(PROCESSED / "public_scorecard_raw_scores.csv", index=False)
    raw.to_parquet(PROCESSED / "public_scorecard_raw_scores.parquet", index=False)
    derived.to_csv(PROCESSED / "public_v11_cycle_aggregate_differences.csv", index=False)
    derived.to_parquet(
        PROCESSED / "public_v11_cycle_aggregate_differences.parquet", index=False
    )
    table = write_summary_table(derived)
    plot_rmse_scorecard(derived)
    plot_observation_lead_time(derived)
    plot_activity_scorecard(derived)
    summary = {
        "sources": source_summary,
        "raw_rows": len(raw),
        "derived_rows": len(derived),
        "summary_rows": len(table),
        "controlled_comparison_complete": False,
        "reason": (
            "The public scorecards have different case populations and use "
            "different analysis truth for analysis-based 49r1 versus 50r1 scores."
        ),
    }
    (OUTPUTS / "public_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
