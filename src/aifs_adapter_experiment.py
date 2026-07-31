"""Evaluate frozen, operationally deployable adapters for IFS 50r1 inputs.

The original correction experiment cross-fits a detailed residual-bias
profile across all 32 post-cutover cases.  This module uses a chronological
split instead:

* calibration: all cutover 49r1 states and 50r1 states through
  2026-05-16 12 UTC;
* evaluation: 50r1 initializations from 2026-05-17 00 UTC onward.

The six-hour lag of the first evaluation case is 2026-05-16 18 UTC, so no
evaluation input state appears in calibration.  Fitted profiles are frozen
and require no ERA5T data at forecast initialization time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

import aifs_forecast_experiment as base


ROOT = base.ROOT
ADAPTER_ROOT = base.EXPERIMENT_ROOT / "adapter_experiments"
FROZEN_RESIDUAL_STATES = (
    ADAPTER_ROOT / "frozen_residual_state_medians.parquet"
)
FROZEN_RESIDUAL_PROFILE = ADAPTER_ROOT / "frozen_residual_profiles.parquet"
FROZEN_RESIDUAL_PROFILE_CSV = FROZEN_RESIDUAL_PROFILE.with_suffix(".csv")
Q50_AFFINE_PROFILE = ADAPTER_ROOT / "q50_robust_affine_profiles.parquet"
Q50_AFFINE_PROFILE_CSV = Q50_AFFINE_PROFILE.with_suffix(".csv")

CALIBRATION_50_START = datetime(2026, 5, 13, 0, tzinfo=timezone.utc)
CALIBRATION_50_STOP = datetime(2026, 5, 16, 12, tzinfo=timezone.utc)
EVALUATION_START = datetime(2026, 5, 17, 0, tzinfo=timezone.utc)

Q_TARGETS = ("q_50", "q_100", "q_150")
GUARDRAIL_VARIABLES = ("z_500", "t_850", "2t", "10u", "10v", "msl")
LATITUDE_BAND_WIDTH = 10.0
PRESSURE_PREFIXES = ("z_", "t_", "u_", "v_", "w_", "q_")
DYNAMIC_VARIABLES = tuple(
    field.aifs_name for field in base.DYNAMIC_FIELD_KEYS
)
PRESSURE_VARIABLES = tuple(
    variable
    for variable in DYNAMIC_VARIABLES
    if variable.startswith(PRESSURE_PREFIXES)
)
SURFACE_VARIABLES = tuple(
    variable
    for variable in DYNAMIC_VARIABLES
    if variable not in PRESSURE_VARIABLES
)


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    residual_variables: frozenset[str]
    residual_strength: float = 1.0
    q50_affine: bool = False
    q50_affine_bands: frozenset[int] | None = None
    description: str = ""


_ALL_DYNAMIC = frozenset(DYNAMIC_VARIABLES)
ADAPTERS: dict[str, AdapterSpec] = {
    "residual_all": AdapterSpec(
        name="residual_all",
        residual_variables=_ALL_DYNAMIC,
        description=(
            "Frozen additive IFS-minus-ERA5T residual alignment for all "
            "90 dynamic fields."
        ),
    ),
    "residual_all_no_q50": AdapterSpec(
        name="residual_all_no_q50",
        residual_variables=_ALL_DYNAMIC - {"q_50"},
        description=(
            "All-field residual alignment, leaving q50 unchanged because the "
            "original additive correction reduced its bias but increased RMSE."
        ),
    ),
    "residual_all_no_q50_half": AdapterSpec(
        name="residual_all_no_q50_half",
        residual_variables=_ALL_DYNAMIC - {"q_50"},
        residual_strength=0.5,
        description=(
            "Half-strength regularized all-field residual alignment with "
            "q50 unchanged."
        ),
    ),
    "residual_q100_q150": AdapterSpec(
        name="residual_q100_q150",
        residual_variables=frozenset({"q_100", "q_150"}),
        description=(
            "Targeted additive residual alignment for q100 and q150 only."
        ),
    ),
    "hybrid_q50_affine_q100_q150": AdapterSpec(
        name="hybrid_q50_affine_q100_q150",
        residual_variables=frozenset({"q_100", "q_150"}),
        q50_affine=True,
        description=(
            "Robust 50r1-to-49r1 location/scale mapping for q50 plus additive "
            "ERA5T-residual alignment for q100 and q150."
        ),
    ),
    "hybrid_q50_affine_extratropics_q100_q150": AdapterSpec(
        name="hybrid_q50_affine_extratropics_q100_q150",
        residual_variables=frozenset({"q_100", "q_150"}),
        q50_affine=True,
        q50_affine_bands=frozenset((*range(0, 7), *range(11, 18))),
        description=(
            "Targeted q100/q150 residual alignment plus robust q50 "
            "location/scale mapping outside the 20S-20N tropical bands only."
        ),
    ),
}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_frame_pair(frame: pd.DataFrame, parquet_path: Path) -> None:
    base._write_frame_atomic(frame, parquet_path)
    base._write_frame_atomic(frame, parquet_path.with_suffix(".csv"))


def _as_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def calibration_initializations() -> pd.DataFrame:
    """Return the t0 states used to fit the frozen profiles."""
    frame = base.cohort_initializations().copy()
    timestamps = _as_datetime(frame["timestamp"])
    cutover49 = frame[
        frame["memberships"].str.contains("cutover:49r1", regex=False)
    ]
    train50 = frame[
        frame["cycle"].eq("50r1")
        & timestamps.ge(CALIBRATION_50_START)
        & timestamps.le(CALIBRATION_50_STOP)
    ]
    result = (
        pd.concat([cutover49, train50], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    counts = result.groupby("cycle").size().to_dict()
    if counts != {"49r1": 32, "50r1": 15}:
        raise RuntimeError(f"Unexpected frozen calibration counts: {counts}")
    return result


def evaluation_initializations() -> pd.DataFrame:
    """Return the 16 strictly post-calibration 50r1 forecast cases."""
    frame = base.cohort_initializations().copy()
    timestamps = _as_datetime(frame["timestamp"])
    result = (
        frame[frame["cycle"].eq("50r1") & timestamps.ge(EVALUATION_START)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(result) != 16:
        raise RuntimeError(f"Expected 16 evaluation cases, found {len(result)}")
    calibration_states = {
        datetime.fromisoformat(str(value))
        for value in calibration_initializations()["timestamp"]
        if datetime.fromisoformat(str(value)).year == 2026
    }
    evaluation_states: set[datetime] = set()
    for value in result["timestamp"]:
        timestamp = datetime.fromisoformat(str(value))
        evaluation_states.update((timestamp - timedelta(hours=6), timestamp))
    overlap = calibration_states & evaluation_states
    if overlap:
        raise RuntimeError(
            "Calibration/evaluation input-state overlap: "
            + ", ".join(sorted(value.isoformat() for value in overlap))
        )
    return result


@lru_cache(maxsize=1)
def stratum_geometry() -> dict[str, Any]:
    """Return fixed 10-degree latitude and land/ocean masks on N320."""
    latitudes, _, _, region_masks = base.evaluation_geometry()
    band = np.floor(
        (latitudes + 90.0) / LATITUDE_BAND_WIDTH
    ).astype(int)
    band = np.clip(band, 0, 17)
    masks: dict[tuple[int, str], np.ndarray] = {}
    for latitude_band in range(18):
        latitude_mask = band == latitude_band
        masks[(latitude_band, "all")] = latitude_mask
        masks[(latitude_band, "land")] = (
            latitude_mask & region_masks["global_land"]
        )
        masks[(latitude_band, "ocean")] = (
            latitude_mask & region_masks["global_ocean"]
        )
    return {"latitudes": latitudes, "band": band, "masks": masks}


def _field_categories(variable: str) -> tuple[str, ...]:
    if variable in PRESSURE_VARIABLES:
        return ("all",)
    if variable.startswith(("stl", "swvl")):
        return ("land",)
    return ("land", "ocean")


def build_frozen_residual_states(*, force: bool = False) -> pd.DataFrame:
    """Compute calibration-state medians of IFS step-0 minus ERA5T."""
    if FROZEN_RESIDUAL_STATES.exists() and not force:
        return pd.read_parquet(FROZEN_RESIDUAL_STATES)
    calibration = calibration_initializations()
    masks = stratum_geometry()["masks"]
    rows: list[dict[str, Any]] = []
    for position, record in enumerate(
        calibration.itertuples(index=False), start=1
    ):
        timestamp = datetime.fromisoformat(str(record.timestamp))
        input_path = (
            base.INPUT_CACHE / f"{base.timestamp_id(timestamp)}.npz"
        )
        reference_path = base.reference_path(timestamp)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if not reference_path.exists():
            raise FileNotFoundError(reference_path)
        with (
            np.load(input_path) as input_state,
            np.load(reference_path) as reference_state,
        ):
            for variable in DYNAMIC_VARIABLES:
                residual = (
                    np.asarray(input_state[variable], dtype=np.float32)
                    - np.asarray(reference_state[variable], dtype=np.float32)
                )
                for latitude_band in range(18):
                    for category in _field_categories(variable):
                        mask = masks[(latitude_band, category)]
                        if not np.any(mask):
                            continue
                        values = residual[mask]
                        finite = np.isfinite(values)
                        rows.append(
                            {
                                "timestamp": timestamp.isoformat(),
                                "cycle": str(record.cycle),
                                "variable": variable,
                                "latitude_band": latitude_band,
                                "latitude_min": (
                                    -90 + 10 * latitude_band
                                ),
                                "latitude_max": (
                                    -80 + 10 * latitude_band
                                ),
                                "surface_category": category,
                                "spatial_median_residual": float(
                                    np.median(values[finite])
                                ),
                                "finite_gridpoint_count": int(finite.sum()),
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "frozen_residual_state_summarized",
                    "position": position,
                    "total": len(calibration),
                    "timestamp": timestamp.isoformat(),
                }
            ),
            flush=True,
        )
    result = pd.DataFrame(rows)
    state_counts = result.groupby(
        [
            "cycle",
            "variable",
            "latitude_band",
            "surface_category",
        ],
        sort=False,
    )["timestamp"].nunique()
    expected_counts = {
        "49r1": 32,
        "50r1": 15,
    }
    for cycle, expected in expected_counts.items():
        observed = set(
            state_counts.loc[cycle].to_numpy(dtype=int)
        )
        if observed != {expected}:
            raise RuntimeError(
                f"Unexpected {cycle} residual-state counts: {observed}"
            )
    _write_frame_pair(result, FROZEN_RESIDUAL_STATES)
    return result


def build_frozen_residual_profiles(*, force: bool = False) -> pd.DataFrame:
    """Aggregate calibration residual summaries without UTC strata."""
    if FROZEN_RESIDUAL_PROFILE.exists() and not force:
        return pd.read_parquet(FROZEN_RESIDUAL_PROFILE)
    summaries = build_frozen_residual_states(force=force)
    cohort49 = summaries[summaries["cycle"].eq("49r1")].copy()
    cohort50 = summaries[summaries["cycle"].eq("50r1")].copy()
    keys = [
        "variable",
        "latitude_band",
        "latitude_min",
        "latitude_max",
        "surface_category",
    ]
    median49 = (
        cohort49.groupby(keys, sort=True)["spatial_median_residual"]
        .agg(median_residual_49r1="median", samples_49r1="size")
        .reset_index()
    )
    median50 = (
        cohort50.groupby(keys, sort=True)["spatial_median_residual"]
        .agg(median_residual_50r1="median", samples_50r1="size")
        .reset_index()
    )
    profile = median49.merge(median50, on=keys, validate="one_to_one")
    if set(profile["samples_49r1"]) != {32}:
        raise RuntimeError("Every residual profile must contain 32 49r1 states")
    if set(profile["samples_50r1"]) != {15}:
        raise RuntimeError("Every residual profile must contain 15 50r1 states")
    profile["correction_d"] = (
        profile["median_residual_50r1"] - profile["median_residual_49r1"]
    )
    profile["calibration_50_stop"] = CALIBRATION_50_STOP.isoformat()
    profile = profile[
        [
            *keys,
            "samples_49r1",
            "samples_50r1",
            "median_residual_49r1",
            "median_residual_50r1",
            "correction_d",
            "calibration_50_stop",
        ]
    ]
    _write_frame_pair(profile, FROZEN_RESIDUAL_PROFILE)
    _residual_lookup.cache_clear()
    return profile


@lru_cache(maxsize=1)
def _residual_lookup() -> dict[tuple[str, int, str], float]:
    profiles = build_frozen_residual_profiles()
    return {
        (
            str(row.variable),
            int(row.latitude_band),
            str(row.surface_category),
        ): float(row.correction_d)
        for row in profiles.itertuples(index=False)
    }


def build_q50_affine_profiles(*, force: bool = False) -> pd.DataFrame:
    """Fit an equal-state-weight robust q50 marginal location/scale map."""
    if Q50_AFFINE_PROFILE.exists() and not force:
        return pd.read_parquet(Q50_AFFINE_PROFILE)
    masks = stratum_geometry()["masks"]
    rows: list[dict[str, Any]] = []
    for record in calibration_initializations().itertuples(index=False):
        timestamp = datetime.fromisoformat(str(record.timestamp))
        path = base.INPUT_CACHE / f"{base.timestamp_id(timestamp)}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as state:
            values = np.asarray(state["q_50"], dtype=np.float32)
            for latitude_band in range(18):
                current = values[masks[(latitude_band, "all")]]
                current = current[np.isfinite(current)]
                q25, median, q75 = np.quantile(current, (0.25, 0.5, 0.75))
                rows.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "cycle": str(record.cycle),
                        "latitude_band": latitude_band,
                        "state_median": float(median),
                        "state_iqr": float(q75 - q25),
                    }
                )
    states = pd.DataFrame(rows)
    grouped = (
        states.groupby(["cycle", "latitude_band"], sort=True)
        .agg(
            location=("state_median", "median"),
            scale=("state_iqr", "median"),
            samples=("timestamp", "nunique"),
        )
        .reset_index()
    )
    profile49 = grouped[grouped["cycle"].eq("49r1")].drop(columns="cycle")
    profile50 = grouped[grouped["cycle"].eq("50r1")].drop(columns="cycle")
    profile = profile49.merge(
        profile50,
        on="latitude_band",
        suffixes=("_49r1", "_50r1"),
        validate="one_to_one",
    )
    if set(profile["samples_49r1"]) != {32}:
        raise RuntimeError("Every q50 affine profile must contain 32 49r1 states")
    if set(profile["samples_50r1"]) != {15}:
        raise RuntimeError("Every q50 affine profile must contain 15 50r1 states")
    raw_ratio = profile["scale_49r1"] / profile["scale_50r1"]
    profile["scale_ratio_raw"] = raw_ratio
    profile["scale_ratio"] = raw_ratio.clip(lower=0.25, upper=4.0)
    profile["scale_ratio_clipped"] = profile["scale_ratio"].ne(raw_ratio)
    profile["calibration_50_stop"] = CALIBRATION_50_STOP.isoformat()
    _write_frame_pair(profile, Q50_AFFINE_PROFILE)
    _q50_affine_lookup.cache_clear()
    return profile


@lru_cache(maxsize=1)
def _q50_affine_lookup() -> dict[int, tuple[float, float, float]]:
    profile = build_q50_affine_profiles()
    return {
        int(row.latitude_band): (
            float(row.location_49r1),
            float(row.location_50r1),
            float(row.scale_ratio),
        )
        for row in profile.itertuples(index=False)
    }


def _apply_physical_bounds(
    values: np.ndarray, variable: str
) -> tuple[np.ndarray, int, int]:
    lower_count = 0
    upper_count = 0
    if variable.startswith("q_"):
        lower = values < 0.0
        lower_count = int(lower.sum())
        np.maximum(values, 0.0, out=values)
    elif variable.startswith("swvl"):
        lower = values < 0.0
        upper = values > 1.0
        lower_count = int(lower.sum())
        upper_count = int(upper.sum())
        np.clip(values, 0.0, 1.0, out=values)
    return values, lower_count, upper_count


def apply_adapter_field(
    values: np.ndarray,
    variable: str,
    adapter_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one frozen adapter field transformation."""
    spec = ADAPTERS[adapter_name]
    corrected = np.asarray(values, dtype=np.float32).copy()
    masks = stratum_geometry()["masks"]
    transformation = "identity"
    magnitudes: list[float] = []
    if spec.q50_affine and variable == "q_50":
        lookup = _q50_affine_lookup()
        transformation = "robust_affine"
        for latitude_band in range(18):
            if (
                spec.q50_affine_bands is not None
                and latitude_band not in spec.q50_affine_bands
            ):
                continue
            mask = masks[(latitude_band, "all")]
            location49, location50, ratio = lookup[latitude_band]
            corrected[mask] = (
                location49 + (corrected[mask] - location50) * ratio
            )
            magnitudes.append(ratio)
    elif variable in spec.residual_variables:
        lookup = _residual_lookup()
        transformation = "residual_additive"
        for latitude_band in range(18):
            for category in _field_categories(variable):
                mask = masks[(latitude_band, category)]
                if not np.any(mask):
                    continue
                correction_d = (
                    lookup[(variable, latitude_band, category)]
                    * spec.residual_strength
                )
                corrected[mask] -= correction_d
                magnitudes.append(correction_d)
    preclip_min = float(np.nanmin(corrected))
    preclip_max = float(np.nanmax(corrected))
    lower_count = 0
    upper_count = 0
    if transformation != "identity":
        corrected, lower_count, upper_count = _apply_physical_bounds(
            corrected, variable
        )
    diagnostics = {
        "adapter": adapter_name,
        "variable": variable,
        "transformation": transformation,
        "gridpoint_count": int(corrected.size),
        "lower_clipped_count": lower_count,
        "upper_clipped_count": upper_count,
        "preclip_min": preclip_min,
        "preclip_max": preclip_max,
        "postclip_min": float(np.nanmin(corrected)),
        "postclip_max": float(np.nanmax(corrected)),
        "transform_parameter_min": (
            float(np.min(magnitudes)) if magnitudes else 0.0
        ),
        "transform_parameter_median": (
            float(np.median(magnitudes)) if magnitudes else 0.0
        ),
        "transform_parameter_max": (
            float(np.max(magnitudes)) if magnitudes else 0.0
        ),
    }
    return corrected, diagnostics


def adapter_input_state(
    timestamp: datetime, adapter_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lag = timestamp - timedelta(hours=6)
    lag_path = base.INPUT_CACHE / f"{base.timestamp_id(lag)}.npz"
    now_path = base.INPUT_CACHE / f"{base.timestamp_id(timestamp)}.npz"
    static_path = base.INPUT_CACHE / "static_50r1.npz"
    fields: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    with np.load(lag_path) as lag_data, np.load(now_path) as now_data:
        for field in base.DYNAMIC_FIELD_KEYS:
            lag_values, lag_diagnostics = apply_adapter_field(
                lag_data[field.aifs_name], field.aifs_name, adapter_name
            )
            now_values, now_diagnostics = apply_adapter_field(
                now_data[field.aifs_name], field.aifs_name, adapter_name
            )
            lag_diagnostics["state_time"] = lag.isoformat()
            now_diagnostics["state_time"] = timestamp.isoformat()
            fields[field.aifs_name] = np.stack([lag_values, now_values])
            diagnostics.extend((lag_diagnostics, now_diagnostics))
    with np.load(static_path) as static_data:
        for field in base.STATIC_FIELD_KEYS:
            values = np.asarray(static_data[field.aifs_name], dtype=np.float32)
            fields[field.aifs_name] = np.stack([values, values])
    return {"date": timestamp, "fields": fields}, diagnostics


def adapter_directory(adapter_name: str) -> Path:
    return ADAPTER_ROOT / adapter_name


def adapter_part_path(adapter_name: str, timestamp: datetime) -> Path:
    return (
        adapter_directory(adapter_name)
        / "forecast_metric_parts"
        / f"{base.timestamp_id(timestamp)}.parquet"
    )


def audit_adapter(adapter_name: str) -> dict[str, Any]:
    """Audit all unique evaluation input states without running AIFS."""
    diagnostics: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for record in evaluation_initializations().itertuples(index=False):
        timestamp = datetime.fromisoformat(str(record.timestamp))
        for state_time in (timestamp - timedelta(hours=6), timestamp):
            if state_time in seen:
                continue
            seen.add(state_time)
            path = base.INPUT_CACHE / f"{base.timestamp_id(state_time)}.npz"
            with np.load(path) as state:
                for field in base.DYNAMIC_FIELD_KEYS:
                    _, current = apply_adapter_field(
                        state[field.aifs_name], field.aifs_name, adapter_name
                    )
                    current["state_time"] = state_time.isoformat()
                    diagnostics.append(current)
    frame = pd.DataFrame(diagnostics)
    clipping = (
        frame.groupby(["variable", "transformation"], sort=True)
        .agg(
            transformed_values=("gridpoint_count", "sum"),
            lower_clipped_values=("lower_clipped_count", "sum"),
            upper_clipped_values=("upper_clipped_count", "sum"),
            minimum_preclip=("preclip_min", "min"),
            maximum_preclip=("preclip_max", "max"),
        )
        .reset_index()
    )
    clipping["clipped_fraction"] = (
        clipping["lower_clipped_values"] + clipping["upper_clipped_values"]
    ) / clipping["transformed_values"]
    payload = {
        "adapter": adapter_name,
        "description": ADAPTERS[adapter_name].description,
        "calibration_50_start": CALIBRATION_50_START.isoformat(),
        "calibration_50_stop": CALIBRATION_50_STOP.isoformat(),
        "evaluation_start": EVALUATION_START.isoformat(),
        "calibration_cases": {
            key: int(value)
            for key, value in calibration_initializations()
            .groupby("cycle")
            .size()
            .items()
        },
        "evaluation_cases": len(evaluation_initializations()),
        "unique_evaluation_input_states": len(seen),
        "residual_profile_sha256": base.sha256(FROZEN_RESIDUAL_PROFILE),
        "q50_affine_profile_sha256": (
            base.sha256(Q50_AFFINE_PROFILE)
            if ADAPTERS[adapter_name].q50_affine
            else None
        ),
        "clipping": clipping.to_dict(orient="records"),
    }
    target = adapter_directory(adapter_name) / "input_audit.json"
    _write_json_atomic(target, payload)
    return payload


def run_adapter_forecast_and_score(
    timestamp: datetime,
    memberships: str,
    adapter_name: str,
    *,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Run one adapter forecast; retain metrics but not multi-GB field arrays."""
    from anemoi.inference.runners.simple import SimpleRunner

    target = adapter_part_path(adapter_name, timestamp)
    if target.exists() and not force:
        return target
    missing_reference = [
        base.reference_path(timestamp + timedelta(hours=lead))
        for lead in base.LEADS
        if not base.reference_path(timestamp + timedelta(hours=lead)).exists()
    ]
    if missing_reference:
        raise FileNotFoundError(
            f"Missing {len(missing_reference)} references; "
            f"first={missing_reference[0]}"
        )
    _, _, weights, masks = base.evaluation_geometry()
    input_payload, diagnostics = adapter_input_state(timestamp, adapter_name)
    runner = SimpleRunner(str(base.MODEL_PATH), device=device)
    rows: list[dict[str, Any]] = []
    retained_leads: list[int] = []
    available_variables: list[str] | None = None
    started = time.monotonic()
    for state in runner.run(input_state=input_payload, lead_time=240):
        lead = int((state["date"] - timestamp).total_seconds() // 3600)
        if lead not in base.LEADS:
            continue
        retained_leads.append(lead)
        current_variables = sorted(state["fields"])
        if available_variables is None:
            available_variables = current_variables
            missing = sorted(
                set(base.EVALUATION_VARIABLES) - set(current_variables)
            )
            if missing:
                raise ValueError(f"AIFS output is missing variables: {missing}")
        elif current_variables != available_variables:
            raise ValueError("AIFS output inventory changed within forecast")
        valid_time = timestamp + timedelta(hours=lead)
        with np.load(base.reference_path(valid_time)) as reference:
            units = json.loads(str(reference["_units_json"]))
            forecast_matrix = np.stack(
                [
                    np.asarray(state["fields"][variable], dtype=np.float32)
                    for variable in base.EVALUATION_VARIABLES
                ]
            )
            reference_matrix = np.stack(
                [
                    np.asarray(reference[variable], dtype=np.float32)
                    for variable in base.EVALUATION_VARIABLES
                ]
            )
            matrix_scores = base.score_field_matrix(
                forecast_matrix, reference_matrix, weights, masks
            )
            for variable_index, variable in enumerate(
                base.EVALUATION_VARIABLES
            ):
                for metrics in matrix_scores[variable_index]:
                    rows.append(
                        {
                            "initialization": timestamp.isoformat(),
                            "initialization_id": base.timestamp_id(timestamp),
                            "cycle": f"50r1-{adapter_name}",
                            "memberships": memberships,
                            "lead_hours": lead,
                            "valid_time": valid_time.isoformat(),
                            "reference_product": "ERA5T",
                            "variable": variable,
                            "units": units[variable],
                            **metrics,
                        }
                    )
        print(
            json.dumps(
                {
                    "event": "adapter_forecast_lead_scored",
                    "adapter": adapter_name,
                    "initialization": timestamp.isoformat(),
                    "lead_hours": lead,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )
    if tuple(retained_leads) != base.LEADS:
        raise RuntimeError(f"Incomplete adapter forecast leads: {retained_leads}")
    frame = pd.DataFrame(rows)
    expected_rows = (
        len(base.LEADS) * len(base.EVALUATION_VARIABLES) * len(base.REGIONS)
    )
    if len(frame) != expected_rows:
        raise RuntimeError(f"Metric row mismatch {len(frame)} != {expected_rows}")
    base._write_frame_atomic(frame, target)
    diagnostic_frame = pd.DataFrame(diagnostics)
    metadata = {
        "adapter": adapter_name,
        "description": ADAPTERS[adapter_name].description,
        "initialization": timestamp.isoformat(),
        "cycle": f"50r1-{adapter_name}",
        "memberships": memberships,
        "device": device,
        "model_sha256": base.sha256(base.MODEL_PATH),
        "residual_profile_sha256": base.sha256(FROZEN_RESIDUAL_PROFILE),
        "input_lower_clipped_values": int(
            diagnostic_frame["lower_clipped_count"].sum()
        ),
        "input_upper_clipped_values": int(
            diagnostic_frame["upper_clipped_count"].sum()
        ),
        "retained_leads": retained_leads,
        "available_variables": available_variables,
        "evaluated_variables": list(base.EVALUATION_VARIABLES),
        "metric_rows": len(frame),
        "elapsed_seconds": time.monotonic() - started,
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return target


def run_adapter_batch(
    adapter_name: str,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    device: str = "cuda",
    limit: int | None = None,
    force: bool = False,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    frame = evaluation_initializations()
    frame = frame.iloc[
        [
            position % shard_count == shard_index
            for position in range(len(frame))
        ]
    ]
    if limit is not None:
        frame = frame.iloc[:limit]
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        timestamp = datetime.fromisoformat(str(row.timestamp))
        started = time.monotonic()
        try:
            path = run_adapter_forecast_and_score(
                timestamp,
                str(row.memberships),
                adapter_name,
                device=device,
                force=force,
            )
        finally:
            gc.collect()
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "adapter_forecast_complete",
                    "adapter": adapter_name,
                    "position": position,
                    "total": len(frame),
                    "initialization": timestamp.isoformat(),
                    "path": str(path),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )


def paired_cell_statistics(
    mse_baseline: np.ndarray,
    mse_adapter: np.ndarray,
    bias_baseline: np.ndarray,
    bias_adapter: np.ndarray,
    *,
    draws: int,
    seed_parts: tuple[Any, ...],
) -> dict[str, Any]:
    """Compute paired cohort statistics using common bootstrap indices."""
    mse_baseline = np.asarray(mse_baseline, dtype=np.float64)
    mse_adapter = np.asarray(mse_adapter, dtype=np.float64)
    bias_baseline = np.asarray(bias_baseline, dtype=np.float64)
    bias_adapter = np.asarray(bias_adapter, dtype=np.float64)
    sizes = {
        len(mse_baseline),
        len(mse_adapter),
        len(bias_baseline),
        len(bias_adapter),
    }
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("Paired arrays must have one common non-zero length")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            mse_baseline,
            mse_adapter,
            bias_baseline,
            bias_adapter,
        )
    ):
        raise ValueError("Paired arrays must be finite")
    rmse_baseline = float(np.sqrt(np.mean(mse_baseline)))
    rmse_adapter = float(np.sqrt(np.mean(mse_adapter)))
    relative_change = 100.0 * (rmse_adapter / rmse_baseline - 1.0)
    bias_difference = float(np.mean(bias_adapter - bias_baseline))
    seed = int.from_bytes(
        hashlib.sha256(repr(seed_parts).encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(mse_baseline), size=(draws, len(mse_baseline))
    )
    bootstrap_relative = 100.0 * (
        np.sqrt(np.mean(mse_adapter[indices], axis=1))
        / np.sqrt(np.mean(mse_baseline[indices], axis=1))
        - 1.0
    )
    bootstrap_bias = np.mean(
        bias_adapter[indices] - bias_baseline[indices], axis=1
    )
    relative_ci = np.quantile(bootstrap_relative, (0.025, 0.975))
    bias_ci = np.quantile(bootstrap_bias, (0.025, 0.975))
    return {
        "rmse_baseline": rmse_baseline,
        "rmse_adapter": rmse_adapter,
        "relative_rmse_change_percent": relative_change,
        "relative_rmse_change_ci95_low": float(relative_ci[0]),
        "relative_rmse_change_ci95_high": float(relative_ci[1]),
        "relative_rmse_change_significant_95": bool(
            relative_ci[0] > 0 or relative_ci[1] < 0
        ),
        "bias_baseline": float(np.mean(bias_baseline)),
        "bias_adapter": float(np.mean(bias_adapter)),
        "bias_difference_adapter_minus_baseline": bias_difference,
        "bias_difference_ci95_low": float(bias_ci[0]),
        "bias_difference_ci95_high": float(bias_ci[1]),
        "bias_difference_significant_95": bool(
            bias_ci[0] > 0 or bias_ci[1] < 0
        ),
        "forecast_case_count": len(mse_baseline),
    }


def _paired_summary(frame: pd.DataFrame) -> dict[str, Any]:
    global_rows = frame[frame["region"].eq("global")]
    early_global = global_rows[global_rows["lead_hours"].le(120)]
    q100_q150 = early_global[
        early_global["variable"].isin(("q_100", "q_150"))
    ]
    q50 = early_global[early_global["variable"].eq("q_50")]
    guardrails = frame[
        frame["region"].isin(("global", "northern_extratropics"))
        & frame["lead_hours"].le(120)
        & frame["variable"].isin(GUARDRAIL_VARIABLES)
    ]
    significant = frame[frame["relative_rmse_change_significant_95"]]
    return {
        "global_all_cell_mean_percent": float(
            global_rows["relative_rmse_change_percent"].mean()
        ),
        "global_all_cell_median_percent": float(
            global_rows["relative_rmse_change_percent"].median()
        ),
        "early_global_q100_q150_mean_percent": float(
            q100_q150["relative_rmse_change_percent"].mean()
        ),
        "early_global_q50_mean_percent": float(
            q50["relative_rmse_change_percent"].mean()
        ),
        "guardrail_mean_percent": float(
            guardrails["relative_rmse_change_percent"].mean()
        ),
        "guardrail_worst_percent": float(
            guardrails["relative_rmse_change_percent"].max()
        ),
        "cells_improved": int(
            (frame["relative_rmse_change_percent"] < 0).sum()
        ),
        "cells_worsened": int(
            (frame["relative_rmse_change_percent"] > 0).sum()
        ),
        "significant_cells_improved": int(
            (significant["relative_rmse_change_percent"] < 0).sum()
        ),
        "significant_cells_worsened": int(
            (significant["relative_rmse_change_percent"] > 0).sum()
        ),
    }


def combine_adapter(
    adapter_name: str, *, bootstrap_draws: int = 4000
) -> dict[str, Any]:
    evaluation = evaluation_initializations()
    adapter_parts = [
        adapter_part_path(
            adapter_name, datetime.fromisoformat(str(timestamp))
        )
        for timestamp in evaluation["timestamp"]
    ]
    missing = [str(path) for path in adapter_parts if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} adapter parts; first={missing[0]}"
        )
    adapter_cases = pd.concat(
        [pd.read_parquet(path) for path in adapter_parts], ignore_index=True
    )
    baseline_parts = [
        base.forecast_part_path(datetime.fromisoformat(str(timestamp)))
        for timestamp in evaluation["timestamp"]
    ]
    baseline_cases = pd.concat(
        [pd.read_parquet(path) for path in baseline_parts], ignore_index=True
    )
    expected_rows = 16 * 5760
    if (
        len(adapter_cases) != expected_rows
        or len(baseline_cases) != expected_rows
    ):
        raise RuntimeError(
            "Unexpected paired case row counts: "
            f"{len(baseline_cases)}, {len(adapter_cases)}"
        )
    output_directory = adapter_directory(adapter_name)
    base._write_frame_atomic(
        adapter_cases, output_directory / "case_metrics.parquet"
    )
    keys = [
        "initialization_id",
        "variable",
        "units",
        "region",
        "lead_hours",
    ]
    paired = baseline_cases[
        keys + ["mean_squared_error", "bias", "valid_gridpoint_count"]
    ].merge(
        adapter_cases[
            keys + ["mean_squared_error", "bias", "valid_gridpoint_count"]
        ],
        on=keys,
        suffixes=("_baseline", "_adapter"),
        validate="one_to_one",
    )
    if not paired["valid_gridpoint_count_baseline"].equals(
        paired["valid_gridpoint_count_adapter"]
    ):
        raise RuntimeError("Baseline and adapter valid-point counts differ")
    rows: list[dict[str, Any]] = []
    group_keys = ["variable", "units", "region", "lead_hours"]
    for key, group in paired.groupby(group_keys, sort=True, dropna=False):
        variable, units, region, lead = key
        group = group.sort_values("initialization_id")
        statistics = paired_cell_statistics(
            group["mean_squared_error_baseline"].to_numpy(),
            group["mean_squared_error_adapter"].to_numpy(),
            group["bias_baseline"].to_numpy(),
            group["bias_adapter"].to_numpy(),
            draws=bootstrap_draws,
            seed_parts=(adapter_name, *key),
        )
        rows.append(
            {
                "adapter": adapter_name,
                "variable": variable,
                "units": units,
                "region": region,
                "lead_hours": int(lead),
                **statistics,
            }
        )
    comparisons = pd.DataFrame(rows)
    _write_frame_pair(
        comparisons, output_directory / "paired_comparisons.parquet"
    )
    payload = {
        "adapter": adapter_name,
        "description": ADAPTERS[adapter_name].description,
        "calibration_50_start": CALIBRATION_50_START.isoformat(),
        "calibration_50_stop": CALIBRATION_50_STOP.isoformat(),
        "evaluation_start": EVALUATION_START.isoformat(),
        "evaluation_cases": len(evaluation),
        "paired_comparison_rows": len(comparisons),
        "bootstrap_draws": bootstrap_draws,
        "residual_profile_sha256": base.sha256(FROZEN_RESIDUAL_PROFILE),
        "summary": _paired_summary(comparisons),
    }
    _write_json_atomic(output_directory / "summary.json", payload)
    return payload


def run_smoke(
    adapter_name: str,
    timestamp: datetime,
    *,
    lead_time: int = 24,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run a short adapter forecast and compare it with the retained baseline."""
    from anemoi.inference.runners.simple import SimpleRunner

    input_payload, diagnostics = adapter_input_state(timestamp, adapter_name)
    runner = SimpleRunner(str(base.MODEL_PATH), device=device)
    retained = None
    for state in runner.run(input_state=input_payload, lead_time=lead_time):
        if state["date"] == timestamp + timedelta(hours=lead_time):
            retained = state
    if retained is None:
        raise RuntimeError("Adapter smoke forecast did not produce target lead")
    _, _, weights, masks = base.evaluation_geometry()
    valid_time = timestamp + timedelta(hours=lead_time)
    variables = (*Q_TARGETS, *GUARDRAIL_VARIABLES)
    result: dict[str, Any] = {}
    baseline_part = pd.read_parquet(base.forecast_part_path(timestamp))
    with np.load(base.reference_path(valid_time)) as reference:
        for variable in variables:
            adapter_score = next(
                row
                for row in base.score_field(
                    retained["fields"][variable],
                    reference[variable],
                    weights,
                    masks,
                )
                if row["region"] == "global"
            )
            baseline_row = baseline_part[
                baseline_part["variable"].eq(variable)
                & baseline_part["region"].eq("global")
                & baseline_part["lead_hours"].eq(lead_time)
            ].iloc[0]
            result[variable] = {
                "baseline_rmse": float(baseline_row["rmse"]),
                "adapter_rmse": float(adapter_score["rmse"]),
                "relative_rmse_change_percent": float(
                    100.0
                    * (adapter_score["rmse"] / baseline_row["rmse"] - 1.0)
                ),
                "baseline_bias": float(baseline_row["bias"]),
                "adapter_bias": float(adapter_score["bias"]),
            }
    diagnostics_frame = pd.DataFrame(diagnostics)
    payload = {
        "adapter": adapter_name,
        "initialization": timestamp.isoformat(),
        "lead_hours": lead_time,
        "valid_time": valid_time.isoformat(),
        "scores": result,
        "input_lower_clipped_values": int(
            diagnostics_frame["lower_clipped_count"].sum()
        ),
        "input_upper_clipped_values": int(
            diagnostics_frame["upper_clipped_count"].sum()
        ),
    }
    target = (
        adapter_directory(adapter_name)
        / f"smoke_{base.timestamp_id(timestamp)}_lead{lead_time:03d}.json"
    )
    _write_json_atomic(target, payload)
    return payload


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser("audit-split")
    split_parser.add_argument("--json", action="store_true")

    profile_parser = commands.add_parser("build-profiles")
    profile_parser.add_argument("--force", action="store_true")

    audit_parser = commands.add_parser("audit-adapter")
    audit_parser.add_argument("adapter", choices=sorted(ADAPTERS))

    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("adapter", choices=sorted(ADAPTERS))
    smoke_parser.add_argument(
        "--timestamp", default="2026-05-17T00:00:00+00:00"
    )
    smoke_parser.add_argument("--lead-time", type=int, default=24)
    smoke_parser.add_argument("--device", default="cuda")

    batch_parser = commands.add_parser("run-batch")
    batch_parser.add_argument("adapter", choices=sorted(ADAPTERS))
    batch_parser.add_argument("--shard-index", type=int, default=0)
    batch_parser.add_argument("--shard-count", type=int, default=1)
    batch_parser.add_argument("--device", default="cuda")
    batch_parser.add_argument("--limit", type=int)
    batch_parser.add_argument("--force", action="store_true")

    combine_parser = commands.add_parser("combine")
    combine_parser.add_argument("adapter", choices=sorted(ADAPTERS))
    combine_parser.add_argument("--bootstrap-draws", type=int, default=4000)

    args = parser.parse_args()
    if args.command == "audit-split":
        payload = {
            "calibration": calibration_initializations().to_dict(
                orient="records"
            ),
            "evaluation": evaluation_initializations().to_dict(
                orient="records"
            ),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                json.dumps(
                    {
                        "calibration_cases": len(payload["calibration"]),
                        "evaluation_cases": len(payload["evaluation"]),
                        "calibration_50_stop": (
                            CALIBRATION_50_STOP.isoformat()
                        ),
                        "evaluation_start": EVALUATION_START.isoformat(),
                    },
                    indent=2,
                )
            )
    elif args.command == "build-profiles":
        residual = build_frozen_residual_profiles(force=args.force)
        affine = build_q50_affine_profiles(force=args.force)
        print(
            json.dumps(
                {
                    "residual_profile_rows": len(residual),
                    "q50_affine_profile_rows": len(affine),
                },
                indent=2,
            )
        )
    elif args.command == "audit-adapter":
        print(json.dumps(audit_adapter(args.adapter), indent=2))
    elif args.command == "smoke":
        print(
            json.dumps(
                run_smoke(
                    args.adapter,
                    _parse_timestamp(args.timestamp),
                    lead_time=args.lead_time,
                    device=args.device,
                ),
                indent=2,
            )
        )
    elif args.command == "run-batch":
        run_adapter_batch(
            args.adapter,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            limit=args.limit,
            force=args.force,
        )
    elif args.command == "combine":
        print(
            json.dumps(
                combine_adapter(
                    args.adapter, bootstrap_draws=args.bootstrap_draws
                ),
                indent=2,
            )
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
