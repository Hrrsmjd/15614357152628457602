"""Compare unpaired 49r1/50r1 public AIFS input-proxy cohorts.

This is deliberately a distribution comparison, not a paired cycle-effect
estimate.  It reproduces the public AIFS v1.1 notebook preprocessing:

* 0.25-degree IFS control forecast, step 0
* longitude roll followed by interpolation to N320
* ``gh * 9.80665`` to obtain geopotential
* checkpoint normalization, including soil-variable renaming
* two-state diagnostics for both x(t0) and x(t0)-x(t-6h)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import earthkit.regrid as ekr
from earthkit.regrid.utils import caching as regrid_caching
import eccodes
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from extract_aifs_v11_metadata import N320_GRID, NORMALIZATION_JSON
from open_data_inputs import (
    COHORTS,
    DYNAMIC_FIELD_KEYS,
    STATIC_FIELD_KEYS,
    ROOT,
    SIX_HOURS,
    FieldKey,
    build_plan,
)
from variable_order import add_display_metadata


PROCESSED = ROOT / "data" / "processed"
PARTS = PROCESSED / "open_data_distribution_parts"
MAPS = PROCESSED / "open_data_distribution_maps"
COHORT_STATS_CSV = PROCESSED / "open_data_cohort_statistics.csv"
COHORT_STATS_PARQUET = PROCESSED / "open_data_cohort_statistics.parquet"
COMPARISONS_CSV = PROCESSED / "open_data_cycle_comparisons.csv"
COMPARISONS_PARQUET = PROCESSED / "open_data_cycle_comparisons.parquet"
ANALYSIS_SUMMARY = ROOT / "outputs" / "open_data_distribution_summary.json"
EARTHKIT_CACHE = ROOT / "data" / "cache" / "earthkit-regrid"
STATIC_COMPARISON_CSV = PROCESSED / "open_data_static_comparison.csv"
STATIC_COMPARISON_PARQUET = PROCESSED / "open_data_static_comparison.parquet"

REGIONS = ("global", "tropics", "extratropics")
SURFACES = ("all", "land", "ocean")
UTC_GROUPS: tuple[int | str, ...] = ("all", 0, 6, 12, 18)
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
DEFAULT_MAP_FIELDS = {
    "2t",
    "skt",
    "msl",
    "stl1",
    "swvl1",
    "t_850",
    "q_850",
    "z_500",
}


def configure_project_cache() -> None:
    """Keep downloaded interpolation matrices inside the project directory."""
    EARTHKIT_CACHE.mkdir(parents=True, exist_ok=True)
    regrid_caching.SETTINGS["cache-policy"] = "user"
    regrid_caching.SETTINGS["user-cache-directory"] = str(EARTHKIT_CACHE)
    regrid_caching.CACHE._settings_changed()


configure_project_cache()


def safe_field_name(field_name: str) -> str:
    return field_name.replace("/", "_").replace(" ", "_")


def cohort_key(comparison: str, cohort: str) -> str:
    return f"{comparison}:{cohort}"


def pair_memberships() -> dict[datetime, tuple[str, ...]]:
    result: dict[datetime, list[str]] = defaultdict(list)
    for cohort in COHORTS:
        key = cohort_key(cohort.comparison, cohort.cohort)
        for timestamp in cohort.pair_times:
            result[timestamp].append(key)
    return {timestamp: tuple(keys) for timestamp, keys in result.items()}


def _read_selected_index(path: Path) -> dict[str, dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lookup = {str(record["_aifs_name"]): record for record in records}
    if len(lookup) != len(records):
        raise ValueError(f"Duplicate AIFS names in {path}")
    return lookup


class StateReader:
    def __init__(self, grib_path: Path, index_path: Path):
        self.grib_path = grib_path
        self.index = _read_selected_index(index_path)

    def read(self, aifs_name: str) -> tuple[np.ndarray, dict[str, Any]]:
        record = self.index[aifs_name]
        with self.grib_path.open("rb") as stream:
            stream.seek(int(record["_local_offset"]))
            message = stream.read(int(record["_length"]))
        if len(message) != int(record["_length"]):
            raise IOError(f"Truncated local GRIB message {self.grib_path}:{aifs_name}")
        handle = eccodes.codes_new_from_message(message)
        try:
            values = np.asarray(eccodes.codes_get_values(handle), dtype=np.float64)
            metadata = {
                "shortName": eccodes.codes_get(handle, "shortName"),
                "level": int(eccodes.codes_get(handle, "level")),
                "units": eccodes.codes_get(handle, "units"),
                "Ni": int(eccodes.codes_get(handle, "Ni")),
                "Nj": int(eccodes.codes_get(handle, "Nj")),
                "gridType": eccodes.codes_get(handle, "gridType"),
                "numberOfMissing": int(eccodes.codes_get(handle, "numberOfMissing")),
            }
        finally:
            eccodes.codes_release(handle)
        return values, metadata


def regrid_to_n320(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    if (
        metadata["gridType"] != "regular_ll"
        or metadata["Ni"] != 1440
        or metadata["Nj"] != 721
        or values.size != 1_038_240
    ):
        raise ValueError(f"Unexpected open-data grid: {metadata}")
    if metadata["numberOfMissing"] != 0 or not np.all(np.isfinite(values)):
        raise ValueError("Missing or non-finite values before N320 interpolation")
    regular = values.reshape(721, 1440)
    regular = np.roll(regular, -regular.shape[1] // 2, axis=1)
    n320 = np.asarray(
        ekr.interpolate(
            regular,
            {"grid": (0.25, 0.25)},
            {"grid": "N320"},
        ),
        dtype=np.float64,
    ).reshape(-1)
    if n320.size != 542_080 or not np.all(np.isfinite(n320)):
        raise ValueError(f"Invalid interpolated N320 field: {n320.shape}")
    return n320


def transformed_field(
    reader: StateReader, field: FieldKey
) -> tuple[np.ndarray, str]:
    values, metadata = reader.read(field.aifs_name)
    values = regrid_to_n320(values, metadata)
    units = str(metadata["units"])
    if field.levtype == "pl" and field.parameter == "gh":
        values = values * 9.80665
        units = "m2 s-2"
    return values, units


def reduced_gaussian_weights(latitudes: np.ndarray) -> np.ndarray:
    """Exact Gaussian-latitude weights divided among points in each row."""
    latitudes = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    rounded = np.round(latitudes, 10)
    unique, inverse, counts = np.unique(
        rounded, return_inverse=True, return_counts=True
    )
    if unique.size != 640:
        raise ValueError(f"N320 should contain 640 latitude rows, found {unique.size}")
    nodes, gaussian_weights = np.polynomial.legendre.leggauss(unique.size)
    gaussian_latitudes = np.degrees(np.arcsin(nodes))
    if not np.allclose(unique, gaussian_latitudes, atol=1e-6):
        raise ValueError("Checkpoint latitudes do not match an N320 Gaussian grid")
    per_row = gaussian_weights / counts
    weights = per_row[inverse]
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Invalid N320 area weights")
    return weights


def validate_regrid_order(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> dict[str, float]:
    """Prove the interpolation output follows checkpoint N320 point ordering."""
    source_latitude = np.broadcast_to(
        np.linspace(90.0, -90.0, 721)[:, None], (721, 1440)
    )
    interpolated_latitude = np.asarray(
        ekr.interpolate(
            source_latitude,
            {"grid": (0.25, 0.25)},
            {"grid": "N320"},
        )
    )
    latitude_error = np.abs(interpolated_latitude - latitudes)
    source_longitude = np.arange(1440, dtype=np.float64) * 0.25
    source_cosine = np.broadcast_to(
        np.cos(np.deg2rad(source_longitude)), (721, 1440)
    )
    interpolated_cosine = np.asarray(
        ekr.interpolate(
            source_cosine,
            {"grid": (0.25, 0.25)},
            {"grid": "N320"},
        )
    )
    longitude_error = np.abs(
        interpolated_cosine - np.cos(np.deg2rad(longitudes))
    )
    latitude_max = float(np.max(latitude_error))
    longitude_q9999 = float(np.quantile(longitude_error, 0.9999))
    if latitude_max > 1e-3 or longitude_q9999 > 1e-3:
        raise ValueError(
            "N320 output ordering does not match checkpoint coordinates: "
            f"latitude_max={latitude_max}, longitude_q9999={longitude_q9999}"
        )
    return {
        "synthetic_latitude_max_abs_error_degrees": latitude_max,
        "synthetic_longitude_cosine_q99_99_abs_error": longitude_q9999,
    }


def _static_reader(cycle: str) -> StateReader:
    directory = ROOT / "data" / "raw" / "ecmwf_open_data" / "static" / cycle
    return StateReader(directory / "static.grib2", directory / "static.index.jsonl")


def common_surface_categories() -> tuple[np.ndarray, dict[str, Any]]:
    masks = {}
    for cycle in ("49r1", "50r1"):
        values, _ = transformed_field(_static_reader(cycle), FieldKey("sfc", "lsm"))
        masks[cycle] = values
    land = (masks["49r1"] >= 0.5) & (masks["50r1"] >= 0.5)
    ocean = (masks["49r1"] < 0.5) & (masks["50r1"] < 0.5)
    changed = ~(land | ocean)
    category = np.full(land.size, 2, dtype=np.int8)
    category[land] = 0
    category[ocean] = 1
    return category, {
        "common_land_points": int(land.sum()),
        "common_ocean_points": int(ocean.sum()),
        "changed_or_ambiguous_points": int(changed.sum()),
        "mask_policy": (
            "Land/ocean results use the intersection of the two cycle masks; "
            "changed/ambiguous coastal points appear only in surface=all."
        ),
    }


def spatial_cells(
    latitudes: np.ndarray, surface_category: np.ndarray
) -> np.ndarray:
    latitude_category = (np.abs(latitudes) > 20.0).astype(np.int8)
    return latitude_category * 3 + surface_category


def cell_indices(region: str, surface: str) -> tuple[int, ...]:
    latitude_categories = {
        "global": (0, 1),
        "tropics": (0,),
        "extratropics": (1,),
    }[region]
    surface_categories = {
        "all": (0, 1, 2),
        "land": (0,),
        "ocean": (1,),
    }[surface]
    return tuple(
        latitude_category * 3 + surface_category
        for latitude_category in latitude_categories
        for surface_category in surface_categories
    )


@dataclass
class BinnedMoments:
    weight: np.ndarray
    weighted_sum: np.ndarray
    weighted_square_sum: np.ndarray
    count: np.ndarray

    @classmethod
    def empty(cls) -> "BinnedMoments":
        return cls(*(np.zeros(6, dtype=np.float64) for _ in range(4)))

    def update(
        self,
        values: np.ndarray,
        area_weights: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        valid = np.isfinite(values)
        cell = cells[valid]
        weights = area_weights[valid]
        data = values[valid]
        self.weight += np.bincount(cell, weights=weights, minlength=6)
        self.weighted_sum += np.bincount(
            cell, weights=weights * data, minlength=6
        )
        self.weighted_square_sum += np.bincount(
            cell, weights=weights * data * data, minlength=6
        )
        self.count += np.bincount(cell, minlength=6)

    def __add__(self, other: "BinnedMoments") -> "BinnedMoments":
        return BinnedMoments(
            self.weight + other.weight,
            self.weighted_sum + other.weighted_sum,
            self.weighted_square_sum + other.weighted_square_sum,
            self.count + other.count,
        )

    def summarize(self, indices: tuple[int, ...]) -> dict[str, float]:
        selected = np.asarray(indices, dtype=int)
        weight = float(self.weight[selected].sum())
        if weight <= 0:
            return {
                "mean": float("nan"),
                "standard_deviation": float("nan"),
                "variance": float("nan"),
                "weight_sum": 0.0,
                "point_observations": 0,
            }
        mean = float(self.weighted_sum[selected].sum() / weight)
        second = float(self.weighted_square_sum[selected].sum() / weight)
        variance = max(second - mean * mean, 0.0)
        return {
            "mean": mean,
            "standard_deviation": float(np.sqrt(variance)),
            "variance": variance,
            "weight_sum": weight,
            "point_observations": int(self.count[selected].sum()),
        }


def combined_moments(
    moments: dict[tuple[str, int], BinnedMoments],
    kind: str,
    cohort: str,
    utc: int | str,
) -> BinnedMoments:
    if utc != "all":
        return moments[(f"{kind}:{cohort}", int(utc))]
    result = BinnedMoments.empty()
    for hour in (0, 6, 12, 18):
        result = result + moments[(f"{kind}:{cohort}", hour)]
    return result


def _quantile_columns(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {f"q{int(q*100):02d}": float("nan") for q in QUANTILES}
    quantiles = np.quantile(values, QUANTILES)
    return {
        f"q{int(q*100):02d}": float(value)
        for q, value in zip(QUANTILES, quantiles, strict=True)
    }


def combined_samples(
    samples: dict[tuple[str, int], list[np.ndarray]],
    kind: str,
    cohort: str,
    utc: int | str,
) -> np.ndarray:
    hours = (0, 6, 12, 18) if utc == "all" else (int(utc),)
    arrays = [
        value
        for hour in hours
        for value in samples.get((f"{kind}:{cohort}", hour), [])
    ]
    if not arrays:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(arrays)


def sample_group(
    matrix: np.ndarray,
    sample_cells: np.ndarray,
    region: str,
    surface: str,
) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.float64)
    mask = np.isin(sample_cells, cell_indices(region, surface))
    return matrix[:, mask].reshape(-1)


def map_accumulator() -> dict[str, Any]:
    return {"count": 0, "sum": None, "square_sum": None}


def update_map(accumulator: dict[str, Any], values: np.ndarray) -> None:
    if accumulator["sum"] is None:
        accumulator["sum"] = np.zeros_like(values)
        accumulator["square_sum"] = np.zeros_like(values)
    accumulator["sum"] += values
    accumulator["square_sum"] += values * values
    accumulator["count"] += 1


def finish_map(accumulator: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    count = int(accumulator["count"])
    if count <= 0:
        raise ValueError("Empty map accumulator")
    mean = accumulator["sum"] / count
    variance = np.maximum(accumulator["square_sum"] / count - mean * mean, 0.0)
    return mean, variance


def analyze_static_fields(
    normalization: dict[str, Any],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    area_weights: np.ndarray,
    cells: np.ndarray,
    sample_indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    MAPS.mkdir(parents=True, exist_ok=True)
    sample_cells = cells[sample_indices]
    for field in STATIC_FIELD_KEYS:
        values49, units49 = transformed_field(_static_reader("49r1"), field)
        values50, units50 = transformed_field(_static_reader("50r1"), field)
        if units49 != units50:
            raise ValueError(
                f"Static units changed for {field.aifs_name}: {units49} vs {units50}"
            )
        _, scale, method = _normalization(normalization, field.aifs_name)
        moments49 = BinnedMoments.empty()
        moments50 = BinnedMoments.empty()
        moments49.update(values49, area_weights, cells)
        moments50.update(values50, area_weights, cells)
        sampled49 = values49[sample_indices]
        sampled50 = values50[sample_indices]
        for region in REGIONS:
            for surface in SURFACES:
                indices = cell_indices(region, surface)
                summary49 = moments49.summarize(indices)
                summary50 = moments50.summarize(indices)
                mask = np.isin(sample_cells, indices)
                sample49 = sampled49[mask]
                sample50 = sampled50[mask]
                variance49 = summary49["variance"]
                variance50 = summary50["variance"]
                rows.append(
                    {
                        "variable": field.aifs_name,
                        "units": units49,
                        "region": region,
                        "surface": surface,
                        "mean_49r1": summary49["mean"],
                        "mean_50r1": summary50["mean"],
                        "mean_shift_50r1_minus_49r1": (
                            summary50["mean"] - summary49["mean"]
                        ),
                        "normalized_mean_shift": (
                            summary50["mean"] - summary49["mean"]
                        )
                        / scale,
                        "std_49r1": summary49["standard_deviation"],
                        "std_50r1": summary50["standard_deviation"],
                        "variance_ratio_50r1_over_49r1": (
                            variance50 / variance49
                            if variance49 > 0
                            else float("nan")
                        ),
                        "wasserstein_distance": float(
                            wasserstein_distance(sample49, sample50)
                        ),
                        "normalized_wasserstein_distance": float(
                            wasserstein_distance(sample49 / scale, sample50 / scale)
                        ),
                        "normalization_method": method,
                        "normalization_scale": scale,
                        "sample_size": int(sample49.size),
                    }
                )
        np.savez_compressed(
            MAPS / f"static_{safe_field_name(field.aifs_name)}.npz",
            latitude=latitudes,
            longitude=longitudes,
            value_49r1=values49,
            value_50r1=values50,
            shift_50r1_minus_49r1=values50 - values49,
            normalized_shift=(values50 - values49) / scale,
            variable=field.aifs_name,
            units=units49,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(STATIC_COMPARISON_CSV, index=False)
    frame.to_parquet(STATIC_COMPARISON_PARQUET, index=False)
    return frame


def _normalization(
    normalization: dict[str, Any], field_name: str
) -> tuple[float, float, str]:
    record = normalization["variables"][field_name]
    return (
        float(record["normalization_offset"]),
        float(record["normalization_scale"]),
        str(record["method"]),
    )


def _state_paths(row: pd.Series) -> tuple[Path, Path]:
    return ROOT / str(row["grib_path"]), ROOT / str(row["selected_index_path"])


def process_field(
    field: FieldKey,
    plan: pd.DataFrame,
    normalization: dict[str, Any],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    area_weights: np.ndarray,
    cells: np.ndarray,
    sample_indices: np.ndarray,
    *,
    make_map: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    field_name = field.aifs_name
    offset, scale, normalization_method = _normalization(normalization, field_name)
    memberships = pair_memberships()
    moments: dict[tuple[str, int], BinnedMoments] = defaultdict(BinnedMoments.empty)
    samples: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    maps = {
        cohort_key(item.comparison, item.cohort): map_accumulator()
        for item in COHORTS
    }
    sample_cells = cells[sample_indices]
    previous_time: datetime | None = None
    previous_values: np.ndarray | None = None
    units: str | None = None

    for _, row in plan.sort_values("timestamp").iterrows():
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
        grib_path, index_path = _state_paths(row)
        if not grib_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"Missing {field_name} state at {timestamp}: run the downloader first"
            )
        values, current_units = transformed_field(
            StateReader(grib_path, index_path), field
        )
        if units is None:
            units = current_units
        elif units != current_units:
            raise ValueError(
                f"Units changed for {field_name}: {units!r} to {current_units!r}"
            )
        current_memberships = memberships.get(timestamp, ())
        for current_cohort in current_memberships:
            moments[(f"state:{current_cohort}", timestamp.hour)].update(
                values, area_weights, cells
            )
            samples[(f"state:{current_cohort}", timestamp.hour)].append(
                values[sample_indices]
            )
            if make_map:
                update_map(maps[current_cohort], values)

        if previous_time is not None and timestamp - previous_time == SIX_HOURS:
            assert previous_values is not None
            tendency = values - previous_values
            for current_cohort in current_memberships:
                moments[(f"tendency:{current_cohort}", timestamp.hour)].update(
                    tendency, area_weights, cells
                )
                samples[(f"tendency:{current_cohort}", timestamp.hour)].append(
                    tendency[sample_indices]
                )
        elif current_memberships:
            raise ValueError(
                f"Missing t-6 state for pair {timestamp}; previous={previous_time}"
            )
        previous_time = timestamp
        previous_values = values

    cohort_rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        current_cohort = cohort_key(cohort.comparison, cohort.cohort)
        for kind in ("state", "tendency"):
            normalized_offset = offset if kind == "state" else 0.0
            for utc in UTC_GROUPS:
                current_moments = combined_moments(
                    moments, kind, current_cohort, utc
                )
                sample_matrix = combined_samples(
                    samples, kind, current_cohort, utc
                )
                for region in REGIONS:
                    for surface in SURFACES:
                        summary = current_moments.summarize(
                            cell_indices(region, surface)
                        )
                        raw_sample = sample_group(
                            sample_matrix, sample_cells, region, surface
                        )
                        normalized_sample = (
                            raw_sample - normalized_offset
                        ) / scale
                        cohort_rows.append(
                            {
                                "comparison": cohort.comparison,
                                "cohort": cohort.cohort,
                                "cycle": cohort.cycle,
                                "kind": kind,
                                "variable": field_name,
                                "source_parameter": field.parameter,
                                "level": field.level,
                                "level_type": field.levtype,
                                "units": units,
                                "region": region,
                                "surface": surface,
                                "utc": str(utc).zfill(2) if utc != "all" else "all",
                                "normalization_method": normalization_method,
                                "normalization_offset": normalized_offset,
                                "normalization_scale": scale,
                                **summary,
                                "normalized_mean": (
                                    summary["mean"] - normalized_offset
                                )
                                / scale,
                                "normalized_standard_deviation": (
                                    summary["standard_deviation"] / abs(scale)
                                ),
                                "sample_size": int(raw_sample.size),
                                **_quantile_columns(raw_sample),
                                **{
                                    f"normalized_{key}": value
                                    for key, value in _quantile_columns(
                                        normalized_sample
                                    ).items()
                                },
                            }
                        )
    cohort_frame = pd.DataFrame(cohort_rows)

    comparison_rows: list[dict[str, Any]] = []
    for comparison in ("same_season", "cutover"):
        for kind in ("state", "tendency"):
            normalized_offset = offset if kind == "state" else 0.0
            for utc in UTC_GROUPS:
                sample49 = combined_samples(
                    samples, kind, cohort_key(comparison, "49r1"), utc
                )
                sample50 = combined_samples(
                    samples, kind, cohort_key(comparison, "50r1"), utc
                )
                for region in REGIONS:
                    for surface in SURFACES:
                        selection = (
                            (cohort_frame["comparison"] == comparison)
                            & (cohort_frame["kind"] == kind)
                            & (cohort_frame["utc"] == (
                                str(utc).zfill(2) if utc != "all" else "all"
                            ))
                            & (cohort_frame["region"] == region)
                            & (cohort_frame["surface"] == surface)
                        )
                        selected = cohort_frame[selection].set_index("cohort")
                        row49 = selected.loc["49r1"]
                        row50 = selected.loc["50r1"]
                        raw49 = sample_group(
                            sample49, sample_cells, region, surface
                        )
                        raw50 = sample_group(
                            sample50, sample_cells, region, surface
                        )
                        normalized49 = (raw49 - normalized_offset) / scale
                        normalized50 = (raw50 - normalized_offset) / scale
                        variance49 = float(row49["variance"])
                        variance50 = float(row50["variance"])
                        comparison_rows.append(
                            {
                                "comparison": comparison,
                                "kind": kind,
                                "variable": field_name,
                                "source_parameter": field.parameter,
                                "level": field.level,
                                "level_type": field.levtype,
                                "units": units,
                                "region": region,
                                "surface": surface,
                                "utc": (
                                    str(utc).zfill(2) if utc != "all" else "all"
                                ),
                                "mean_49r1": float(row49["mean"]),
                                "mean_50r1": float(row50["mean"]),
                                "mean_shift_50r1_minus_49r1": (
                                    float(row50["mean"]) - float(row49["mean"])
                                ),
                                "normalized_mean_shift": (
                                    float(row50["mean"]) - float(row49["mean"])
                                )
                                / scale,
                                "std_49r1": float(row49["standard_deviation"]),
                                "std_50r1": float(row50["standard_deviation"]),
                                "variance_ratio_50r1_over_49r1": (
                                    variance50 / variance49
                                    if variance49 > 0
                                    else float("nan")
                                ),
                                "wasserstein_distance": (
                                    float(wasserstein_distance(raw49, raw50))
                                    if raw49.size and raw50.size
                                    else float("nan")
                                ),
                                "normalized_wasserstein_distance": (
                                    float(
                                        wasserstein_distance(
                                            normalized49, normalized50
                                        )
                                    )
                                    if normalized49.size and normalized50.size
                                    else float("nan")
                                ),
                                "sample_size_49r1": int(raw49.size),
                                "sample_size_50r1": int(raw50.size),
                                "normalization_method": normalization_method,
                                "normalization_scale": scale,
                            }
                        )
    comparison_frame = pd.DataFrame(comparison_rows)

    if make_map:
        MAPS.mkdir(parents=True, exist_ok=True)
        for comparison in ("same_season", "cutover"):
            mean49, variance49 = finish_map(
                maps[cohort_key(comparison, "49r1")]
            )
            mean50, variance50 = finish_map(
                maps[cohort_key(comparison, "50r1")]
            )
            variance_ratio = np.divide(
                variance50,
                variance49,
                out=np.full_like(variance49, np.nan),
                where=variance49 > 0,
            )
            np.savez_compressed(
                MAPS / f"{comparison}_{safe_field_name(field_name)}.npz",
                latitude=latitudes,
                longitude=longitudes,
                mean_49r1=mean49,
                mean_50r1=mean50,
                mean_shift_50r1_minus_49r1=mean50 - mean49,
                normalized_mean_shift=(mean50 - mean49) / scale,
                variance_49r1=variance49,
                variance_50r1=variance50,
                variance_ratio_50r1_over_49r1=variance_ratio,
                variable=field_name,
                units=units,
                comparison=comparison,
            )
    return cohort_frame, comparison_frame


def write_part(
    field_name: str,
    cohort_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    safe = safe_field_name(field_name)
    cohort_frame.to_parquet(PARTS / f"{safe}.cohorts.parquet", index=False)
    comparison_frame.to_parquet(PARTS / f"{safe}.comparisons.parquet", index=False)


def combine_parts() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_paths = sorted(PARTS.glob("*.cohorts.parquet"))
    comparison_paths = sorted(PARTS.glob("*.comparisons.parquet"))
    if not cohort_paths or not comparison_paths:
        raise ValueError("No analysis parts found")
    cohorts = add_display_metadata(
        pd.concat(
            [pd.read_parquet(path) for path in cohort_paths],
            ignore_index=True,
        )
    )
    comparisons = add_display_metadata(
        pd.concat(
            [pd.read_parquet(path) for path in comparison_paths],
            ignore_index=True,
        )
    )
    dimension_orders = {
        "comparison": {"same_season": 0, "cutover": 1},
        "kind": {"state": 0, "tendency": 1},
        "region": {"global": 0, "tropics": 1, "extratropics": 2},
        "surface": {"all": 0, "land": 1, "ocean": 2},
        "utc": {"all": 0, "00": 1, "06": 2, "12": 3, "18": 4},
        "cohort": {"49r1": 0, "50r1": 1},
    }
    for dimension, order in dimension_orders.items():
        if dimension in cohorts:
            cohorts[f"_{dimension}_order"] = cohorts[dimension].map(order)
        if dimension in comparisons:
            comparisons[f"_{dimension}_order"] = comparisons[dimension].map(order)
    common_sort = [
        "_comparison_order",
        "_kind_order",
        "_region_order",
        "_surface_order",
        "_utc_order",
        "display_order",
    ]
    cohorts = cohorts.sort_values(
        [*common_sort, "_cohort_order"], kind="stable"
    ).reset_index(drop=True)
    comparisons = comparisons.sort_values(
        common_sort, kind="stable"
    ).reset_index(drop=True)
    cohorts = cohorts.drop(
        columns=[
            column
            for column in cohorts
            if column.startswith("_") and column.endswith("_order")
        ]
    )
    comparisons = comparisons.drop(
        columns=[
            column
            for column in comparisons
            if column.startswith("_") and column.endswith("_order")
        ]
    )
    cohorts.to_csv(COHORT_STATS_CSV, index=False)
    cohorts.to_parquet(COHORT_STATS_PARQUET, index=False)
    comparisons.to_csv(COMPARISONS_CSV, index=False)
    comparisons.to_parquet(COMPARISONS_PARQUET, index=False)
    return cohorts, comparisons


def available_fields(values: list[str] | None) -> list[FieldKey]:
    if not values:
        return list(DYNAMIC_FIELD_KEYS)
    requested = set(values)
    selected = [field for field in DYNAMIC_FIELD_KEYS if field.aifs_name in requested]
    missing = requested - {field.aifs_name for field in selected}
    if missing:
        raise ValueError(f"Unknown requested AIFS fields: {sorted(missing)}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION_JSON)
    parser.add_argument("--n320-grid", type=Path, default=N320_GRID)
    parser.add_argument("--field", action="append")
    parser.add_argument("--limit-fields", type=int)
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=49_050)
    parser.add_argument("--overwrite-parts", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--no-maps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.combine_only:
        cohorts, comparisons = combine_parts()
        print(
            json.dumps(
                {
                    "cohort_rows": len(cohorts),
                    "comparison_rows": len(comparisons),
                },
                indent=2,
            )
        )
        return
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    with np.load(args.n320_grid) as grid:
        latitudes = np.asarray(grid["latitude"], dtype=np.float64)
        longitudes = np.asarray(grid["longitude"], dtype=np.float64)
    area_weights = reduced_gaussian_weights(latitudes)
    grid_order_summary = validate_regrid_order(latitudes, longitudes)
    surface_category, surface_summary = common_surface_categories()
    cells = spatial_cells(latitudes, surface_category)
    rng = np.random.default_rng(args.seed)
    sample_indices = np.sort(
        rng.choice(
            latitudes.size,
            size=args.sample_points,
            replace=False,
            p=area_weights / area_weights.sum(),
        )
    )
    plan = build_plan()
    fields = available_fields(args.field)
    if args.limit_fields is not None:
        fields = fields[: args.limit_fields]
    completed = []
    skipped = []
    for index, field in enumerate(fields, start=1):
        safe = safe_field_name(field.aifs_name)
        cohort_part = PARTS / f"{safe}.cohorts.parquet"
        comparison_part = PARTS / f"{safe}.comparisons.parquet"
        if (
            not args.overwrite_parts
            and cohort_part.exists()
            and comparison_part.exists()
        ):
            skipped.append(field.aifs_name)
            print(f"[{index}/{len(fields)}] {field.aifs_name}: existing", flush=True)
            continue
        print(f"[{index}/{len(fields)}] {field.aifs_name}: processing", flush=True)
        cohort_frame, comparison_frame = process_field(
            field,
            plan,
            normalization,
            latitudes,
            longitudes,
            area_weights,
            cells,
            sample_indices,
            make_map=(not args.no_maps and field.aifs_name in DEFAULT_MAP_FIELDS),
        )
        write_part(field.aifs_name, cohort_frame, comparison_frame)
        completed.append(field.aifs_name)
    cohorts, comparisons = combine_parts()
    static_comparison = analyze_static_fields(
        normalization,
        latitudes,
        longitudes,
        area_weights,
        cells,
        sample_indices,
    )
    summary = {
        "analysis_type": "unpaired_public_step0_input_proxy_distribution_comparison",
        "cohort_rows": len(cohorts),
        "comparison_rows": len(comparisons),
        "static_comparison_rows": len(static_comparison),
        "fields_completed_this_run": completed,
        "fields_skipped_existing": skipped,
        "sample_points_per_state": args.sample_points,
        "normalization_checkpoint_run_id": normalization.get("checkpoint_run_id"),
        "grid_order_validation": grid_order_summary,
        "surface_mask": surface_summary,
        "limitations": [
            "Different dates/years mean weather, season and interannual variability remain confounders.",
            "Public step-0 forecasts on 0.25 degrees are not bit-identical to native O1280 operational analyses.",
            "Quantiles and Wasserstein distances use a deterministic area-weighted spatial sample.",
        ],
    }
    ANALYSIS_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_SUMMARY.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
