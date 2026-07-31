"""Run AIFS Single v1.1 forecasts and verify them against ARCO ERA5/ERA5T.

The module is deliberately split into restartable commands.  Preprocessed
inputs, downloaded reference slices and per-initialization metric parts are
content-addressable experiment artifacts; a failed long run can therefore
resume without repeating completed work.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import gc
import io
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import earthkit.regrid as ekr

from analyze_open_data_inputs import (
    StateReader,
    common_surface_categories,
    configure_project_cache,
    reduced_gaussian_weights,
    transformed_field,
    validate_regrid_order,
)
from extract_aifs_v11_metadata import N320_GRID
from open_data_inputs import (
    COHORTS,
    DYNAMIC_FIELD_KEYS,
    ROOT,
    STATIC_FIELD_KEYS,
    FieldKey,
    build_plan,
    timestamp_id,
)


configure_project_cache()

MODEL_REPOSITORY = "ecmwf/aifs-single-1.1"
MODEL_REVISION = "049b9ab1ccac3382b6332870ae550fd20a432faf"
MODEL_FILENAME = "aifs-single-mse-1.1.ckpt"
MODEL_DIR = ROOT / "models" / "aifs-single-1.1"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

EXPERIMENT_ROOT = ROOT / "data" / "forecast_experiment"
INPUT_CACHE = EXPERIMENT_ROOT / "inputs_n320"
REFERENCE_CACHE = EXPERIMENT_ROOT / "era5_n320"
FORECAST_PARTS = EXPERIMENT_ROOT / "forecast_metric_parts"
Q_ERROR_FIELDS = EXPERIMENT_ROOT / "q_error_fields"
LOGS = ROOT / "logs"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

ARCO_ROOT = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/raw"
)
ARCO_METADATA_URL = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs"
)

PRESSURE_SOURCE_NAMES = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}
SURFACE_SOURCE_NAMES = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2d": "2m_dewpoint_temperature",
    "2t": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "skt": "skin_temperature",
    "sp": "surface_pressure",
    "tcw": "total_column_water",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "hcc": "high_cloud_cover",
    "mcc": "medium_cloud_cover",
    "lcc": "low_cloud_cover",
    "tcc": "total_cloud_cover",
}

PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
PRESSURE_VARIABLES = tuple(
    f"{parameter}_{level}"
    for parameter in ("z", "t", "u", "v", "w", "q")
    for level in PRESSURE_LEVELS
)
SURFACE_VARIABLES = (
    "10u",
    "10v",
    "2d",
    "2t",
    "msl",
    "skt",
    "sp",
    "tcw",
    "stl1",
    "stl2",
    "swvl1",
    "swvl2",
    "100u",
    "100v",
    "hcc",
    "mcc",
    "lcc",
    "tcc",
)
EVALUATION_VARIABLES = PRESSURE_VARIABLES + SURFACE_VARIABLES
HIGHLIGHT_VARIABLES = ("q_50", "q_100")
LEADS = tuple(range(24, 241, 24))

REGIONS = (
    "global",
    "tropics",
    "northern_extratropics",
    "southern_extratropics",
    "global_land",
    "global_ocean",
)

EXPECTED_UNITS = {
    **{variable: "m2 s-2" for variable in PRESSURE_VARIABLES if variable.startswith("z_")},
    **{variable: "K" for variable in PRESSURE_VARIABLES if variable.startswith("t_")},
    **{
        variable: "m s-1"
        for variable in PRESSURE_VARIABLES
        if variable.startswith(("u_", "v_"))
    },
    **{variable: "Pa s-1" for variable in PRESSURE_VARIABLES if variable.startswith("w_")},
    **{variable: "kg kg-1" for variable in PRESSURE_VARIABLES if variable.startswith("q_")},
    "10u": "m s-1",
    "10v": "m s-1",
    "2d": "K",
    "2t": "K",
    "msl": "Pa",
    "skt": "K",
    "sp": "Pa",
    "tcw": "kg m-2",
    "stl1": "K",
    "stl2": "K",
    "swvl1": "m3 m-3",
    "swvl2": "m3 m-3",
    "100u": "m s-1",
    "100v": "m s-1",
    "hcc": "1",
    "mcc": "1",
    "lcc": "1",
    "tcc": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, *, compressed: bool = False, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        writer = np.savez_compressed if compressed else np.savez
        writer(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _state_paths(timestamp: datetime) -> tuple[Path, Path]:
    directory = (
        ROOT
        / "data"
        / "raw"
        / "ecmwf_open_data"
        / "states"
        / timestamp_id(timestamp)
    )
    return directory / "dynamic.grib2", directory / "dynamic.index.jsonl"


def _static_reader(cycle: str) -> StateReader:
    directory = ROOT / "data" / "raw" / "ecmwf_open_data" / "static" / cycle
    return StateReader(directory / "static.grib2", directory / "static.index.jsonl")


def cache_input_state(timestamp: datetime, cycle: str, *, force: bool = False) -> Path:
    """Interpolate one retained IFS state to N320 and cache float32 fields."""
    target = INPUT_CACHE / f"{timestamp_id(timestamp)}.npz"
    if target.exists() and not force:
        return target
    grib_path, index_path = _state_paths(timestamp)
    if not grib_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Missing initialization state {timestamp_id(timestamp)}")
    reader = StateReader(grib_path, index_path)
    arrays: dict[str, Any] = {}
    units: dict[str, str] = {}
    for field in DYNAMIC_FIELD_KEYS:
        values, field_units = transformed_field(reader, field)
        arrays[field.aifs_name] = values.astype(np.float32)
        units[field.aifs_name] = field_units
    arrays["_timestamp"] = timestamp.isoformat()
    arrays["_cycle"] = cycle
    arrays["_units_json"] = json.dumps(units, sort_keys=True)
    atomic_savez(target, compressed=True, **arrays)
    return target


def cache_static_state(cycle: str, *, force: bool = False) -> Path:
    target = INPUT_CACHE / f"static_{cycle}.npz"
    if target.exists() and not force:
        return target
    reader = _static_reader(cycle)
    arrays: dict[str, Any] = {}
    units: dict[str, str] = {}
    for field in STATIC_FIELD_KEYS:
        values, field_units = transformed_field(reader, field)
        arrays[field.aifs_name] = values.astype(np.float32)
        units[field.aifs_name] = field_units
    arrays["_cycle"] = cycle
    arrays["_units_json"] = json.dumps(units, sort_keys=True)
    atomic_savez(target, compressed=True, **arrays)
    return target


def cache_all_inputs(
    *,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    force: bool = False,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    plan = build_plan().sort_values("timestamp")
    plan = plan.iloc[
        [position % shard_count == shard_index for position in range(len(plan))]
    ]
    if limit is not None:
        plan = plan.iloc[:limit]
    for cycle in ("49r1", "50r1"):
        cache_static_state(cycle, force=force)
    total = len(plan)
    for position, row in enumerate(plan.itertuples(index=False), start=1):
        timestamp = datetime.fromisoformat(str(row.timestamp))
        started = time.monotonic()
        path = cache_input_state(timestamp, str(row.cycle), force=force)
        print(
            json.dumps(
                {
                    "event": "input_cached",
                    "position": position,
                    "total": total,
                    "timestamp": timestamp.isoformat(),
                    "path": str(path),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )


def input_state(timestamp: datetime, cycle: str) -> dict[str, Any]:
    """Build the exact two-time AIFS input dictionary from cached N320 states."""
    lag = timestamp - timedelta(hours=6)
    lag_path = cache_input_state(lag, cycle)
    now_path = cache_input_state(timestamp, cycle)
    static_path = cache_static_state(cycle)
    fields: dict[str, np.ndarray] = {}
    with np.load(lag_path) as lag_data, np.load(now_path) as now_data:
        for field in DYNAMIC_FIELD_KEYS:
            fields[field.aifs_name] = np.stack(
                [lag_data[field.aifs_name], now_data[field.aifs_name]]
            )
    with np.load(static_path) as static_data:
        for field in STATIC_FIELD_KEYS:
            values = np.asarray(static_data[field.aifs_name], dtype=np.float32)
            fields[field.aifs_name] = np.stack([values, values])
    return {"date": timestamp, "fields": fields}


def cohort_initializations() -> pd.DataFrame:
    memberships: dict[datetime, list[str]] = defaultdict(list)
    cycles: dict[datetime, str] = {}
    for cohort in COHORTS:
        for timestamp in cohort.pair_times:
            memberships[timestamp].append(f"{cohort.comparison}:{cohort.cohort}")
            cycles[timestamp] = cohort.cycle
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp.isoformat(),
                "timestamp_id": timestamp_id(timestamp),
                "cycle": cycles[timestamp],
                "memberships": ";".join(sorted(labels)),
            }
            for timestamp, labels in sorted(memberships.items())
        ]
    )


def required_valid_times() -> tuple[datetime, ...]:
    valid = {
        timestamp + timedelta(hours=lead)
        for cohort in COHORTS
        for timestamp in cohort.pair_times
        for lead in LEADS
    }
    return tuple(sorted(valid))


def reference_path(timestamp: datetime) -> Path:
    return REFERENCE_CACHE / f"{timestamp_id(timestamp)}.npz"


def _arco_url(variable: str, timestamp: datetime) -> str:
    date_path = timestamp.strftime("%Y/%m/%d")
    if "_" in variable and variable.rsplit("_", 1)[1].isdigit():
        parameter, level = variable.rsplit("_", 1)
        source_name = PRESSURE_SOURCE_NAMES[parameter]
        return (
            f"{ARCO_ROOT}/date-variable-pressure_level/{date_path}/"
            f"{source_name}/{level}.nc"
        )
    source_name = SURFACE_SOURCE_NAMES[variable]
    return (
        f"{ARCO_ROOT}/date-variable-single_level/{date_path}/"
        f"{source_name}/surface.nc"
    )


@dataclass(frozen=True)
class NetcdfVariableDescriptor:
    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    attributes: dict[str, Any]
    dtype: str
    itemsize: int
    begin: int
    vsize: int


class _HeaderOnlyNetcdf:
    """Parse a NetCDF3 header with scipy without reading any data arrays."""

    def __init__(self, header: bytes):
        from scipy.io import _netcdf

        class Parser(_netcdf.netcdf_file):
            def _read_var_array(inner_self) -> None:
                marker = inner_self.fp.read(4)
                if marker not in [_netcdf.ZERO, _netcdf.NC_VARIABLE]:
                    raise ValueError("Unexpected NetCDF variable header")
                count = inner_self._unpack_int()
                inner_self.descriptors = {}
                for _ in range(count):
                    (
                        name,
                        dimensions,
                        shape,
                        attributes,
                        typecode,
                        size,
                        dtype,
                        begin,
                        vsize,
                    ) = inner_self._read_var()
                    inner_self.descriptors[name] = NetcdfVariableDescriptor(
                        name=name,
                        dimensions=dimensions,
                        shape=tuple(int(value) for value in shape),
                        attributes=attributes,
                        dtype=dtype,
                        itemsize=size,
                        begin=begin,
                        vsize=vsize,
                    )

        parser = Parser(io.BytesIO(header), mode="r", mmap=False)
        try:
            self.dimensions = dict(parser.dimensions)
            self.attributes = dict(parser._attributes)
            self.variables = dict(parser.descriptors)
        finally:
            parser.close()


def _range_get(
    url: str, start: int, end: int, *, attempts: int = 8
) -> tuple[bytes, dict[str, str]]:
    if end <= start:
        raise ValueError(f"Invalid byte range {start}:{end}")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end - 1}"},
                timeout=120,
            )
            if response.status_code not in {200, 206}:
                response.raise_for_status()
            payload = response.content
            if response.status_code == 200 and len(payload) >= end:
                payload = payload[start:end]
            if len(payload) != end - start:
                raise IOError(
                    f"Range size mismatch for {url}: "
                    f"{len(payload)} != {end-start}"
                )
            return payload, {
                "etag": response.headers.get("ETag", ""),
                "generation": response.headers.get("x-goog-generation", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "total_bytes": response.headers.get(
                    "x-goog-stored-content-length", ""
                ),
            }
        except (requests.RequestException, IOError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(min(20.0, 1.5 * 2**attempt))
    raise RuntimeError(f"Failed byte range {url} {start}:{end}: {last_error}")


@lru_cache(maxsize=8192)
def _netcdf_header(url: str) -> tuple[_HeaderOnlyNetcdf, dict[str, str]]:
    # ARCO headers are under 64 KiB. A larger range leaves room for additions.
    payload, source = _range_get(url, 0, 131_072)
    return _HeaderOnlyNetcdf(payload), source


def _data_descriptor(header: _HeaderOnlyNetcdf) -> NetcdfVariableDescriptor:
    candidates = [
        descriptor
        for descriptor in header.variables.values()
        if descriptor.dimensions == ("time", "latitude", "longitude")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one gridded data variable, found {candidates}")
    return candidates[0]


def _decode_units(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def canonical_units(units: str) -> str:
    value = units.lower().replace("**", "").replace("^", "")
    value = value.replace("metres", "m").replace("metre", "m")
    value = value.replace("pascals", "pa").replace("pascal", "pa")
    value = value.replace("kelvin", "k")
    value = value.replace("dimensionless", "1").replace("(0 - 1)", "1")
    value = " ".join(value.replace("/", " ").split())
    replacements = {
        "m2 s-2": "m2 s-2",
        "m s-1": "m s-1",
        "pa s-1": "pa s-1",
        "kg kg-1": "kg kg-1",
        "kg m-2": "kg m-2",
        "m3 m-3": "m3 m-3",
        "k": "k",
        "pa": "pa",
        "1": "1",
        "~": "1",
    }
    return replacements.get(value, value)


def read_arco_regular_field(
    variable: str, timestamp: datetime
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Read and CF-decode one timestamp with two HTTP byte ranges."""
    url = _arco_url(variable, timestamp)
    header, source = _netcdf_header(url)
    descriptor = _data_descriptor(header)
    if descriptor.shape != (24, 721, 1440):
        raise ValueError(f"Unexpected ARCO shape for {url}: {descriptor.shape}")
    elements = 721 * 1440
    slice_bytes = elements * descriptor.itemsize
    start = descriptor.begin + timestamp.hour * slice_bytes
    raw, slice_source = _range_get(url, start, start + slice_bytes)
    packed = np.frombuffer(raw, dtype=np.dtype(descriptor.dtype)).reshape(721, 1440)
    attributes = descriptor.attributes
    fill = attributes.get("_FillValue", attributes.get("missing_value"))
    values = packed.astype(np.float64)
    if fill is not None:
        values[packed == fill] = np.nan
    values = (
        values * float(attributes.get("scale_factor", 1.0))
        + float(attributes.get("add_offset", 0.0))
    )
    units = _decode_units(attributes.get("units", ""))
    expected = EXPECTED_UNITS[variable]
    if canonical_units(units) != canonical_units(expected):
        raise ValueError(
            f"ERA5 unit mismatch for {variable}: {units!r} versus {expected!r}"
        )
    return values, units, {
        "url": url,
        "source_variable": descriptor.name,
        "units": units,
        "byte_start": int(start),
        "byte_end_exclusive": int(start + slice_bytes),
        **source,
        **slice_source,
    }


def regrid_era5_to_n320(values: np.ndarray) -> np.ndarray:
    """Interpolate an ARCO [90..-90, 0..360) regular field to N320."""
    if values.shape != (721, 1440):
        raise ValueError(f"Unexpected ERA5 field shape {values.shape}")
    finite = np.isfinite(values)
    if not finite.all():
        # If a source field contains missing cells, interpolate the numerator
        # and a validity mask, then retain cells with substantial support.
        numerator = np.where(finite, values, 0.0)
        interpolated = np.asarray(
            ekr.interpolate(
                numerator,
                {"grid": (0.25, 0.25)},
                {"grid": "N320"},
            ),
            dtype=np.float64,
        ).reshape(-1)
        support = np.asarray(
            ekr.interpolate(
                finite.astype(np.float64),
                {"grid": (0.25, 0.25)},
                {"grid": "N320"},
            ),
            dtype=np.float64,
        ).reshape(-1)
        result = np.full(interpolated.shape, np.nan, dtype=np.float64)
        valid = support >= 0.999
        result[valid] = interpolated[valid] / support[valid]
    else:
        result = np.asarray(
            ekr.interpolate(
                values,
                {"grid": (0.25, 0.25)},
                {"grid": "N320"},
            ),
            dtype=np.float64,
        ).reshape(-1)
    if result.size != 542_080:
        raise ValueError(f"Unexpected ERA5 N320 size {result.shape}")
    return result


def read_arco_n320_field(
    variable: str, timestamp: datetime
) -> tuple[str, np.ndarray, str, dict[str, Any]]:
    values, units, source = read_arco_regular_field(variable, timestamp)
    return variable, regrid_era5_to_n320(values).astype(np.float32), units, source


def cache_reference_state(
    timestamp: datetime,
    *,
    variables: Iterable[str] = EVALUATION_VARIABLES,
    workers: int = 8,
    force: bool = False,
) -> Path:
    variables = tuple(variables)
    target = reference_path(timestamp)
    if (
        target.exists()
        and not force
        and set(variables) == set(EVALUATION_VARIABLES)
    ):
        return target
    arrays: dict[str, Any] = {}
    units: dict[str, str] = {}
    sources: dict[str, Any] = {}
    if workers <= 1:
        results = [read_arco_n320_field(variable, timestamp) for variable in variables]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(read_arco_n320_field, variable, timestamp): variable
                for variable in variables
            }
            for future in as_completed(futures):
                results.append(future.result())
    for variable, values, field_units, source in results:
        arrays[variable] = values
        units[variable] = field_units
        sources[variable] = source
    arrays["_timestamp"] = timestamp.isoformat()
    arrays["_reference_product"] = "ERA5" if timestamp.year == 2025 else "ERA5T"
    arrays["_units_json"] = json.dumps(units, sort_keys=True)
    arrays["_sources_json"] = json.dumps(sources, sort_keys=True)
    complete = set(variables) == set(EVALUATION_VARIABLES)
    if complete:
        atomic_savez(target, compressed=False, **arrays)
    else:
        target = (
            REFERENCE_CACHE
            / "partial"
            / f"{timestamp_id(timestamp)}_{'_'.join(variables)}.npz"
        )
        atomic_savez(target, compressed=False, **arrays)
    return target


def cache_all_references(
    *,
    limit: int | None = None,
    workers: int = 8,
    shard_index: int = 0,
    shard_count: int = 1,
    force: bool = False,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    timestamps = required_valid_times()
    timestamps = tuple(
        timestamp
        for position, timestamp in enumerate(timestamps)
        if position % shard_count == shard_index
    )
    if limit is not None:
        timestamps = timestamps[:limit]
    for position, timestamp in enumerate(timestamps, start=1):
        started = time.monotonic()
        path = cache_reference_state(timestamp, workers=workers, force=force)
        print(
            json.dumps(
                {
                    "event": "reference_cached",
                    "position": position,
                    "total": len(timestamps),
                    "timestamp": timestamp.isoformat(),
                    "path": str(path),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )


@lru_cache(maxsize=1)
def evaluation_geometry() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]
]:
    grid = np.load(N320_GRID)
    latitudes = np.asarray(grid["latitude"], dtype=np.float64)
    longitudes = np.asarray(grid["longitude"], dtype=np.float64)
    weights = reduced_gaussian_weights(latitudes)
    surface_category, _ = common_surface_categories()
    masks = {
        "global": np.ones(latitudes.size, dtype=bool),
        "tropics": np.abs(latitudes) <= 20.0,
        "northern_extratropics": latitudes > 20.0,
        "southern_extratropics": latitudes < -20.0,
        "global_land": surface_category == 0,
        "global_ocean": surface_category == 1,
    }
    return latitudes, longitudes, weights, masks


def score_field(
    forecast: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    forecast = np.asarray(forecast, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if forecast.size != weights.size or reference.size != weights.size:
        raise ValueError(
            f"Score shape mismatch: forecast={forecast.shape}, "
            f"reference={reference.shape}, weights={weights.shape}"
        )
    finite = np.isfinite(forecast) & np.isfinite(reference)
    error = forecast - reference
    rows = []
    for region, region_mask in masks.items():
        valid = finite & region_mask
        selected_weights = weights[valid]
        weight_sum = float(selected_weights.sum())
        region_weight_sum = float(weights[region_mask].sum())
        if weight_sum <= 0:
            bias = float("nan")
            mse = float("nan")
            rmse = float("nan")
            weighted_error_sum = 0.0
            weighted_squared_error_sum = 0.0
        else:
            selected_error = error[valid]
            weighted_error_sum = float(
                np.dot(selected_weights, selected_error)
            )
            weighted_squared_error_sum = float(
                np.dot(selected_weights, selected_error * selected_error)
            )
            bias = weighted_error_sum / weight_sum
            mse = weighted_squared_error_sum / weight_sum
            rmse = float(np.sqrt(max(mse, 0.0)))
        rows.append(
            {
                "region": region,
                "bias": bias,
                "mean_squared_error": mse,
                "rmse": rmse,
                "weighted_error_sum": weighted_error_sum,
                "weighted_squared_error_sum": weighted_squared_error_sum,
                "weight_sum": weight_sum,
                "region_weight_sum": region_weight_sum,
                "valid_area_fraction": (
                    weight_sum / region_weight_sum
                    if region_weight_sum > 0
                    else float("nan")
                ),
                "valid_gridpoint_count": int(valid.sum()),
                "region_gridpoint_count": int(region_mask.sum()),
            }
        )
    return rows


def score_field_matrix(
    forecasts: np.ndarray,
    references: np.ndarray,
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[list[dict[str, Any]]]:
    """Vectorized area scores for fields × points.

    Matrix multiplication avoids hundreds of small masked dot products at each
    lead and materially reduces the CPU/I/O stall between GPU forecast steps.
    """
    forecasts = np.asarray(forecasts, dtype=np.float32)
    references = np.asarray(references, dtype=np.float32)
    if forecasts.shape != references.shape or forecasts.ndim != 2:
        raise ValueError(
            f"Matrix score shape mismatch {forecasts.shape} vs {references.shape}"
        )
    if forecasts.shape[1] != weights.size:
        raise ValueError(
            f"Matrix point mismatch {forecasts.shape[1]} vs {weights.size}"
        )
    region_names = tuple(masks)
    region_weights = np.column_stack(
        [np.where(masks[region], weights, 0.0) for region in region_names]
    )
    finite = np.isfinite(forecasts) & np.isfinite(references)
    finite_float = finite.astype(np.float64)
    weight_sums = finite_float @ region_weights
    del finite_float

    errors = np.where(finite, forecasts - references, 0.0).astype(np.float64)
    weighted_error_sums = errors @ region_weights
    np.square(errors, out=errors)
    weighted_squared_error_sums = errors @ region_weights
    del errors

    region_weight_sums = region_weights.sum(axis=0)
    region_point_counts = np.array(
        [int(masks[region].sum()) for region in region_names], dtype=np.int64
    )
    valid_counts = np.column_stack(
        [finite[:, masks[region]].sum(axis=1) for region in region_names]
    )
    result: list[list[dict[str, Any]]] = []
    for variable_index in range(forecasts.shape[0]):
        rows = []
        for region_index, region in enumerate(region_names):
            weight_sum = float(weight_sums[variable_index, region_index])
            region_weight_sum = float(region_weight_sums[region_index])
            weighted_error_sum = float(
                weighted_error_sums[variable_index, region_index]
            )
            weighted_squared_error_sum = float(
                weighted_squared_error_sums[variable_index, region_index]
            )
            if weight_sum > 0:
                bias = weighted_error_sum / weight_sum
                mse = weighted_squared_error_sum / weight_sum
                rmse = float(np.sqrt(max(mse, 0.0)))
            else:
                bias = mse = rmse = float("nan")
            rows.append(
                {
                    "region": region,
                    "bias": bias,
                    "mean_squared_error": mse,
                    "rmse": rmse,
                    "weighted_error_sum": weighted_error_sum,
                    "weighted_squared_error_sum": weighted_squared_error_sum,
                    "weight_sum": weight_sum,
                    "region_weight_sum": region_weight_sum,
                    "valid_area_fraction": (
                        weight_sum / region_weight_sum
                        if region_weight_sum > 0
                        else float("nan")
                    ),
                    "valid_gridpoint_count": int(
                        valid_counts[variable_index, region_index]
                    ),
                    "region_gridpoint_count": int(
                        region_point_counts[region_index]
                    ),
                }
            )
        result.append(rows)
    return result


def _write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        raise ValueError(f"Unsupported frame output {path}")
    os.replace(temporary, path)


def forecast_part_path(timestamp: datetime) -> Path:
    return FORECAST_PARTS / f"{timestamp_id(timestamp)}.parquet"


def run_forecast_and_score(
    timestamp: datetime,
    cycle: str,
    memberships: str,
    *,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Run one 10-day forecast and write all per-case sufficient statistics."""
    from anemoi.inference.runners.simple import SimpleRunner

    target = forecast_part_path(timestamp)
    if target.exists() and not force:
        return target
    missing_reference = [
        reference_path(timestamp + timedelta(hours=lead))
        for lead in LEADS
        if not reference_path(timestamp + timedelta(hours=lead)).exists()
    ]
    if missing_reference:
        raise FileNotFoundError(
            f"Missing {len(missing_reference)} reference states; "
            f"first={missing_reference[0]}"
        )
    if not MODEL_PATH.exists():
        download_model()
    _, _, weights, masks = evaluation_geometry()
    runner = SimpleRunner(str(MODEL_PATH), device=device)
    rows: list[dict[str, Any]] = []
    retained_leads: list[int] = []
    available_variables: list[str] | None = None
    started = time.monotonic()
    for state in runner.run(input_state=input_state(timestamp, cycle), lead_time=240):
        lead = int((state["date"] - timestamp).total_seconds() // 3600)
        if lead not in LEADS:
            continue
        retained_leads.append(lead)
        current_variables = sorted(state["fields"])
        if available_variables is None:
            available_variables = current_variables
            missing = sorted(set(EVALUATION_VARIABLES) - set(current_variables))
            if missing:
                raise ValueError(f"AIFS output is missing variables: {missing}")
        elif current_variables != available_variables:
            raise ValueError("AIFS output variable inventory changed within forecast")
        valid_time = timestamp + timedelta(hours=lead)
        current_reference = reference_path(valid_time)
        with np.load(current_reference) as reference:
            units = json.loads(str(reference["_units_json"]))
            forecast_matrix = np.stack(
                [
                    np.asarray(state["fields"][variable], dtype=np.float32)
                    for variable in EVALUATION_VARIABLES
                ]
            )
            reference_matrix = np.stack(
                [
                    np.asarray(reference[variable], dtype=np.float32)
                    for variable in EVALUATION_VARIABLES
                ]
            )
            matrix_scores = score_field_matrix(
                forecast_matrix, reference_matrix, weights, masks
            )
            for variable_index, variable in enumerate(EVALUATION_VARIABLES):
                for metrics in matrix_scores[variable_index]:
                    rows.append(
                        {
                            "initialization": timestamp.isoformat(),
                            "initialization_id": timestamp_id(timestamp),
                            "cycle": cycle,
                            "memberships": memberships,
                            "lead_hours": lead,
                            "valid_time": valid_time.isoformat(),
                            "reference_product": (
                                "ERA5" if valid_time.year == 2025 else "ERA5T"
                            ),
                            "variable": variable,
                            "units": units[variable],
                            **metrics,
                        }
                    )
                if variable in HIGHLIGHT_VARIABLES:
                    forecast_values = forecast_matrix[variable_index]
                    reference_values = reference_matrix[variable_index]
                    q_target = (
                        Q_ERROR_FIELDS
                        / timestamp_id(timestamp)
                        / f"lead_{lead:03d}_{variable}.npz"
                    )
                    atomic_savez(
                        q_target,
                        compressed=True,
                        error=(forecast_values - reference_values).astype(np.float32),
                        forecast=forecast_values.astype(np.float32),
                        reference=reference_values.astype(np.float32),
                        initialization=timestamp.isoformat(),
                        valid_time=valid_time.isoformat(),
                        lead_hours=lead,
                        variable=variable,
                        units=units[variable],
                    )
        print(
            json.dumps(
                {
                    "event": "forecast_lead_scored",
                    "initialization": timestamp.isoformat(),
                    "cycle": cycle,
                    "lead_hours": lead,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )
    if tuple(retained_leads) != LEADS:
        raise RuntimeError(f"Incomplete forecast leads: {retained_leads}")
    frame = pd.DataFrame(rows)
    expected_rows = len(LEADS) * len(EVALUATION_VARIABLES) * len(REGIONS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"Metric row mismatch {len(frame)} != {expected_rows}")
    _write_frame_atomic(frame, target)
    metadata = {
        "initialization": timestamp.isoformat(),
        "cycle": cycle,
        "memberships": memberships,
        "device": device,
        "model_sha256": sha256(MODEL_PATH),
        "retained_leads": retained_leads,
        "available_variables": available_variables,
        "evaluated_variables": list(EVALUATION_VARIABLES),
        "metric_rows": len(frame),
        "elapsed_seconds": time.monotonic() - started,
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return target


def run_forecast_batch(
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    device: str = "cuda",
    limit: int | None = None,
    force: bool = False,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    frame = cohort_initializations()
    frame = frame.iloc[
        [position % shard_count == shard_index for position in range(len(frame))]
    ]
    if limit is not None:
        frame = frame.iloc[:limit]
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        timestamp = datetime.fromisoformat(str(row.timestamp))
        started = time.monotonic()
        try:
            path = run_forecast_and_score(
                timestamp,
                str(row.cycle),
                str(row.memberships),
                device=device,
                force=force,
            )
        finally:
            # SimpleRunner owns a large CUDA graph/model and some of its
            # objects participate in reference cycles.  Force collection
            # between independent initializations so a new runner does not
            # encounter allocator fragmentation from the preceding one.
            gc.collect()
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "forecast_complete",
                    "position": position,
                    "total": len(frame),
                    "initialization": timestamp.isoformat(),
                    "path": str(path),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )


def _bootstrap_interval(
    values: np.ndarray,
    *,
    transform: str,
    seed_parts: tuple[Any, ...],
    draws: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    seed = int.from_bytes(
        hashlib.sha256(repr(seed_parts).encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    if transform == "sqrt":
        means = np.sqrt(np.maximum(means, 0.0))
    elif transform != "identity":
        raise ValueError(transform)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def build_reference_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp in required_valid_times():
        path = reference_path(timestamp)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            product = str(data["_reference_product"])
            units = json.loads(str(data["_units_json"]))
            sources = json.loads(str(data["_sources_json"]))
        for variable in EVALUATION_VARIABLES:
            rows.append(
                {
                    "valid_time": timestamp.isoformat(),
                    "valid_time_id": timestamp_id(timestamp),
                    "reference_product": product,
                    "variable": variable,
                    "decoded_units": units[variable],
                    "cache_path": str(path.relative_to(ROOT)),
                    **sources[variable],
                }
            )
    frame = pd.DataFrame(rows)
    _write_frame_atomic(
        frame, PROCESSED / "aifs_v11_era5_reference_manifest.parquet"
    )
    _write_frame_atomic(frame, PROCESSED / "aifs_v11_era5_reference_manifest.csv")
    return frame


def combine_forecast_parts(*, bootstrap_draws: int = 4000) -> dict[str, Any]:
    expected = cohort_initializations()
    paths = [
        forecast_part_path(datetime.fromisoformat(timestamp))
        for timestamp in expected["timestamp"]
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} forecast metric parts; first={missing[0]}"
        )
    cases = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    _write_frame_atomic(cases, PROCESSED / "aifs_v11_forecast_case_metrics.parquet")
    _write_frame_atomic(cases, PROCESSED / "aifs_v11_forecast_case_metrics.csv")

    expanded = []
    for comparison in ("same_season", "cutover"):
        for cohort in ("49r1", "50r1"):
            label = f"{comparison}:{cohort}"
            selected = cases[cases["memberships"].str.contains(label, regex=False)].copy()
            selected["comparison"] = comparison
            selected["cohort"] = cohort
            expanded.append(selected)
    membership_cases = pd.concat(expanded, ignore_index=True)

    score_rows: list[dict[str, Any]] = []
    keys = ["comparison", "cohort", "variable", "units", "region", "lead_hours"]
    for key, group in membership_cases.groupby(keys, sort=True, dropna=False):
        comparison, cohort, variable, units, region, lead = key
        bias_values = group["bias"].to_numpy(dtype=float)
        mse_values = group["mean_squared_error"].to_numpy(dtype=float)
        finite_bias = bias_values[np.isfinite(bias_values)]
        finite_mse = mse_values[np.isfinite(mse_values)]
        bias = (
            float(np.mean(finite_bias)) if finite_bias.size else float("nan")
        )
        rmse = (
            float(np.sqrt(max(float(np.mean(finite_mse)), 0.0)))
            if finite_mse.size
            else float("nan")
        )
        bias_low, bias_high = _bootstrap_interval(
            bias_values,
            transform="identity",
            seed_parts=(*key, "bias"),
            draws=bootstrap_draws,
        )
        rmse_low, rmse_high = _bootstrap_interval(
            mse_values,
            transform="sqrt",
            seed_parts=(*key, "rmse"),
            draws=bootstrap_draws,
        )
        score_rows.append(
            {
                "comparison": comparison,
                "cohort": cohort,
                "variable": variable,
                "units": units,
                "region": region,
                "lead_hours": int(lead),
                "bias": bias,
                "bias_ci95_low": bias_low,
                "bias_ci95_high": bias_high,
                "rmse": rmse,
                "rmse_ci95_low": rmse_low,
                "rmse_ci95_high": rmse_high,
                "forecast_case_count": int(group["initialization_id"].nunique()),
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
    scores = pd.DataFrame(score_rows)
    _write_frame_atomic(scores, PROCESSED / "aifs_v11_forecast_cohort_scores.parquet")
    _write_frame_atomic(scores, PROCESSED / "aifs_v11_forecast_cohort_scores.csv")

    comparison_rows: list[dict[str, Any]] = []
    group_keys = ["comparison", "variable", "units", "region", "lead_hours"]
    for key, group in membership_cases.groupby(group_keys, sort=True, dropna=False):
        comparison, variable, units, region, lead = key
        by_cohort = {
            cohort: cohort_group
            for cohort, cohort_group in group.groupby("cohort", sort=False)
        }
        group49 = by_cohort["49r1"]
        group50 = by_cohort["50r1"]
        bias49 = group49["bias"].to_numpy(dtype=float)
        bias50 = group50["bias"].to_numpy(dtype=float)
        mse49 = group49["mean_squared_error"].to_numpy(dtype=float)
        mse50 = group50["mean_squared_error"].to_numpy(dtype=float)
        bias49 = bias49[np.isfinite(bias49)]
        bias50 = bias50[np.isfinite(bias50)]
        mse49 = mse49[np.isfinite(mse49)]
        mse50 = mse50[np.isfinite(mse50)]
        complete = all(
            values.size > 0 for values in (bias49, bias50, mse49, mse50)
        )
        if complete:
            rmse49 = float(np.sqrt(np.mean(mse49)))
            rmse50 = float(np.sqrt(np.mean(mse50)))
            bias_difference = float(np.mean(bias50) - np.mean(bias49))
            relative_rmse = 100.0 * (rmse50 / rmse49 - 1.0)
        else:
            rmse49 = rmse50 = bias_difference = relative_rmse = float("nan")
        seed = int.from_bytes(
            hashlib.sha256(repr(key).encode()).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        if complete:
            index_mse49 = rng.integers(
                0, len(mse49), size=(bootstrap_draws, len(mse49))
            )
            index_mse50 = rng.integers(
                0, len(mse50), size=(bootstrap_draws, len(mse50))
            )
            index_bias49 = rng.integers(
                0, len(bias49), size=(bootstrap_draws, len(bias49))
            )
            index_bias50 = rng.integers(
                0, len(bias50), size=(bootstrap_draws, len(bias50))
            )
            bootstrap_rmse49 = np.sqrt(
                np.mean(mse49[index_mse49], axis=1)
            )
            bootstrap_rmse50 = np.sqrt(
                np.mean(mse50[index_mse50], axis=1)
            )
            bootstrap_relative = 100.0 * (
                bootstrap_rmse50 / bootstrap_rmse49 - 1.0
            )
            bootstrap_bias_difference = (
                np.mean(bias50[index_bias50], axis=1)
                - np.mean(bias49[index_bias49], axis=1)
            )
            rmse_ci = np.quantile(bootstrap_relative, (0.025, 0.975))
            bias_ci = np.quantile(bootstrap_bias_difference, (0.025, 0.975))
        else:
            rmse_ci = bias_ci = np.array([np.nan, np.nan])
        comparison_rows.append(
            {
                "comparison": comparison,
                "variable": variable,
                "units": units,
                "region": region,
                "lead_hours": int(lead),
                "rmse_49r1": rmse49,
                "rmse_50r1": rmse50,
                "relative_rmse_change_percent": relative_rmse,
                "relative_rmse_change_ci95_low": float(rmse_ci[0]),
                "relative_rmse_change_ci95_high": float(rmse_ci[1]),
                "relative_rmse_change_significant_95": bool(
                    rmse_ci[0] > 0 or rmse_ci[1] < 0
                ),
                "bias_49r1": (
                    float(np.mean(bias49)) if bias49.size else float("nan")
                ),
                "bias_50r1": (
                    float(np.mean(bias50)) if bias50.size else float("nan")
                ),
                "bias_difference_50r1_minus_49r1": bias_difference,
                "bias_difference_ci95_low": float(bias_ci[0]),
                "bias_difference_ci95_high": float(bias_ci[1]),
                "bias_difference_significant_95": bool(
                    bias_ci[0] > 0 or bias_ci[1] < 0
                ),
                "forecast_case_count_49r1": int(
                    group49["initialization_id"].nunique()
                ),
                "forecast_case_count_50r1": int(
                    group50["initialization_id"].nunique()
                ),
                "reference_products_49r1": ";".join(
                    sorted(group49["reference_product"].unique())
                ),
                "reference_products_50r1": ";".join(
                    sorted(group50["reference_product"].unique())
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    _write_frame_atomic(
        comparisons, PROCESSED / "aifs_v11_forecast_cycle_comparisons.parquet"
    )
    _write_frame_atomic(
        comparisons, PROCESSED / "aifs_v11_forecast_cycle_comparisons.csv"
    )
    reference_manifest = build_reference_manifest()
    reference_metadata_response = requests.get(ARCO_METADATA_URL, timeout=60)
    reference_metadata_response.raise_for_status()
    payload = {
        "forecast_parts": len(paths),
        "case_metric_rows": len(cases),
        "cohort_score_rows": len(scores),
        "comparison_rows": len(comparisons),
        "reference_manifest_rows": len(reference_manifest),
        "bootstrap_draws": bootstrap_draws,
        "model": json.loads(
            (MODEL_DIR / "manifest.json").read_text(encoding="utf-8")
        ),
        "reference_archive_metadata": reference_metadata_response.json(),
        "excluded_accumulated_variable_mismatches": [
            "cp",
            "tp",
            "ro",
            "sf",
            "ssrd",
            "strd",
        ],
        "storage_policy": (
            "Per-case sufficient statistics for 96 instantaneous fields; full "
            "forecast/reference/error arrays retained for q_50 and q_100."
        ),
        "comparisons": {
            comparison: {
                "cohort_score_rows": int(
                    (scores["comparison"] == comparison).sum()
                ),
                "comparison_rows": int(
                    (comparisons["comparison"] == comparison).sum()
                ),
            }
            for comparison in ("same_season", "cutover")
        },
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "aifs_v11_forecast_experiment_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def download_model(*, force: bool = False) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if force and MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if not MODEL_PATH.exists():
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=MODEL_FILENAME,
                revision=MODEL_REVISION,
                local_dir=MODEL_DIR,
            )
        )
        if downloaded.resolve() != MODEL_PATH.resolve():
            raise RuntimeError(f"Unexpected model path {downloaded}")
    payload = {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "filename": MODEL_FILENAME,
        "bytes": MODEL_PATH.stat().st_size,
        "sha256": sha256(MODEL_PATH),
        "path": str(MODEL_PATH),
    }
    manifest = MODEL_DIR / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_smoke_forecast(
    timestamp: datetime,
    cycle: str,
    *,
    lead_time: int = 24,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one short forecast and retain a compact numerical smoke artifact."""
    from anemoi.inference.runners.simple import SimpleRunner

    if not MODEL_PATH.exists():
        download_model()
    runner = SimpleRunner(str(MODEL_PATH), device=device)
    started = time.monotonic()
    states = []
    for state in runner.run(input_state=input_state(timestamp, cycle), lead_time=lead_time):
        state_lead = int((state["date"] - timestamp).total_seconds() // 3600)
        if state_lead == lead_time:
            states.append(state)
    if len(states) != 1:
        raise RuntimeError(f"Expected one retained lead {lead_time}, found {len(states)}")
    state = states[0]
    missing = sorted(set(EVALUATION_VARIABLES) - set(state["fields"]))
    values = {
        variable: np.asarray(state["fields"][variable], dtype=np.float32)
        for variable in HIGHLIGHT_VARIABLES
        if variable in state["fields"]
    }
    target = EXPERIMENT_ROOT / "smoke" / (
        f"{timestamp_id(timestamp)}_lead{lead_time:03d}.npz"
    )
    atomic_savez(
        target,
        compressed=True,
        **values,
        _timestamp=timestamp.isoformat(),
        _valid_time=state["date"].isoformat(),
        _available_variables_json=json.dumps(sorted(state["fields"])),
    )
    payload = {
        "timestamp": timestamp.isoformat(),
        "cycle": cycle,
        "lead_hours": lead_time,
        "valid_time": state["date"].isoformat(),
        "available_variable_count": len(state["fields"]),
        "available_variables": sorted(state["fields"]),
        "missing_evaluation_variables": missing,
        "elapsed_seconds": time.monotonic() - started,
        "artifact": str(target),
        "highlight_statistics": {
            variable: {
                "minimum": float(np.nanmin(array)),
                "mean": float(np.nanmean(array)),
                "maximum": float(np.nanmax(array)),
                "finite": int(np.isfinite(array).sum()),
            }
            for variable, array in values.items()
        },
    }
    target.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_smoke_end_to_end(
    timestamp: datetime,
    cycle: str,
    *,
    lead_time: int = 24,
    device: str = "cuda",
) -> dict[str, Any]:
    """Forecast, retrieve q50/q100 ERA5, and score one end-to-end case."""
    forecast_payload = run_smoke_forecast(
        timestamp,
        cycle,
        lead_time=lead_time,
        device=device,
    )
    valid_time = timestamp + timedelta(hours=lead_time)
    reference_file = cache_reference_state(
        valid_time,
        variables=HIGHLIGHT_VARIABLES,
        workers=2,
    )
    smoke_file = Path(forecast_payload["artifact"])
    _, _, weights, masks = evaluation_geometry()
    scores: list[dict[str, Any]] = []
    with np.load(smoke_file) as forecast, np.load(reference_file) as reference:
        units = json.loads(str(reference["_units_json"]))
        for variable in HIGHLIGHT_VARIABLES:
            field_scores = score_field(
                forecast[variable],
                reference[variable],
                weights,
                masks,
            )
            for row in field_scores:
                scores.append(
                    {
                        "variable": variable,
                        "units": units[variable],
                        **row,
                    }
                )
    payload = {
        "status": "passed",
        "forecast": forecast_payload,
        "reference_artifact": str(reference_file),
        "reference_product": "ERA5" if valid_time.year == 2025 else "ERA5T",
        "score_rows": scores,
    }
    target = EXPERIMENT_ROOT / "smoke" / (
        f"{timestamp_id(timestamp)}_lead{lead_time:03d}_e2e.json"
    )
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(target)
    return payload


def _parse_timestamp(value: str) -> datetime:
    if value.endswith("Z") and "T" in value and "-" not in value:
        return datetime.strptime(value, "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def audit() -> dict[str, Any]:
    plan = build_plan()
    missing = []
    for row in plan.itertuples(index=False):
        timestamp = datetime.fromisoformat(str(row.timestamp))
        for path in _state_paths(timestamp):
            if not path.exists():
                missing.append(str(path))
    for cycle in ("49r1", "50r1"):
        reader = _static_reader(cycle)
        for path in (reader.grib_path,):
            if not path.exists():
                missing.append(str(path))
    grid = np.load(N320_GRID)
    latitudes = grid["latitude"]
    longitudes = grid["longitude"]
    regrid_check = validate_regrid_order(latitudes, longitudes)
    weights = reduced_gaussian_weights(latitudes)
    surface_categories, surface_summary = common_surface_categories()
    response = requests.get(ARCO_METADATA_URL, timeout=60)
    response.raise_for_status()
    reference_metadata = response.json()
    first_lags = {
        f"{cohort.comparison}:{cohort.cohort}": (
            cohort.start - timedelta(hours=6)
        ).isoformat()
        for cohort in COHORTS
    }
    return {
        "plan_rows": len(plan),
        "unique_forecasts": len(cohort_initializations()),
        "cases_per_cohort": {
            f"{cohort.comparison}:{cohort.cohort}": len(cohort.pair_times)
            for cohort in COHORTS
        },
        "first_lag_states": first_lags,
        "missing_input_paths": missing,
        "evaluation_variable_count": len(EVALUATION_VARIABLES),
        "evaluation_variables": EVALUATION_VARIABLES,
        "excluded_accumulated_variables": ["cp", "tp", "ro", "sf", "ssrd", "strd"],
        "n320_points": int(latitudes.size),
        "area_weight_sum": float(weights.sum()),
        "surface_categories": {
            **surface_summary,
            "array_points": int(surface_categories.size),
        },
        "regrid_check": regrid_check,
        "reference_metadata": reference_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")

    model = commands.add_parser("download-model")
    model.add_argument("--force", action="store_true")

    cache = commands.add_parser("cache-inputs")
    cache.add_argument("--limit", type=int)
    cache.add_argument("--shard-index", type=int, default=0)
    cache.add_argument("--shard-count", type=int, default=1)
    cache.add_argument("--force", action="store_true")

    reference = commands.add_parser("cache-references")
    reference.add_argument("--limit", type=int)
    reference.add_argument("--workers", type=int, default=8)
    reference.add_argument("--shard-index", type=int, default=0)
    reference.add_argument("--shard-count", type=int, default=1)
    reference.add_argument("--force", action="store_true")

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--timestamp", default="20260513T0000Z")
    smoke.add_argument("--cycle", default="50r1", choices=("49r1", "50r1"))
    smoke.add_argument("--lead-time", type=int, default=24)
    smoke.add_argument("--device", default="cuda")

    smoke_e2e = commands.add_parser("smoke-e2e")
    smoke_e2e.add_argument("--timestamp", default="20260513T0000Z")
    smoke_e2e.add_argument("--cycle", default="50r1", choices=("49r1", "50r1"))
    smoke_e2e.add_argument("--lead-time", type=int, default=24)
    smoke_e2e.add_argument("--device", default="cuda")

    run = commands.add_parser("run-batch")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--device", default="cuda")
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true")

    combine = commands.add_parser("combine")
    combine.add_argument("--bootstrap-draws", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.command == "download-model":
        print(json.dumps(download_model(force=args.force), indent=2))
    elif args.command == "cache-inputs":
        cache_all_inputs(
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            force=args.force,
        )
    elif args.command == "cache-references":
        cache_all_references(
            limit=args.limit,
            workers=args.workers,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            force=args.force,
        )
    elif args.command == "smoke":
        print(
            json.dumps(
                run_smoke_forecast(
                    _parse_timestamp(args.timestamp),
                    args.cycle,
                    lead_time=args.lead_time,
                    device=args.device,
                ),
                indent=2,
            )
        )
    elif args.command == "smoke-e2e":
        print(
            json.dumps(
                run_smoke_end_to_end(
                    _parse_timestamp(args.timestamp),
                    args.cycle,
                    lead_time=args.lead_time,
                    device=args.device,
                ),
                indent=2,
            )
        )
    elif args.command == "run-batch":
        run_forecast_batch(
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            limit=args.limit,
            force=args.force,
        )
    elif args.command == "combine":
        print(
            json.dumps(
                combine_forecast_parts(bootstrap_draws=args.bootstrap_draws),
                indent=2,
            )
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
