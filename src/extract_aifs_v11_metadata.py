"""Extract AIFS v1.1 normalization and N320 coordinates without downloading 1 GB.

The Anemoi checkpoint is a ZIP archive.  ``remotezip`` reads only its central
directory and the small metadata members using HTTP byte ranges; model weights
are never downloaded or unpickled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_URL = (
    "https://huggingface.co/ecmwf/aifs-single-1.1/resolve/main/"
    "aifs-single-mse-1.1.ckpt"
)
NORMALIZATION_JSON = (
    ROOT / "data" / "processed" / "aifs_v1.1_normalization.json"
)
N320_GRID = ROOT / "data" / "processed" / "aifs_v1.1_n320_grid.npz"


def walk_dataset_attributes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if (
            isinstance(value.get("variables"), list)
            and isinstance(value.get("statistics"), dict)
        ):
            yield value
        for child in value.values():
            yield from walk_dataset_attributes(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dataset_attributes(child)


def normalization_method(variable: str, config: dict[str, Any]) -> tuple[str, str]:
    remap = config.get("remap") or {}
    statistics_variable = str(remap.get(variable, variable))
    if variable in set(config.get("none") or []):
        return "none", statistics_variable
    if variable in set(config.get("max") or []):
        return "max", statistics_variable
    if variable in set(config.get("min-max") or []):
        return "min-max", statistics_variable
    if variable in set(config.get("std") or []):
        return "std", statistics_variable
    return str(config.get("default", "mean-std")), statistics_variable


def extract_statistics(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, float]] = {}
    source_labels: dict[str, str] = {}
    for source_index, attrs in enumerate(walk_dataset_attributes(metadata["dataset"])):
        variables = [str(variable) for variable in attrs["variables"]]
        statistics = attrs["statistics"]
        for statistic in ("mean", "stdev", "minimum", "maximum"):
            values = statistics.get(statistic)
            if values is None or len(values) != len(variables):
                raise ValueError(
                    f"Malformed {statistic} statistics in source {source_index}"
                )
        for index, variable in enumerate(variables):
            sources[variable] = {
                statistic: float(statistics[statistic][index])
                for statistic in ("mean", "stdev", "minimum", "maximum")
            }
            source_labels[variable] = f"checkpoint_dataset_{source_index}"

    config = metadata["config"]["data"]["normalizer"]
    result: dict[str, dict[str, Any]] = {}
    for variable, raw_statistics in sorted(sources.items()):
        method, statistics_variable = normalization_method(variable, config)
        selected = sources.get(statistics_variable, raw_statistics)
        if method in {"mean-std", "std"}:
            scale = selected["stdev"]
            offset = selected["mean"] if method == "mean-std" else 0.0
        elif method == "max":
            scale = selected["maximum"]
            offset = 0.0
        elif method == "min-max":
            scale = selected["maximum"] - selected["minimum"]
            offset = selected["minimum"]
        elif method == "none":
            scale = 1.0
            offset = 0.0
        else:
            raise ValueError(f"Unsupported normalization method {method!r}")
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid normalization scale for {variable}: {scale}")
        result[variable] = {
            **raw_statistics,
            "method": method,
            "statistics_variable": statistics_variable,
            "normalization_offset": float(offset),
            "normalization_scale": float(scale),
            "source": source_labels[variable],
        }
    return result


def _member_name(remote: RemoteZip, suffix: str) -> str:
    matches = [item.filename for item in remote.infolist() if item.filename.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint member ending {suffix!r}: {matches}")
    return matches[0]


def extract_checkpoint_metadata(
    checkpoint_url: str = CHECKPOINT_URL,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with RemoteZip(checkpoint_url) as remote:
        metadata_name = _member_name(remote, "anemoi-metadata/ai-models.json")
        metadata = json.loads(remote.read(metadata_name))
        arrays = {}
        for name in ("latitudes", "longitudes"):
            descriptor = metadata["supporting_arrays_paths"][name]
            member_name = str(descriptor["path"])
            raw = remote.read(member_name)
            array = np.frombuffer(raw, dtype=np.dtype(descriptor["dtype"]))
            arrays[name] = array.reshape(tuple(descriptor["shape"])).copy()
        latitudes = arrays["latitudes"]
        longitudes = arrays["longitudes"]
    return metadata, latitudes, longitudes


def build_outputs(
    checkpoint_url: str = CHECKPOINT_URL,
    normalization_path: Path = NORMALIZATION_JSON,
    grid_path: Path = N320_GRID,
) -> dict[str, Any]:
    metadata, latitudes, longitudes = extract_checkpoint_metadata(checkpoint_url)
    if latitudes.shape != longitudes.shape or latitudes.size != 542_080:
        raise ValueError(
            f"Unexpected N320 coordinates: lat={latitudes.shape}, lon={longitudes.shape}"
        )
    statistics = extract_statistics(metadata)
    required = {
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
        *(
            f"{parameter}_{level}"
            for parameter in ("z", "t", "u", "v", "w", "q")
            for level in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
        ),
    }
    missing = required - set(statistics)
    if missing:
        raise ValueError(f"Checkpoint normalization is missing AIFS inputs: {sorted(missing)}")
    payload = {
        "checkpoint_url": checkpoint_url,
        "checkpoint_metadata_version": metadata.get("version"),
        "checkpoint_run_id": metadata.get("run_id"),
        "checkpoint_uuid": metadata.get("uuid"),
        "training_statistics_note": (
            "Extracted from the checkpoint's embedded Anemoi dataset metadata; "
            "model weights were not downloaded or unpickled."
        ),
        "variables": statistics,
    }
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    normalization_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        grid_path,
        latitude=np.asarray(latitudes, dtype=np.float64),
        longitude=np.mod(np.asarray(longitudes, dtype=np.float64), 360.0),
    )
    return {
        "normalization_path": str(normalization_path.relative_to(ROOT)),
        "grid_path": str(grid_path.relative_to(ROOT)),
        "variable_count": len(statistics),
        "grid_point_count": int(latitudes.size),
        "checkpoint_run_id": metadata.get("run_id"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-url", default=CHECKPOINT_URL)
    parser.add_argument("--normalization-output", type=Path, default=NORMALIZATION_JSON)
    parser.add_argument("--grid-output", type=Path, default=N320_GRID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_outputs(
                args.checkpoint_url,
                args.normalization_output,
                args.grid_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
