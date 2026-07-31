"""Plan, range-download, and audit the public AIFS input-proxy cohorts.

The public ECMWF files are IFS control-forecast fields at step zero on a
regular 0.25-degree grid.  They are the inputs used by ECMWF's public AIFS
inference notebook, but they are not bit-identical to the native operational
control analyses used inside ECMWF.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import eccodes
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "ecmwf_open_data"
PLAN_CSV = ROOT / "data" / "processed" / "open_data_cohort_plan.csv"
MANIFEST_CSV = ROOT / "data" / "processed" / "open_data_download_manifest.csv"
PLAN_SUMMARY = ROOT / "outputs" / "open_data_cohort_plan.json"

AWS_ROOT = "https://ecmwf-forecasts.s3.amazonaws.com"
IFS_50R1_CUTOVER = datetime(2026, 5, 12, 6, tzinfo=timezone.utc)
SIX_HOURS = timedelta(hours=6)

PRESSURE_PARAMETERS = ("gh", "t", "u", "v", "w", "q")
PRESSURE_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
SURFACE_PARAMETERS = ("10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw")
SOIL_PARAMETERS = ("vsw", "sot")
SOIL_LEVELS = (1, 2)
STATIC_PARAMETERS = ("lsm", "z", "slor", "sdor")


@dataclass(frozen=True)
class Cohort:
    comparison: str
    cohort: str
    cycle: str
    start: datetime
    end: datetime

    @property
    def pair_times(self) -> tuple[datetime, ...]:
        return tuple(six_hourly(self.start, self.end))

    @property
    def state_times(self) -> tuple[datetime, ...]:
        return (self.start - SIX_HOURS, *self.pair_times)


COHORTS = (
    Cohort(
        "same_season",
        "49r1",
        "49r1",
        datetime(2025, 5, 13, 0, tzinfo=timezone.utc),
        datetime(2025, 5, 20, 18, tzinfo=timezone.utc),
    ),
    Cohort(
        "same_season",
        "50r1",
        "50r1",
        datetime(2026, 5, 13, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 20, 18, tzinfo=timezone.utc),
    ),
    Cohort(
        "cutover",
        "49r1",
        "49r1",
        datetime(2026, 5, 4, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 11, 18, tzinfo=timezone.utc),
    ),
    Cohort(
        "cutover",
        "50r1",
        "50r1",
        datetime(2026, 5, 13, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 20, 18, tzinfo=timezone.utc),
    ),
)


@dataclass(frozen=True, order=True)
class FieldKey:
    levtype: str
    parameter: str
    level: int = 0

    @property
    def aifs_name(self) -> str:
        if self.levtype == "pl":
            parameter = "z" if self.parameter == "gh" else self.parameter
            return f"{parameter}_{self.level}"
        if self.levtype == "sol":
            prefix = {"vsw": "swvl", "sot": "stl"}[self.parameter]
            return f"{prefix}{self.level}"
        return self.parameter


def dynamic_field_keys() -> tuple[FieldKey, ...]:
    fields = [
        FieldKey("pl", parameter, level)
        for parameter in PRESSURE_PARAMETERS
        for level in PRESSURE_LEVELS
    ]
    fields.extend(FieldKey("sfc", parameter) for parameter in SURFACE_PARAMETERS)
    fields.extend(
        FieldKey("sol", parameter, level)
        for parameter in SOIL_PARAMETERS
        for level in SOIL_LEVELS
    )
    return tuple(fields)


def static_field_keys() -> tuple[FieldKey, ...]:
    return tuple(FieldKey("sfc", parameter) for parameter in STATIC_PARAMETERS)


DYNAMIC_FIELD_KEYS = dynamic_field_keys()
STATIC_FIELD_KEYS = static_field_keys()


def six_hourly(start: datetime, end: datetime) -> Iterable[datetime]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Cohort timestamps must be timezone-aware")
    if start > end:
        raise ValueError("Cohort start must not be after its end")
    current = start
    while current <= end:
        yield current
        current += SIX_HOURS


def operational_cycle(timestamp: datetime) -> str:
    return "50r1" if timestamp >= IFS_50R1_CUTOVER else "49r1"


def stream_for(timestamp: datetime, cycle: str) -> str:
    if cycle == "50r1":
        return "oper"
    if cycle != "49r1":
        raise ValueError(f"Unsupported cycle {cycle!r}")
    return "scda" if timestamp.hour in {6, 18} else "oper"


def source_urls(timestamp: datetime, cycle: str) -> tuple[str, str]:
    stream = stream_for(timestamp, cycle)
    date = timestamp.strftime("%Y%m%d")
    hour = timestamp.strftime("%H")
    stem = f"{date}{hour}0000-0h-{stream}-fc"
    base = f"{AWS_ROOT}/{date}/{hour}z/ifs/0p25/{stream}/{stem}"
    return f"{base}.index", f"{base}.grib2"


def timestamp_id(timestamp: datetime) -> str:
    return timestamp.strftime("%Y%m%dT%H%MZ")


def state_paths(timestamp: datetime) -> tuple[Path, Path]:
    directory = RAW_ROOT / "states" / timestamp_id(timestamp)
    return directory / "dynamic.grib2", directory / "dynamic.index.jsonl"


def build_plan(cohorts: tuple[Cohort, ...] = COHORTS) -> pd.DataFrame:
    memberships: dict[datetime, list[tuple[str, str, str, bool]]] = {}
    for cohort in cohorts:
        if operational_cycle(cohort.start) != cohort.cycle:
            raise ValueError(f"{cohort} starts outside its declared operational cycle")
        if operational_cycle(cohort.end) != cohort.cycle:
            raise ValueError(f"{cohort} ends outside its declared operational cycle")
        pair_times = set(cohort.pair_times)
        for timestamp in cohort.state_times:
            memberships.setdefault(timestamp, []).append(
                (cohort.comparison, cohort.cohort, cohort.cycle, timestamp in pair_times)
            )

    rows: list[dict[str, Any]] = []
    for timestamp, member_rows in sorted(memberships.items()):
        cycles = {row[2] for row in member_rows}
        if len(cycles) != 1:
            raise ValueError(f"Timestamp {timestamp} is assigned to multiple cycles: {cycles}")
        cycle = cycles.pop()
        if operational_cycle(timestamp) != cycle:
            raise ValueError(
                f"Timestamp {timestamp} is {operational_cycle(timestamp)}, not declared {cycle}"
            )
        grib_path, index_path = state_paths(timestamp)
        index_url, grib_url = source_urls(timestamp, cycle)
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "timestamp_id": timestamp_id(timestamp),
                "cycle": cycle,
                "stream": stream_for(timestamp, cycle),
                "is_pair_t0": any(row[3] for row in member_rows),
                "memberships": ";".join(
                    sorted(
                        f"{comparison}:{cohort}:{'t0' if is_t0 else 'lag'}"
                        for comparison, cohort, _, is_t0 in member_rows
                    )
                ),
                "index_url": index_url,
                "grib_url": grib_url,
                "grib_path": str(grib_path.relative_to(ROOT)),
                "selected_index_path": str(index_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def cohort_summary(plan: pd.DataFrame) -> dict[str, Any]:
    cohorts = [
        {
            "comparison": cohort.comparison,
            "cohort": cohort.cohort,
            "cycle": cohort.cycle,
            "first_t0": cohort.start.isoformat(),
            "last_t0": cohort.end.isoformat(),
            "pair_count": len(cohort.pair_times),
            "state_count_including_lag": len(cohort.state_times),
        }
        for cohort in COHORTS
    ]
    cycle_counts = plan.groupby(["cycle", "stream"]).size().to_dict()
    return {
        "cohorts": cohorts,
        "unique_state_count": len(plan),
        "unique_states_by_cycle_stream": {
            f"{cycle}/{stream}": int(count)
            for (cycle, stream), count in cycle_counts.items()
        },
        "dynamic_messages_per_state": len(DYNAMIC_FIELD_KEYS),
        "static_messages_per_cycle": len(STATIC_FIELD_KEYS),
        "cutover": IFS_50R1_CUTOVER.isoformat(),
        "may_12_policy": (
            "No 12 May t0 samples. The 18 UTC state is retained only as t-6 for "
            "13 May 00 UTC; both are 50r1."
        ),
    }


def _request(
    session: requests.Session,
    url: str,
    *,
    byte_range: tuple[int, int] | None = None,
    attempts: int = 12,
    timeout: int = 120,
) -> requests.Response:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1] - 1}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            return response
        except (requests.RequestException, TimeoutError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            delay = min(30.0, 1.5 * 2 ** min(attempt, 4)) + random.random()
            time.sleep(delay)
    raise RuntimeError(f"Failed after {attempts} attempts: {url}: {last_error}")


def fetch_index(session: requests.Session, url: str) -> list[dict[str, Any]]:
    response = _request(session, url)
    records = [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty index: {url}")
    return records


def record_key(record: dict[str, Any]) -> FieldKey:
    return FieldKey(
        str(record["levtype"]),
        str(record["param"]),
        int(record.get("levelist", 0)),
    )


def select_records(
    records: list[dict[str, Any]], expected: tuple[FieldKey, ...]
) -> list[dict[str, Any]]:
    expected_set = set(expected)
    selected = [
        record
        for record in records
        if str(record.get("step")) == "0" and record_key(record) in expected_set
    ]
    actual = [record_key(record) for record in selected]
    missing = expected_set - set(actual)
    duplicates = {key for key in actual if actual.count(key) > 1}
    unexpected = set(actual) - expected_set
    if missing or duplicates or unexpected or len(selected) != len(expected):
        raise ValueError(
            "Index selection mismatch: "
            f"missing={sorted(missing)}, duplicates={sorted(duplicates)}, "
            f"unexpected={sorted(unexpected)}, count={len(selected)}/{len(expected)}"
        )
    return sorted(selected, key=lambda record: int(record["_offset"]))


def range_groups(
    records: list[dict[str, Any]], max_gap_bytes: int
) -> list[tuple[int, int, list[dict[str, Any]]]]:
    groups: list[tuple[int, int, list[dict[str, Any]]]] = []
    for record in sorted(records, key=lambda item: int(item["_offset"])):
        start = int(record["_offset"])
        end = start + int(record["_length"])
        if groups and start - groups[-1][1] <= max_gap_bytes:
            group_start, _, group_records = groups[-1]
            groups[-1] = (group_start, end, [*group_records, record])
        else:
            groups.append((start, end, [record]))
    return groups


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_selected_ranges(
    session: requests.Session,
    grib_url: str,
    records: list[dict[str, Any]],
    target: Path,
    selected_index: Path,
    *,
    max_gap_bytes: int,
) -> tuple[int, str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    target_part = target.with_suffix(target.suffix + ".part")
    index_part = selected_index.with_suffix(selected_index.suffix + ".part")
    local_records: list[dict[str, Any]] = []
    local_offset = 0
    with target_part.open("wb") as output:
        for start, end, group_records in range_groups(records, max_gap_bytes):
            response = _request(session, grib_url, byte_range=(start, end))
            payload = response.content
            if response.status_code == 200 and len(payload) >= end:
                payload = payload[start:end]
            if len(payload) != end - start:
                raise IOError(
                    f"Range length mismatch {grib_url} [{start}:{end}]: "
                    f"expected {end-start}, received {len(payload)}"
                )
            for record in group_records:
                remote_start = int(record["_offset"])
                length = int(record["_length"])
                relative_start = remote_start - start
                message = payload[relative_start : relative_start + length]
                if len(message) != length or not message.startswith(b"GRIB"):
                    raise IOError(f"Invalid GRIB message at remote offset {remote_start}")
                output.write(message)
                local_record = dict(record)
                local_record["_remote_offset"] = remote_start
                local_record["_local_offset"] = local_offset
                local_record["_length"] = length
                local_record["_aifs_name"] = record_key(record).aifs_name
                local_records.append(local_record)
                local_offset += length
        output.flush()
        os.fsync(output.fileno())
    with index_part.open("w", encoding="utf-8") as output:
        for record in local_records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(target_part, target)
    os.replace(index_part, selected_index)
    return target.stat().st_size, sha256(target), len(local_records)


def _load_selected_index(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def existing_download_valid(
    target: Path, selected_index: Path, expected_count: int
) -> bool:
    if not target.exists() or target.stat().st_size == 0 or not selected_index.exists():
        return False
    try:
        records = _load_selected_index(selected_index)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if len(records) != expected_count:
        return False
    expected_bytes = sum(int(record["_length"]) for record in records)
    return expected_bytes == target.stat().st_size


def download_state(row: pd.Series, max_gap_bytes: int) -> dict[str, Any]:
    target = ROOT / str(row["grib_path"])
    selected_index = ROOT / str(row["selected_index_path"])
    if existing_download_valid(target, selected_index, len(DYNAMIC_FIELD_KEYS)):
        return {
            **row.to_dict(),
            "status": "existing",
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "message_count": len(DYNAMIC_FIELD_KEYS),
        }
    with requests.Session() as session:
        records = fetch_index(session, str(row["index_url"]))
        selected = select_records(records, DYNAMIC_FIELD_KEYS)
        size, digest, count = _write_selected_ranges(
            session,
            str(row["grib_url"]),
            selected,
            target,
            selected_index,
            max_gap_bytes=max_gap_bytes,
        )
    return {
        **row.to_dict(),
        "status": "downloaded",
        "bytes": size,
        "sha256": digest,
        "message_count": count,
    }


def static_representatives() -> dict[str, datetime]:
    return {
        "49r1": min(
            timestamp
            for cohort in COHORTS
            if cohort.cycle == "49r1"
            for timestamp in cohort.state_times
        ),
        "50r1": min(
            timestamp
            for cohort in COHORTS
            if cohort.cycle == "50r1"
            for timestamp in cohort.state_times
        ),
    }


def download_static(cycle: str, timestamp: datetime, max_gap_bytes: int) -> dict[str, Any]:
    target = RAW_ROOT / "static" / cycle / "static.grib2"
    selected_index = RAW_ROOT / "static" / cycle / "static.index.jsonl"
    index_url, grib_url = source_urls(timestamp, cycle)
    if existing_download_valid(target, selected_index, len(STATIC_FIELD_KEYS)):
        return {
            "timestamp": timestamp.isoformat(),
            "timestamp_id": timestamp_id(timestamp),
            "cycle": cycle,
            "stream": stream_for(timestamp, cycle),
            "status": "existing",
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "message_count": len(STATIC_FIELD_KEYS),
            "grib_path": str(target.relative_to(ROOT)),
            "selected_index_path": str(selected_index.relative_to(ROOT)),
            "kind": "static",
        }
    with requests.Session() as session:
        records = fetch_index(session, index_url)
        selected = select_records(records, STATIC_FIELD_KEYS)
        size, digest, count = _write_selected_ranges(
            session,
            grib_url,
            selected,
            target,
            selected_index,
            max_gap_bytes=max_gap_bytes,
        )
    return {
        "timestamp": timestamp.isoformat(),
        "timestamp_id": timestamp_id(timestamp),
        "cycle": cycle,
        "stream": stream_for(timestamp, cycle),
        "status": "downloaded",
        "bytes": size,
        "sha256": digest,
        "message_count": count,
        "grib_path": str(target.relative_to(ROOT)),
        "selected_index_path": str(selected_index.relative_to(ROOT)),
        "kind": "static",
    }


def estimate_from_probes(plan: pd.DataFrame, max_gap_bytes: int) -> dict[str, Any]:
    probes = (
        plan.groupby(["cycle", "stream"], sort=True)
        .first()
        .reset_index()
    )
    estimates: dict[tuple[str, str], dict[str, float]] = {}
    with requests.Session() as session:
        for _, row in probes.iterrows():
            records = fetch_index(session, str(row["index_url"]))
            selected = select_records(records, DYNAMIC_FIELD_KEYS)
            selected_bytes = sum(int(record["_length"]) for record in selected)
            ranged_bytes = sum(
                end - start
                for start, end, _ in range_groups(selected, max_gap_bytes)
            )
            estimates[(str(row["cycle"]), str(row["stream"]))] = {
                "selected_bytes_per_state": selected_bytes,
                "ranged_bytes_per_state": ranged_bytes,
                "range_requests_per_state": len(range_groups(selected, max_gap_bytes)),
            }
    total_selected = 0.0
    total_ranged = 0.0
    total_requests = 0.0
    rows = []
    for (cycle, stream), count in plan.groupby(["cycle", "stream"]).size().items():
        estimate = estimates[(cycle, stream)]
        rows.append({"cycle": cycle, "stream": stream, "states": int(count), **estimate})
        total_selected += count * estimate["selected_bytes_per_state"]
        total_ranged += count * estimate["ranged_bytes_per_state"]
        total_requests += count * estimate["range_requests_per_state"]
    return {
        "groups": rows,
        "estimated_selected_gib": total_selected / 2**30,
        "estimated_transferred_gib": total_ranged / 2**30,
        "estimated_range_requests": int(total_requests),
        "max_gap_bytes": max_gap_bytes,
        "note": "Estimate uses one index from each cycle/stream combination.",
    }


def audit_grib(
    grib_path: Path,
    expected: tuple[FieldKey, ...],
    timestamp: datetime,
    cycle: str,
) -> list[str]:
    issues: list[str] = []
    keys: list[FieldKey] = []
    with grib_path.open("rb") as stream:
        while True:
            handle = eccodes.codes_grib_new_from_file(stream)
            if handle is None:
                break
            try:
                type_of_level = str(eccodes.codes_get(handle, "typeOfLevel"))
                if type_of_level == "isobaricInhPa":
                    levtype = "pl"
                    level = int(eccodes.codes_get(handle, "level"))
                elif type_of_level == "soilLayer":
                    levtype = "sol"
                    level = int(eccodes.codes_get(handle, "level"))
                else:
                    levtype = "sfc"
                    level = 0
                key = FieldKey(
                    levtype,
                    str(eccodes.codes_get(handle, "shortName")),
                    level,
                )
                keys.append(key)
                checks = {
                    "dataDate": int(timestamp.strftime("%Y%m%d")),
                    "dataTime": timestamp.hour * 100,
                    "endStep": 0,
                    "gridType": "regular_ll",
                    "Ni": 1440,
                    "Nj": 721,
                    "numberOfDataPoints": 1_038_240,
                    "numberOfMissing": 0,
                }
                for metadata_key, expected_value in checks.items():
                    actual = eccodes.codes_get(handle, metadata_key)
                    if actual != expected_value:
                        issues.append(
                            f"{key.aifs_name}: {metadata_key}={actual!r}, "
                            f"expected {expected_value!r}"
                        )
            finally:
                eccodes.codes_release(handle)
    if set(keys) != set(expected) or len(keys) != len(expected):
        issues.append(
            f"field inventory mismatch: actual={len(keys)}, expected={len(expected)}, "
            f"missing={sorted(set(expected)-set(keys))}"
        )
    if operational_cycle(timestamp) != cycle:
        issues.append(
            f"timestamp belongs to {operational_cycle(timestamp)}, declared {cycle}"
        )
    return issues


def write_plan(probe_sizes: bool, max_gap_bytes: int) -> dict[str, Any]:
    plan = build_plan()
    PLAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    PLAN_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(PLAN_CSV, index=False)
    summary = cohort_summary(plan)
    if probe_sizes:
        summary["download_estimate"] = estimate_from_probes(plan, max_gap_bytes)
    PLAN_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_timestamp_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result = set()
    for value in values:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        result.add(timestamp_id(timestamp))
    return result


def command_download(args: argparse.Namespace) -> None:
    plan = build_plan()
    timestamp_filter = parse_timestamp_filter(args.timestamp)
    if timestamp_filter is not None:
        plan = plan[plan["timestamp_id"].isin(timestamp_filter)]
        missing = timestamp_filter - set(plan["timestamp_id"])
        if missing:
            raise SystemExit(f"Requested timestamps are outside the plan: {sorted(missing)}")
    if args.limit is not None:
        plan = plan.head(args.limit)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_state, row, args.max_gap_bytes): row["timestamp_id"]
            for _, row in plan.iterrows()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            timestamp = futures[future]
            result = future.result()
            result["kind"] = "dynamic"
            results.append(result)
            print(
                f"[{completed}/{len(futures)}] {timestamp} "
                f"{result['status']} {result['bytes']/2**20:.1f} MiB",
                flush=True,
            )
    if not args.skip_static:
        for cycle, timestamp in static_representatives().items():
            result = download_static(cycle, timestamp, args.max_gap_bytes)
            results.append(result)
            print(
                f"static {cycle} {result['status']} {result['bytes']/2**20:.1f} MiB",
                flush=True,
            )
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    if MANIFEST_CSV.exists():
        previous = pd.read_csv(MANIFEST_CSV)
        frame = pd.concat([previous, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            ["kind", "cycle", "timestamp_id"], keep="last"
        )
    frame.sort_values(["kind", "timestamp_id", "cycle"]).to_csv(
        MANIFEST_CSV, index=False
    )


def command_audit(args: argparse.Namespace) -> None:
    plan = build_plan()
    timestamp_filter = parse_timestamp_filter(args.timestamp)
    if timestamp_filter is not None:
        plan = plan[plan["timestamp_id"].isin(timestamp_filter)]
    if args.limit is not None:
        plan = plan.head(args.limit)
    failures: dict[str, list[str]] = {}
    for _, row in plan.iterrows():
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
        path = ROOT / str(row["grib_path"])
        if not path.exists():
            failures[str(row["timestamp_id"])] = ["missing"]
            continue
        issues = audit_grib(path, DYNAMIC_FIELD_KEYS, timestamp, str(row["cycle"]))
        if issues:
            failures[str(row["timestamp_id"])] = issues
    static_checked = 0
    if timestamp_filter is None and args.limit is None:
        for cycle, timestamp in static_representatives().items():
            static_checked += 1
            path = RAW_ROOT / "static" / cycle / "static.grib2"
            key = f"static-{cycle}"
            if not path.exists():
                failures[key] = ["missing"]
                continue
            issues = audit_grib(path, STATIC_FIELD_KEYS, timestamp, cycle)
            if issues:
                failures[key] = issues
    payload = {
        "files_checked": len(plan) + static_checked,
        "dynamic_files_checked": len(plan),
        "static_files_checked": static_checked,
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--probe-sizes", action="store_true")
    plan_parser.add_argument("--max-gap-bytes", type=int, default=1_048_576)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--workers", type=int, default=2)
    download_parser.add_argument("--max-gap-bytes", type=int, default=1_048_576)
    download_parser.add_argument("--limit", type=int)
    download_parser.add_argument("--timestamp", action="append")
    download_parser.add_argument("--skip-static", action="store_true")

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--limit", type=int)
    audit_parser.add_argument("--timestamp", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        print(json.dumps(write_plan(args.probe_sizes, args.max_gap_bytes), indent=2))
    elif args.command == "download":
        command_download(args)
    elif args.command == "audit":
        command_audit(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
