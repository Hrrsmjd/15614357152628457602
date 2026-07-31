"""Canonical presentation order for the complete AIFS input inventory.

The inferred scorecard groups fields by parameter and orders pressure levels
from low pressure to high pressure (50 hPa toward the surface).  This module
extends that convention to every dynamic input channel used in this study.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from open_data_inputs import DYNAMIC_FIELD_KEYS, PRESSURE_LEVELS


DISPLAY_PRESSURE_LEVELS = tuple(sorted(PRESSURE_LEVELS))


@dataclass(frozen=True)
class DisplayGroup:
    label: str
    variables: tuple[str, ...]


def pressure_group(parameter: str) -> tuple[str, ...]:
    return tuple(f"{parameter}_{level}" for level in DISPLAY_PRESSURE_LEVELS)


DISPLAY_GROUPS = (
    DisplayGroup("z / hPa", pressure_group("z")),
    DisplayGroup("msl", ("msl",)),
    DisplayGroup("t / hPa", pressure_group("t")),
    DisplayGroup("near-surface T", ("2t", "skt")),
    DisplayGroup("u / hPa", pressure_group("u")),
    DisplayGroup("10u", ("10u",)),
    DisplayGroup("v / hPa", pressure_group("v")),
    DisplayGroup("10v", ("10v",)),
    DisplayGroup("w / hPa", pressure_group("w")),
    DisplayGroup("q / hPa", pressure_group("q")),
    DisplayGroup("surface", ("2d", "sp", "tcw")),
    DisplayGroup("soil", ("stl1", "stl2", "swvl1", "swvl2")),
)

DISPLAY_VARIABLE_ORDER = tuple(
    variable for group in DISPLAY_GROUPS for variable in group.variables
)
DISPLAY_ORDER = {
    variable: position for position, variable in enumerate(DISPLAY_VARIABLE_ORDER)
}
DISPLAY_GROUP = {
    variable: group.label for group in DISPLAY_GROUPS for variable in group.variables
}
DISPLAY_LABEL = {
    variable: variable.replace("_", "") if "_" in variable else variable
    for variable in DISPLAY_VARIABLE_ORDER
}

EXPECTED_VARIABLES = {field.aifs_name for field in DYNAMIC_FIELD_KEYS}
if len(DISPLAY_VARIABLE_ORDER) != 90:
    raise RuntimeError(
        f"Canonical display order has {len(DISPLAY_VARIABLE_ORDER)} fields, expected 90"
    )
if set(DISPLAY_VARIABLE_ORDER) != EXPECTED_VARIABLES:
    raise RuntimeError(
        "Canonical display order does not match the downloaded input inventory: "
        f"missing={sorted(EXPECTED_VARIABLES-set(DISPLAY_VARIABLE_ORDER))}, "
        f"extra={sorted(set(DISPLAY_VARIABLE_ORDER)-EXPECTED_VARIABLES)}"
    )


def add_display_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable display metadata without changing scientific columns."""
    result = frame.copy()
    result["display_order"] = result["variable"].map(DISPLAY_ORDER)
    result["display_label"] = result["variable"].map(DISPLAY_LABEL)
    result["display_group"] = result["variable"].map(DISPLAY_GROUP)
    if result["display_order"].isna().any():
        unknown = sorted(result.loc[result["display_order"].isna(), "variable"].unique())
        raise ValueError(f"Variables missing from canonical display order: {unknown}")
    result["display_order"] = result["display_order"].astype(int)
    return result


def display_order_frame() -> pd.DataFrame:
    rows = []
    for position, variable in enumerate(DISPLAY_VARIABLE_ORDER):
        level_hpa = (
            int(variable.rsplit("_", 1)[1])
            if "_" in variable and variable.rsplit("_", 1)[1].isdigit()
            else pd.NA
        )
        rows.append(
            {
                "display_order": position,
                "variable": variable,
                "display_label": DISPLAY_LABEL[variable],
                "display_group": DISPLAY_GROUP[variable],
                "level_hpa": level_hpa,
            }
        )
    return pd.DataFrame(rows)
