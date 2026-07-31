from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_data_inputs import (  # noqa: E402
    DYNAMIC_FIELD_KEYS,
    STATIC_FIELD_KEYS,
    build_plan,
    operational_cycle,
    range_groups,
    stream_for,
)


class OpenDataInputsTest(unittest.TestCase):
    def test_reduced_plan_counts_and_deduplication(self):
        plan = build_plan()
        self.assertEqual(len(plan), 99)
        self.assertEqual(plan["timestamp_id"].nunique(), 99)
        self.assertEqual((plan["cycle"] == "49r1").sum(), 66)
        self.assertEqual((plan["cycle"] == "50r1").sum(), 33)
        shared = plan[plan["timestamp_id"] == "20260513T0000Z"].iloc[0]
        self.assertIn("same_season:50r1:t0", shared["memberships"])
        self.assertIn("cutover:50r1:t0", shared["memberships"])

    def test_cycle_boundary_and_stream_names(self):
        before = datetime(2026, 5, 12, 0, tzinfo=timezone.utc)
        after = datetime(2026, 5, 12, 6, tzinfo=timezone.utc)
        self.assertEqual(operational_cycle(before), "49r1")
        self.assertEqual(operational_cycle(after), "50r1")
        self.assertEqual(
            stream_for(datetime(2025, 5, 13, 6, tzinfo=timezone.utc), "49r1"),
            "scda",
        )
        self.assertEqual(stream_for(after, "50r1"), "oper")

    def test_may_12_is_only_a_lag_state_at_18_utc(self):
        plan = build_plan()
        may_12 = plan[plan["timestamp_id"].str.startswith("20260512")]
        self.assertEqual(may_12["timestamp_id"].tolist(), ["20260512T1800Z"])
        self.assertFalse(bool(may_12.iloc[0]["is_pair_t0"]))

    def test_exact_field_inventory(self):
        self.assertEqual(len(DYNAMIC_FIELD_KEYS), 90)
        self.assertEqual(len({field.aifs_name for field in DYNAMIC_FIELD_KEYS}), 90)
        self.assertEqual(len(STATIC_FIELD_KEYS), 4)
        self.assertIn("z_500", {field.aifs_name for field in DYNAMIC_FIELD_KEYS})
        self.assertIn("swvl1", {field.aifs_name for field in DYNAMIC_FIELD_KEYS})

    def test_range_grouping(self):
        records = [
            {"_offset": 0, "_length": 100},
            {"_offset": 100, "_length": 50},
            {"_offset": 200, "_length": 25},
        ]
        groups = range_groups(records, max_gap_bytes=49)
        self.assertEqual([(start, end) for start, end, _ in groups], [(0, 150), (200, 225)])
        groups = range_groups(records, max_gap_bytes=50)
        self.assertEqual([(start, end) for start, end, _ in groups], [(0, 225)])


if __name__ == "__main__":
    unittest.main()
