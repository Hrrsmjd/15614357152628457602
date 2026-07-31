from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from public_score_analysis import (  # noqa: E402
    SCORECARDS,
    derive_v11_cycle_comparison,
    extract_subpages_data,
    scorecard_to_frame,
)


class PublicScoreAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frames = []
        for name, metadata in SCORECARDS.items():
            data = extract_subpages_data(metadata["path"])
            frames.append(scorecard_to_frame(name, metadata, data))
        cls.raw = pd.concat(frames, ignore_index=True)
        cls.derived = derive_v11_cycle_comparison(cls.raw)

    def test_embedded_payload_shape(self):
        for metadata in SCORECARDS.values():
            data = extract_subpages_data(metadata["path"])
            self.assertEqual(data["steps"], [24, 48, 72, 96, 120, 144, 168, 192, 216, 240])
            self.assertGreater(len(data), 700)

    def test_system_identities(self):
        systems = set(self.raw["system"])
        self.assertIn("AIFS-Single-v1.1_from_49r1", systems)
        self.assertIn("AIFS-Single-v1.1_from_50r1", systems)
        self.assertIn("AIFS-Single-v2_from_50r1", systems)

    def test_delta_formula(self):
        row = self.derived[
            (self.derived["parameter"] == "2t")
            & (self.derived["region"] == "n.hem")
            & (self.derived["metric"] == "rmsef")
            & (self.derived["reference_type"] == "ob")
            & (self.derived["lead_hours"] == 24)
        ].iloc[0]
        self.assertAlmostEqual(
            row["delta_50r1_minus_49r1"],
            row["score_50r1"] - row["score_49r1"],
            places=12,
        )
        self.assertGreater(row["delta_50r1_minus_49r1"], 0)

    def test_validity_is_never_misrepresented_as_controlled(self):
        self.assertFalse(self.derived["is_requested_controlled_comparison"].any())
        analysis_rmse = self.derived[
            (self.derived["reference_type"] == "an")
            & (self.derived["metric"] == "rmsef")
        ]
        self.assertTrue(
            analysis_rmse["validity"]
            .eq("different_analysis_truth_and_unmatched_aggregate_samples")
            .all()
        )

    def test_known_sample_mismatch(self):
        row = self.derived[
            (self.derived["parameter"] == "2t")
            & (self.derived["region"] == "n.hem")
            & (self.derived["metric"] == "rmsef")
            & (self.derived["reference_type"] == "ob")
            & (self.derived["lead_hours"] == 24)
        ].iloc[0]
        self.assertEqual(row["sample_count_49r1_scorecard"], 176)
        self.assertEqual(row["sample_count_50r1_scorecard"], 175)
        self.assertFalse(row["sample_counts_equal"])


if __name__ == "__main__":
    unittest.main()

