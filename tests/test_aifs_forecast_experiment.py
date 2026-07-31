from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aifs_forecast_experiment import (  # noqa: E402
    EVALUATION_VARIABLES,
    HIGHLIGHT_VARIABLES,
    LEADS,
    canonical_units,
    cohort_initializations,
    required_valid_times,
    score_field,
    score_field_matrix,
)
import numpy as np


class AifsForecastExperimentTest(unittest.TestCase):
    def test_unique_forecasts_and_shared_50r1_cohort(self):
        frame = cohort_initializations()
        self.assertEqual(len(frame), 96)
        shared = frame[
            frame["timestamp"].eq(
                datetime(2026, 5, 13, tzinfo=timezone.utc).isoformat()
            )
        ].iloc[0]
        self.assertIn("cutover:50r1", shared["memberships"])
        self.assertIn("same_season:50r1", shared["memberships"])

    def test_instantaneous_variable_inventory(self):
        self.assertEqual(len(EVALUATION_VARIABLES), 96)
        self.assertEqual(len(set(EVALUATION_VARIABLES)), 96)
        self.assertTrue(set(HIGHLIGHT_VARIABLES) <= set(EVALUATION_VARIABLES))
        self.assertIn("q_50", EVALUATION_VARIABLES)
        self.assertIn("q_100", EVALUATION_VARIABLES)
        self.assertIn("tcc", EVALUATION_VARIABLES)
        self.assertNotIn("tp", EVALUATION_VARIABLES)

    def test_daily_leads(self):
        self.assertEqual(LEADS, tuple(range(24, 241, 24)))

    def test_reference_time_deduplication(self):
        times = required_valid_times()
        self.assertEqual(len(times), 172)
        self.assertEqual(times[0].isoformat(), "2025-05-14T00:00:00+00:00")
        self.assertEqual(times[-1].isoformat(), "2026-05-30T18:00:00+00:00")

    def test_unit_canonicalization(self):
        self.assertEqual(canonical_units("kg kg**-1"), "kg kg-1")
        self.assertEqual(canonical_units("m**2 s**-2"), "m2 s-2")
        self.assertEqual(canonical_units("(0 - 1)"), "1")

    def test_area_weighted_scores_and_missing_values(self):
        forecast = np.array([2.0, 4.0, np.nan])
        reference = np.array([1.0, 2.0, 9.0])
        weights = np.array([1.0, 3.0, 6.0])
        masks = {"global": np.ones(3, dtype=bool)}
        row = score_field(forecast, reference, weights, masks)[0]
        self.assertAlmostEqual(row["bias"], 1.75)
        self.assertAlmostEqual(row["mean_squared_error"], 3.25)
        self.assertAlmostEqual(row["valid_area_fraction"], 0.4)
        self.assertEqual(row["valid_gridpoint_count"], 2)

    def test_vectorized_scores_match_scalar_scores(self):
        forecasts = np.array([[2.0, 4.0, np.nan], [0.0, 1.0, 3.0]])
        references = np.array([[1.0, 2.0, 9.0], [1.0, 1.0, 2.0]])
        weights = np.array([1.0, 3.0, 6.0])
        masks = {
            "all": np.ones(3, dtype=bool),
            "first_two": np.array([True, True, False]),
        }
        matrix = score_field_matrix(forecasts, references, weights, masks)
        for index in range(2):
            scalar = score_field(
                forecasts[index], references[index], weights, masks
            )
            for scalar_row, matrix_row in zip(scalar, matrix[index], strict=True):
                self.assertEqual(scalar_row["region"], matrix_row["region"])
                for key in ("bias", "mean_squared_error", "rmse", "weight_sum"):
                    self.assertAlmostEqual(scalar_row[key], matrix_row[key])
                self.assertEqual(
                    scalar_row["valid_gridpoint_count"],
                    matrix_row["valid_gridpoint_count"],
                )


if __name__ == "__main__":
    unittest.main()
