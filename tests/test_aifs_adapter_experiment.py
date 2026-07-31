from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import aifs_adapter_experiment as adapters  # noqa: E402


class AifsAdapterExperimentTest(unittest.TestCase):
    def test_frozen_split_excludes_evaluation_input_states(self):
        calibration = adapters.calibration_initializations()
        evaluation = adapters.evaluation_initializations()
        self.assertEqual(calibration.groupby("cycle").size().to_dict(), {
            "49r1": 32,
            "50r1": 15,
        })
        self.assertEqual(len(evaluation), 16)
        self.assertEqual(
            evaluation.iloc[0]["timestamp"],
            datetime(2026, 5, 17, tzinfo=timezone.utc).isoformat(),
        )

    def test_adapter_inventory(self):
        self.assertEqual(
            adapters.ADAPTERS["residual_q100_q150"].residual_variables,
            {"q_100", "q_150"},
        )
        self.assertNotIn(
            "q_50",
            adapters.ADAPTERS["residual_all_no_q50"].residual_variables,
        )
        self.assertTrue(
            adapters.ADAPTERS[
                "hybrid_q50_affine_q100_q150"
            ].q50_affine
        )
        self.assertEqual(
            adapters.ADAPTERS[
                "hybrid_q50_affine_extratropics_q100_q150"
            ].q50_affine_bands,
            {*range(0, 7), *range(11, 18)},
        )

    def test_residual_field_correction(self):
        masks = {}
        lookup = {}
        values = np.full(18, 0.2, dtype=np.float32)
        for latitude_band in range(18):
            mask = np.zeros(18, dtype=bool)
            mask[latitude_band] = True
            masks[(latitude_band, "all")] = mask
            lookup[("q_100", latitude_band, "all")] = 0.05
        with (
            patch.object(
                adapters,
                "stratum_geometry",
                return_value={"masks": masks},
            ),
            patch.object(adapters, "_residual_lookup", return_value=lookup),
        ):
            corrected, diagnostics = adapters.apply_adapter_field(
                values, "q_100", "residual_q100_q150"
            )
        np.testing.assert_allclose(corrected, 0.15)
        self.assertEqual(
            diagnostics["transformation"], "residual_additive"
        )

    def test_q50_robust_affine(self):
        masks = {}
        lookup = {}
        values = np.arange(18, dtype=np.float32)
        for latitude_band in range(18):
            mask = np.zeros(18, dtype=bool)
            mask[latitude_band] = True
            masks[(latitude_band, "all")] = mask
            lookup[latitude_band] = (2.0, 1.0, 0.5)
        with (
            patch.object(
                adapters,
                "stratum_geometry",
                return_value={"masks": masks},
            ),
            patch.object(
                adapters, "_q50_affine_lookup", return_value=lookup
            ),
        ):
            corrected, diagnostics = adapters.apply_adapter_field(
                values, "q_50", "hybrid_q50_affine_q100_q150"
            )
        np.testing.assert_allclose(corrected, 2.0 + (values - 1.0) * 0.5)
        self.assertEqual(diagnostics["transformation"], "robust_affine")

    def test_q50_extratropical_affine_preserves_tropical_bands(self):
        masks = {}
        lookup = {}
        values = np.arange(18, dtype=np.float32) + 2.0
        for latitude_band in range(18):
            mask = np.zeros(18, dtype=bool)
            mask[latitude_band] = True
            masks[(latitude_band, "all")] = mask
            lookup[latitude_band] = (2.0, 1.0, 0.5)
        with (
            patch.object(
                adapters,
                "stratum_geometry",
                return_value={"masks": masks},
            ),
            patch.object(
                adapters, "_q50_affine_lookup", return_value=lookup
            ),
        ):
            corrected, _ = adapters.apply_adapter_field(
                values,
                "q_50",
                "hybrid_q50_affine_extratropics_q100_q150",
            )
        expected = 2.0 + (values - 1.0) * 0.5
        expected[7:11] = values[7:11]
        np.testing.assert_allclose(corrected, expected)

    def test_paired_statistics_use_common_cases(self):
        result = adapters.paired_cell_statistics(
            np.array([1.0, 4.0]),
            np.array([1.0, 1.0]),
            np.array([0.1, 0.2]),
            np.array([0.0, 0.1]),
            draws=200,
            seed_parts=("test",),
        )
        self.assertAlmostEqual(result["rmse_baseline"], np.sqrt(2.5))
        self.assertAlmostEqual(result["rmse_adapter"], 1.0)
        self.assertAlmostEqual(
            result["bias_difference_adapter_minus_baseline"], -0.1
        )
        self.assertEqual(result["forecast_case_count"], 2)


if __name__ == "__main__":
    unittest.main()
