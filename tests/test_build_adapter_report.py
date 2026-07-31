from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_adapter_report as report  # noqa: E402


class BuildAdapterReportTest(unittest.TestCase):
    def test_candidate_labels_cover_report_defaults(self):
        self.assertEqual(
            set(report.ADAPTER_LABELS),
            {
                "residual_all_no_q50",
                "residual_all_no_q50_half",
                "residual_q100_q150",
                "hybrid_q50_affine_q100_q150",
                "hybrid_q50_affine_extratropics_q100_q150",
            },
        )

    def test_colour_sign(self):
        self.assertNotEqual(report._color(-5.0), report._color(5.0))
        self.assertEqual(report._color(0.0), "#f7f6f6")

    def test_final_scorecard_uses_paired_treatment_labels(self):
        page = report._final_scorecard_labels(
            "<strong>01 / 49r1 initialization</strong>"
            "<strong>02 / 50r1 initialization</strong>"
            "<th>49r1 RMSE</th><th>50r1 RMSE</th>"
            "Δ values are 50r1 − 49r1. "
            "These two fields are highlighted throughout."
        )
        self.assertIn("uncorrected 50r1 initialization", page)
        self.assertIn("corrected 50r1 initialization", page)
        self.assertIn("uncorrected RMSE", page)
        self.assertIn("corrected RMSE", page)
        self.assertIn(
            "corrected 50r1 − uncorrected 50r1", page
        )
        self.assertIn("These three fields", page)

    def test_selected_adapter_is_regional_hybrid(self):
        self.assertEqual(
            report.SELECTED_ADAPTER,
            "hybrid_q50_affine_extratropics_q100_q150",
        )


if __name__ == "__main__":
    unittest.main()
