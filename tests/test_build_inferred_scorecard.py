from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_inferred_scorecard import (  # noqa: E402
    METRICS,
    OFFICIAL_FIELD_ORDER,
    _colour,
    _metric_legend,
    inferred_scorecard_frame,
    inferred_rmse_frame,
    load_derived_scores,
    render_html,
)


class InferredScorecardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.derived = load_derived_scores()
        cls.all_metrics = inferred_scorecard_frame(cls.derived)
        cls.frame = inferred_rmse_frame(cls.derived)

    def test_contains_all_published_rmse_rows(self):
        self.assertEqual(len(self.frame), 3190)
        self.assertEqual(self.frame["reference_type"].value_counts()["an"], 1620)
        self.assertEqual(self.frame["reference_type"].value_counts()["ob"], 1570)

    def test_requested_rmse_formula(self):
        row = self.frame[
            (self.frame["parameter"] == "2t")
            & (self.frame["region"] == "n.hem")
            & (self.frame["reference_type"] == "ob")
            & (self.frame["lead_hours"] == 24)
        ].iloc[0]
        expected = 100.0 * (row["score_50r1"] / row["score_49r1"] - 1.0)
        self.assertAlmostEqual(
            row["rmse_relative_change_percent"], expected, places=12
        )
        self.assertGreater(expected, 0)

    def test_sample_quality_flags(self):
        observation = self.frame[
            (self.frame["parameter"] == "2t")
            & (self.frame["region"] == "n.hem")
            & (self.frame["reference_type"] == "ob")
            & (self.frame["lead_hours"] == 24)
        ].iloc[0]
        self.assertFalse(observation["severe_sample_mismatch"])
        self.assertEqual(
            observation["comparison_quality"],
            "common_observation_type_one_case_population_gap",
        )
        severe = self.frame[
            (self.frame["parameter"] == "z")
            & (self.frame["level_hpa"] == "50")
            & (self.frame["region"] == "n.hem")
            & (self.frame["reference_type"] == "an")
            & (self.frame["lead_hours"] == 24)
        ].iloc[0]
        self.assertTrue(severe["severe_sample_mismatch"])
        self.assertEqual(severe["sample_count_49r1_scorecard"], 25)
        self.assertEqual(severe["sample_count_50r1_scorecard"], 179)

    def test_html_is_descriptive_and_contains_both_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "scorecard.html"
            render_html(self.all_metrics, destination)
            content = destination.read_text(encoding="utf-8")
        self.assertIn("Against observations", content)
        self.assertIn("Against analysis", content)
        self.assertLess(
            content.index("Against observations"),
            content.index("Against analysis"),
        )
        self.assertIn("RMSE50 / RMSE49", content)
        self.assertIn("No significance frames are shown", content)
        self.assertIn("n=176", content)
        self.assertIn("n=175", content)
        self.assertIn("SEEPS precipitation score", content)
        self.assertIn("Total precipitation", content)
        self.assertIn("ACC50 − ACC49", content)
        self.assertIn("--bg: #F8F2EA", content)
        self.assertIn("ui-monospace", content)
        self.assertIn('href="https://hrrs.ai/"', content)

    def test_official_field_and_level_order(self):
        self.assertEqual(
            [(parameter, level) for parameter, level, _, _ in OFFICIAL_FIELD_ORDER],
            [
                ("z", "50"),
                ("z", "100"),
                ("z", "250"),
                ("z", "500"),
                ("z", "850"),
                ("msl", "0"),
                ("t", "50"),
                ("t", "100"),
                ("t", "250"),
                ("t", "500"),
                ("t", "850"),
                ("2t", "0"),
                ("ff", "50"),
                ("ff", "100"),
                ("ff", "250"),
                ("ff", "500"),
                ("ff", "850"),
                ("10ff", "0"),
                ("2d", "0"),
                ("tp", "0"),
            ],
        )

    def test_every_colour_legend_runs_red_left_to_blue_right(self):
        for metric, metadata in METRICS.items():
            with self.subTest(metric=metric):
                legend = _metric_legend(metric)
                limit = float(metadata["limit"])
                left_value = limit if metadata["reverse_legend"] else -limit
                right_value = -left_value
                self.assertLess(
                    legend.index(_colour(metric, left_value)),
                    legend.index(_colour(metric, right_value)),
                )
                self.assertIn(
                    "Red is on the left and blue is on the right",
                    legend,
                )

    def test_all_embedded_metrics_are_preserved(self):
        self.assertEqual(len(self.all_metrics), 7820)
        self.assertEqual(
            set(self.all_metrics["metric"]),
            {"rmsef", "ccaf", "sdaf", "seeps"},
        )
        precipitation = self.all_metrics[
            (self.all_metrics["parameter"] == "tp")
            & (self.all_metrics["metric"] == "seeps")
        ]
        self.assertEqual(len(precipitation), 90)


if __name__ == "__main__":
    unittest.main()
