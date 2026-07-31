from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_forecast_scorecards import SCORECARDS, _variable_groups  # noqa: E402


class BuildForecastScorecardsTest(unittest.TestCase):
    def test_scorecards_are_separate(self):
        self.assertEqual(set(SCORECARDS), {"same_season", "cutover"})
        self.assertNotEqual(
            SCORECARDS["same_season"]["filename"],
            SCORECARDS["cutover"]["filename"],
        )

    def test_variable_groups_cover_every_field_once(self):
        flattened = [
            variable for _, variables in _variable_groups() for variable in variables
        ]
        self.assertEqual(len(flattened), 96)
        self.assertEqual(len(set(flattened)), 96)
        self.assertIn("q_50", flattened)
        self.assertIn("q_100", flattened)


if __name__ == "__main__":
    unittest.main()
