from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from variable_order import (  # noqa: E402
    DISPLAY_PRESSURE_LEVELS,
    DISPLAY_VARIABLE_ORDER,
)


class VariableOrderTest(unittest.TestCase):
    def test_complete_unique_inventory(self):
        self.assertEqual(len(DISPLAY_VARIABLE_ORDER), 90)
        self.assertEqual(len(set(DISPLAY_VARIABLE_ORDER)), 90)

    def test_extends_inferred_scorecard_pressure_order(self):
        self.assertEqual(
            DISPLAY_PRESSURE_LEVELS,
            (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        )
        self.assertEqual(DISPLAY_VARIABLE_ORDER[:5], ("z_50", "z_100", "z_150", "z_200", "z_250"))
        self.assertEqual(DISPLAY_VARIABLE_ORDER[13], "msl")
        self.assertEqual(DISPLAY_VARIABLE_ORDER[14], "t_50")


if __name__ == "__main__":
    unittest.main()
