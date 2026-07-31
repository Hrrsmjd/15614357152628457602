from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_open_data_inputs import (  # noqa: E402
    BinnedMoments,
    cell_indices,
    spatial_cells,
)


class AnalyzeOpenDataInputsTest(unittest.TestCase):
    def test_binned_moments_and_derived_groups(self):
        latitudes = np.array([0.0, 10.0, 30.0, -45.0, 5.0, 70.0])
        surface = np.array([0, 1, 0, 1, 2, 2], dtype=np.int8)
        cells = spatial_cells(latitudes, surface)
        values = np.arange(1.0, 7.0)
        weights = np.ones(6)
        moments = BinnedMoments.empty()
        moments.update(values, weights, cells)
        global_all = moments.summarize(cell_indices("global", "all"))
        self.assertAlmostEqual(global_all["mean"], 3.5)
        self.assertEqual(global_all["point_observations"], 6)
        tropical_land = moments.summarize(cell_indices("tropics", "land"))
        self.assertAlmostEqual(tropical_land["mean"], 1.0)
        extratropical_ocean = moments.summarize(
            cell_indices("extratropics", "ocean")
        )
        self.assertAlmostEqual(extratropical_ocean["mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
