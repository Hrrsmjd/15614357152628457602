from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract_aifs_v11_metadata import (  # noqa: E402
    extract_statistics,
    normalization_method,
)


class AIFSV11MetadataTest(unittest.TestCase):
    def test_normalization_method_precedence(self):
        config = {
            "default": "mean-std",
            "none": ["swvl1"],
            "std": ["q_850"],
            "max": ["z"],
            "min-max": ["bounded"],
            "remap": {"cp": "tp"},
        }
        self.assertEqual(normalization_method("swvl1", config), ("none", "swvl1"))
        self.assertEqual(normalization_method("q_850", config), ("std", "q_850"))
        self.assertEqual(normalization_method("z", config), ("max", "z"))
        self.assertEqual(normalization_method("bounded", config), ("min-max", "bounded"))
        self.assertEqual(normalization_method("cp", config), ("mean-std", "tp"))

    def test_extract_statistics_maps_variables_to_scales(self):
        metadata = {
            "config": {
                "data": {
                    "normalizer": {
                        "default": "mean-std",
                        "none": ["swvl1"],
                        "std": ["q_850"],
                        "max": [],
                        "min-max": [],
                        "remap": {},
                    }
                }
            },
            "dataset": {
                "variables": ["2t", "q_850", "swvl1"],
                "statistics": {
                    "mean": [280.0, 0.002, 0.2],
                    "stdev": [5.0, 0.001, 0.1],
                    "minimum": [200.0, 0.0, 0.0],
                    "maximum": [330.0, 0.02, 1.0],
                },
            },
        }
        result = extract_statistics(metadata)
        self.assertEqual(result["2t"]["normalization_offset"], 280.0)
        self.assertEqual(result["2t"]["normalization_scale"], 5.0)
        self.assertEqual(result["q_850"]["normalization_offset"], 0.0)
        self.assertEqual(result["q_850"]["normalization_scale"], 0.001)
        self.assertEqual(result["swvl1"]["normalization_scale"], 1.0)
        self.assertTrue(np.isfinite(result["2t"]["normalization_scale"]))


if __name__ == "__main__":
    unittest.main()
