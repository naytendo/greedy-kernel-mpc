"""Scientific checks on the complete supplied reference experiment."""
from pathlib import Path
import json
import unittest

from verify_results import verify_reference_files, verify_results

ROOT = Path(__file__).resolve().parents[1]


class ReferenceIntegrityTests(unittest.TestCase):
    def test_checksums_and_all_saved_trajectories(self):
        reference = ROOT / "data" / "reference"
        self.assertEqual(verify_reference_files(reference), 183)
        result = verify_results(reference, reference)
        self.assertEqual(result["run_count"], 180)
        self.assertEqual(result["recorded_steps"], 5400)
        self.assertTrue(result["passed"])

    def test_center_comparison_and_derivative_scope(self):
        reference = ROOT / "data" / "reference"
        d = json.loads((reference / "center-selection-comparison.json").read_text())
        self.assertEqual(d["protocol"]["random_repetitions"], 200)
        self.assertEqual(d["protocol"]["rng_seed"], 20261107)
        d = json.loads((reference / "derivative-certification.json").read_text())
        self.assertFalse(d["is_formal_interval_certificate"])
        self.assertEqual([m["centers"] for m in d["models"]], [9, 17, 33, 66])
        self.assertTrue(all(m["learned_model_contraction_upper"] < .7 for m in d["models"]))


if __name__ == "__main__":
    unittest.main()
