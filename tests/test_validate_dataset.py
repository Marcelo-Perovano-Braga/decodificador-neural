import tempfile
import unittest
from pathlib import Path

from config import SimulationConfig
from data_pipeline import generate_dataset
from validate_dataset import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def test_generated_dataset_passes_integrity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.sqlite"
            generate_dataset(
                path,
                samples_per_class=30,
                seed=11,
                config=SimulationConfig(
                    time_ms=80.0,
                    dt_ms=0.2,
                    current_min=12.0,
                    current_max=25.0,
                    minimum_spikes=1,
                ),
            )
            report = validate_dataset(path)
            self.assertTrue(report["valid"])
            self.assertEqual(report["total_samples"], 90)


if __name__ == "__main__":
    unittest.main()
