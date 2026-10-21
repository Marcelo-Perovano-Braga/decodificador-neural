import sqlite3
import tempfile
import unittest
from pathlib import Path

from config import SimulationConfig
from data_pipeline import generate_dataset, load_split, read_metadata


class DatasetTests(unittest.TestCase):
    def test_sqlite_dataset_is_balanced_and_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.sqlite"
            config = SimulationConfig(
                time_ms=80.0,
                dt_ms=0.2,
                current_min=12.0,
                current_max=25.0,
                parameter_jitter=0.01,
                observation_noise_std=0.0,
                minimum_spikes=1,
            )
            generate_dataset(
                path,
                samples_per_class=30,
                seed=7,
                config=config,
            )
            metadata = read_metadata(path)
            self.assertEqual(metadata["samples_per_class"], 30)
            self.assertEqual(metadata["simulation"]["window_size"], 400)

            expected_per_class = {"train": 21, "validation": 4, "test": 5}
            with sqlite3.connect(path) as connection:
                for split, expected in expected_per_class.items():
                    rows = connection.execute(
                        "SELECT label, COUNT(*) FROM samples WHERE split=? GROUP BY label",
                        (split,),
                    ).fetchall()
                    self.assertEqual(rows, [(0, expected), (1, expected), (2, expected)])

            traces, labels, sample_metadata = load_split(path, "test")
            self.assertEqual(traces.shape, (15, 400))
            self.assertEqual(labels.shape, (15,))
            self.assertEqual(sample_metadata["spike_count"].shape, (15,))


if __name__ == "__main__":
    unittest.main()

