import csv
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import CLASS_CODES, SimulationConfig
from data_pipeline import generate_dataset
from research_protocol import generate_cohort, make_protocol


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ResearchStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from models import NeuronClassifier1D

        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.config = SimulationConfig(time_ms=60, dt_ms=.2, current_min=15, current_max=30)
        cls.checkpoint = cls.root / "UNTRAINED_TEST_FIXTURE.pt"
        cls.database = generate_dataset(cls.root / "fixture.sqlite", samples_per_class=30, config=cls.config)
        model = NeuronClassifier1D()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier[-1].bias[0] = 3
        torch.save({
            "model_state": model.state_dict(), "simulation": cls.config.to_dict(),
            "normalization": {"mean": -60., "std": 10.},
            "class_codes": list(CLASS_CODES), "confidence_threshold": .7,
        }, cls.checkpoint)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)
        cls.directory.cleanup()

    def test_baseline_ignores_validation_and_test_and_rejects_mismatched_metadata(self):
        from research_study import fit_baselines

        first, source, hashes = fit_baselines(self.config, database=self.database, synthetic_samples=30, burst_isi_ms=5)
        modified = self.root / "modified.sqlite"
        modified.write_bytes(self.database.read_bytes())
        with sqlite3.connect(modified) as connection:
            connection.execute("UPDATE samples SET trace=? WHERE split!='train'", (np.zeros(300, dtype=np.float32).tobytes(),))
        second, other_source, other_hashes = fit_baselines(self.config, database=modified, synthetic_samples=30, burst_isi_ms=5)
        self.assertEqual(source["rows_fitted"], 63)
        self.assertEqual(source["class_counts_fitted"], [21, 21, 21])
        self.assertEqual(hashes, other_hashes)
        self.assertNotEqual(source["sha256"], other_source["sha256"])
        for model in first:
            self.assertEqual(first[model].to_dict(), second[model].to_dict())
        with self.assertRaisesRegex(ValueError, "settings differ"):
            fit_baselines(replace(self.config, current_max=40), database=self.database, synthetic_samples=30, burst_isi_ms=5)

    def test_frozen_inference_remaps_columns_skips_invalid_input_and_preserves_traces(self):
        import torch
        from research_study import frozen_probabilities, load_frozen_checkpoint

        model, checkpoint, _ = load_frozen_checkpoint(self.checkpoint)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        checkpoint["class_codes"] = ["FS", "RS", "CH"]
        traces = np.full((4, 300), -65, dtype=np.float32, order="F")
        traces[-1, 100] = np.nan
        original = traces.copy()
        probabilities = frozen_probabilities(traces, model, checkpoint, batch_size=2)
        expected = torch.softmax(torch.tensor([0., 0., 3.]), dim=0).numpy()
        np.testing.assert_allclose(probabilities[:3], np.tile(expected, (3, 1)))
        self.assertTrue(np.isnan(probabilities[3]).all())
        np.testing.assert_array_equal(traces, original)

    def test_complete_study_keeps_all_rows_and_preserves_checkpoint_and_training_database(self):
        from research_study import file_digest, run_study

        before = file_digest(self.checkpoint), file_digest(self.database)
        conditions = {condition.name: condition for condition in make_protocol(self.config)}
        selected = [conditions[name] for name in ("reference", "noise_10", "jitter_0.15", "network_0.5", "pulse_onset_0.5", "ramp", "sine", "current_below_range")]
        output = self.root / "complete"
        report = run_study(checkpoint_path=self.checkpoint, database=self.database, output=output,
                           conditions=selected, seeds=[101, 202], samples_per_class=4, progress=lambda message: None)
        self.assertEqual((file_digest(self.checkpoint), file_digest(self.database)), before)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["results"]), 16)
        self.assertEqual(len(report["aggregates"]), 24)
        with (output / "predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16 * 12)
        for row in report["results"]:
            for score in row["scores"].values():
                self.assertEqual(score["total"], 12)
            self.assertEqual(row["scores"]["cnn_frozen"]["correct"], 4)
            self.assertAlmostEqual(row["scores"]["cnn_frozen"]["accuracy"], 1 / 3)
            self.assertLessEqual(row["scores"]["cnn_frozen"]["selective"]["coverage"], 1)
        serialized = json.loads((output / "study.json").read_text())
        self.assertEqual(serialized, report)
        self.assertIn("1.0", (output / "summary.csv").read_text())
        self.assertTrue((output / "report.html").is_file())
        self.assertTrue((output / "report.md").is_file())
        self.assertFalse((output / "predictions.csv.partial").exists())
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            run_study(checkpoint_path=self.checkpoint, database=self.database, output=output,
                      conditions=selected, seeds=[101], samples_per_class=4)
        self.assertEqual(json.loads((output / "study.json").read_text()), report)

    def test_quiet_cases_count_in_raw_accuracy_but_not_confidence_coverage(self):
        from research_features import extract_timing_features
        from research_study import fit_baselines, frozen_probabilities, load_frozen_checkpoint, score_condition

        baselines, _, _ = fit_baselines(self.config, database=self.database, synthetic_samples=30, burst_isi_ms=5)
        condition = replace(make_protocol(self.config)[0], current_range=(-20, -10), noise_std=0, jitter=0)
        cohort = generate_cohort(condition, self.config, 4, 101)
        features, counts, finite = extract_timing_features(cohort.traces, .2)
        model, checkpoint, _ = load_frozen_checkpoint(self.checkpoint)
        probabilities = frozen_probabilities(cohort.traces, model, checkpoint)
        scores, _, _, _, _ = score_condition(cohort, features, counts, finite, probabilities, baselines, self.config, .7)
        self.assertEqual(scores["cnn_frozen"]["total"], 12)
        self.assertAlmostEqual(scores["cnn_frozen"]["accuracy"], 1 / 3)
        self.assertEqual(scores["cnn_frozen"]["selective"]["coverage"], 0)
        self.assertIsNone(scores["cnn_frozen"]["selective"]["metrics"]["accuracy"])

    def test_exact_training_overlap_fails_and_marks_partial_output(self):
        from data_pipeline import load_split
        from research_study import run_study

        condition = make_protocol(self.config)[0]
        cohort = generate_cohort(condition, self.config, 1, 101)
        traces, _, _ = load_split(self.database, "train")
        cohort = replace(cohort, traces=traces[:3])
        output = self.root / "overlap"
        with patch("research_study.generate_cohort", return_value=cohort), self.assertRaisesRegex(ValueError, "overlap"):
            run_study(checkpoint_path=self.checkpoint, database=self.database, output=output,
                      conditions=[condition], seeds=[101], samples_per_class=1, progress=lambda message: None)
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
        self.assertFalse((output / "study.json").exists())

    def test_cli_requires_explicit_synthetic_fit_and_can_run_it(self):
        base = [sys.executable, str(ROOT / "evaluate_research.py"), "--checkpoint", str(self.checkpoint),
                "--database", str(self.root / "missing.sqlite"), "--threads", "2"]
        missing = subprocess.run(base, text=True, capture_output=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("--baseline-source synthetic", missing.stderr)
        output = self.root / "cli_synthetic"
        completed = subprocess.run(base + ["--baseline-source", "synthetic", "--synthetic-fit-per-class", "30",
                                          "--samples-per-class", "3", "--seeds", "101", "--conditions", "reference", "ramp",
                                          "--output", str(output)], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Checkpoint unchanged", completed.stdout)
        report = json.loads((output / "study.json").read_text())
        self.assertEqual(report["baseline_source"]["kind"], "explicit_synthetic_fit")
        self.assertEqual(len(report["results"]), 2)


if __name__ == "__main__":
    unittest.main()
