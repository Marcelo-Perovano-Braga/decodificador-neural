import importlib.util
import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import CLASS_CODES, CLASS_NAMES, SimulationConfig
from research_protocol import Condition, generate_cohort, stimulus
from robust_protocols import EXPANDED_PROFILES, EXPANDED_WITHHELD


class ExpandedDriveTests(unittest.TestCase):
    def test_random_drives_are_reproducible_varied_and_use_explicit_rng(self):
        amplitudes = np.full(6, 20, dtype=np.float32)
        for profile in EXPANDED_PROFILES:
            if not profile.startswith("random_"):
                continue
            first = stimulus(amplitudes, profile, 300, .2, rng=np.random.default_rng(101))
            repeated = stimulus(amplitudes, profile, 300, .2, rng=np.random.default_rng(101))
            other = stimulus(amplitudes, profile, 300, .2, rng=np.random.default_rng(102))
            np.testing.assert_array_equal(first, repeated)
            self.assertFalse(np.array_equal(first, other))
            self.assertFalse(np.array_equal(first[0], first[1]))
            self.assertEqual(first.dtype, np.float32)
            self.assertTrue(first.flags.f_contiguous)
            self.assertTrue(np.isfinite(first).all())
            self.assertGreaterEqual(first.min(), 0)
            self.assertLessEqual(first.max(), 40)
            with self.assertRaises(ValueError):
                stimulus(amplitudes, profile, 300, .2)

    def test_random_drive_stream_is_paired_across_noise_and_contains_no_labels(self):
        config = SimulationConfig(time_ms=30, dt_ms=.2)
        first = Condition("random", "test", 0, 0, .03, profile="random_smooth")
        clean = generate_cohort(first, config, 4, 123, namespace=31)
        noisy = generate_cohort(replace(first, noise_std=5), config, 4, 123, namespace=31)
        np.testing.assert_array_equal(clean.physical_spike_counts, noisy.physical_spike_counts)
        np.testing.assert_array_equal(clean.amplitudes, noisy.amplitudes)
        for name in clean.parameters:
            np.testing.assert_array_equal(clean.parameters[name], noisy.parameters[name])
        self.assertFalse(np.array_equal(clean.traces, noisy.traces))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ExpandedCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from models import NeuronClassifier1D
        from robust_candidate import train_candidate

        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.config = SimulationConfig(time_ms=60, dt_ms=.2, current_min=15, current_max=30)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.candidate = train_candidate(cls.config, cls.root / "training", train_per_class=2,
                                        validation_per_class=2, epochs=1, patience=1, protocol_version=2,
                                        out_degree=3, batch_size=32, progress=lambda _: None)
        cls.baseline = cls.root / "UNTRAINED_BASELINE_FIXTURE.pt"
        torch.save({"model_state": NeuronClassifier1D().state_dict(), "class_codes": list(CLASS_CODES),
                    "class_names": list(CLASS_NAMES), "simulation": cls.config.to_dict(),
                    "normalization": {"mean": -60., "std": 20.}, "confidence_threshold": .7}, cls.baseline)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)
        cls.directory.cleanup()

    def test_new_reservation_has_fresh_seeds_and_families_and_is_saved_before_fitting(self):
        from research_study import load_frozen_checkpoint
        from robust_candidate import COMPARISON_SEEDS, comparison_plan, plan_digest

        _, candidate, _ = load_frozen_checkpoint(self.candidate)
        protocol = candidate["training_protocol"]
        reserved_path = self.root / "training/reserved_comparison.json"
        reserved = json.loads(reserved_path.read_text())
        self.assertEqual(reserved, comparison_plan(self.config, 2))
        self.assertEqual(protocol["reserved_comparison_sha256"], plan_digest(reserved))
        self.assertLessEqual(reserved_path.stat().st_mtime_ns, self.candidate.stat().st_mtime_ns)
        self.assertFalse(set(reserved["seeds"]) & set(COMPARISON_SEEDS))
        self.assertEqual(reserved["namespace"], 33)
        fitted = {row["condition"]["profile"] for split in protocol["splits"].values() for row in split}
        self.assertFalse(fitted & set(EXPANDED_WITHHELD))
        self.assertTrue({"triangle", "double_pulse", "random_smooth"} <= fitted)
        self.assertEqual(len(reserved["conditions"]), 30)
        self.assertTrue(all(row["held_out_reason"] is None for row in reserved["conditions"] if row["profile"] == "triangle"))
        self.assertEqual({row["namespace"] for rows in protocol["splits"].values() for row in rows}, {30, 31, 32})
        self.assertIn("worst_condition_class_nll", protocol["best_selection"])

    def test_grouped_selection_detects_hidden_class_failure_and_reference_floor(self):
        from robust_candidate import Corpus, selection_summary

        labels = np.tile(np.repeat(np.arange(3), 2), 2)
        records = [{"condition": {"name": name}, "rows": 6} for name in ("constant_0", "triangle")]
        corpus = Corpus(np.empty((12, 0)), labels, records)
        good = np.eye(3)[labels] * 5
        failed = good.copy()
        failed[6:8] = [0, 0, 5]
        first, second = selection_summary(good, corpus), selection_summary(failed, corpus)
        self.assertTrue(second["reference_eligible"])
        self.assertEqual(second["minimum_condition_class_recall"], 0)
        self.assertGreater(second["score"], first["score"])
        failed[0] = [0, 0, 5]
        self.assertFalse(selection_summary(failed, corpus)["reference_eligible"])

    def test_complete_paired_expanded_comparison_preserves_both_checkpoints(self):
        from research_study import file_digest
        from robust_candidate import compare_candidates

        before = file_digest(self.baseline), file_digest(self.candidate)
        report = compare_candidates(self.baseline, self.candidate, self.config, self.root / "comparison",
                                     samples_per_class=2, progress=lambda _: None)
        self.assertEqual((file_digest(self.baseline), file_digest(self.candidate)), before)
        self.assertEqual(len(report["results"]), 150)
        self.assertEqual(report["protocol_version"], 2)
        self.assertEqual(set(report["withheld_profiles"]), set(EXPANDED_WITHHELD))
        for row in report["results"]:
            self.assertEqual(row["baseline"]["total"], 6)
            self.assertEqual(row["candidate"]["total"], 6)

    def test_adoption_rejects_old_protocol_or_modified_conditions_and_preserves_backup(self):
        from research_study import file_digest
        from robust_candidate import adopt_candidate, adoption_decision, comparison_plan, plan_digest

        # Artificial passing scores exercise the promotion guard, not scientific performance.
        plan = comparison_plan(self.config, 2)
        active = self.root / "active.pt"
        shutil.copyfile(self.baseline, active)
        old = active.read_bytes()
        results = []
        for condition in plan["conditions"]:
            for seed in plan["seeds"]:
                row = {"condition": deepcopy(condition), "seed": seed}
                for model, accuracy, error in (("baseline", .995 if condition["name"] == "reference" else .6, .2),
                                               ("candidate", 1., .01)):
                    row[model] = {"accuracy": accuracy, "macro_f1": accuracy, "coverage": .95,
                                  "accepted_errors_per_requested_neuron": error,
                                  "probabilities": {"nll": error, "brier": error, "ece": error}}
                results.append(row)
        report = {"status": "complete", "protocol_version": 2, "comparison_namespace": plan["namespace"],
                  "reserved_comparison_sha256": plan_digest(plan), "results": results,
                  "decision": adoption_decision(results), "baseline_sha256": file_digest(active),
                  "candidate_sha256": file_digest(self.candidate)}
        changed = deepcopy(report)
        changed["comparison_namespace"] = 23
        with self.assertRaises(ValueError):
            adopt_candidate(active, self.candidate, changed)
        changed = deepcopy(report)
        changed["results"][0]["condition"]["noise_std"] = 999
        with self.assertRaises(ValueError):
            adopt_candidate(active, self.candidate, changed)
        self.assertEqual(active.read_bytes(), old)
        adopted = adopt_candidate(active, self.candidate, json.loads(json.dumps(report)))
        self.assertEqual(active.read_bytes(), self.candidate.read_bytes())
        self.assertEqual(Path(adopted["backup"]).read_bytes(), old)


if __name__ == "__main__":
    unittest.main()
