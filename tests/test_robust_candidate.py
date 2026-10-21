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
from research_protocol import generate_cohort, make_protocol, stimulus


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ExpandedStimulusTests(unittest.TestCase):
    def test_pulse_duration_and_late_onset_and_candidate_holdout_labels(self):
        amplitude = np.array([20], dtype=np.float32)
        drive = stimulus(amplitude, "pulse", 100, .1, .6, .3)
        np.testing.assert_array_equal(drive[0, :60], 0)
        np.testing.assert_array_equal(drive[0, 60:90], 20)
        np.testing.assert_array_equal(drive[0, 90:], 0)
        config = SimulationConfig(time_ms=10)
        original = {c.name: c for c in make_protocol(config)}
        robust = {c.name: c for c in make_protocol(config, training_profiles=["constant", "pulse"])}
        self.assertIsNotNone(original["pulse_onset_0.1"].held_out_reason)
        self.assertIsNone(robust["pulse_onset_0.1"].held_out_reason)
        self.assertIsNotNone(robust["ramp"].held_out_reason)
        with self.assertRaises(ValueError):
            generate_cohort(replace(original["pulse_onset_0.1"], onset_fraction=.8), config, 1, 1)

    def test_new_waveforms_have_expected_envelopes_and_frequency_varies(self):
        amplitudes = np.array([20], dtype=np.float32)
        double = stimulus(amplitudes, "double_pulse", 100, 1)
        np.testing.assert_array_equal(np.flatnonzero(double[0]), np.r_[10:30, 60:80])
        triangle = stimulus(amplitudes, "triangle", 100, 1)
        self.assertAlmostEqual(triangle[0, 0], 5)
        self.assertAlmostEqual(triangle[0, 50], 35)
        self.assertFalse(np.array_equal(stimulus(amplitudes, "sine", 100, 1, frequency_hz=10),
                                        stimulus(amplitudes, "sine", 100, 1, frequency_hz=20)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class RobustCandidateTests(unittest.TestCase):
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
                                         validation_per_class=2, epochs=1, patience=1,
                                         out_degree=3, batch_size=32, progress=lambda _: None)
        model = NeuronClassifier1D()
        cls.baseline = cls.root / "UNTRAINED_BASELINE_FIXTURE.pt"
        torch.save({"model_state": model.state_dict(), "class_codes": list(CLASS_CODES),
                    "class_names": list(CLASS_NAMES), "simulation": cls.config.to_dict(),
                    "normalization": {"mean": -60., "std": 20.}, "confidence_threshold": .7}, cls.baseline)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)
        cls.directory.cleanup()

    @staticmethod
    def hypothetical_results():
        rows = []
        for seed in (1, 2, 3):
            for name, axis, old, new in (("reference", "reference", .99, .995),
                                         ("pulse", "pulse", .45, .95), ("noise", "noise", .7, .9)):
                row = {"condition": {"name": name, "axis": axis}, "seed": seed}
                for model, accuracy, error, nll in (("baseline", old, .2, 1.), ("candidate", new, .02, .1)):
                    row[model] = {"accuracy": accuracy, "macro_f1": accuracy, "coverage": .95,
                                  "accepted_errors_per_requested_neuron": error,
                                  "probabilities": {"nll": nll, "brier": nll / 2, "ece": nll / 3}}
                rows.append(row)
        return rows

    def test_training_emits_loadable_calibrated_checkpoint_and_split_provenance(self):
        from research_study import load_frozen_checkpoint
        from robust_candidate import COMPARISON_NAMESPACE, COMPARISON_SEEDS, comparison_conditions

        _, checkpoint, config = load_frozen_checkpoint(self.candidate)
        self.assertEqual(config, self.config)
        self.assertEqual(checkpoint["training_profiles"], ["constant", "pulse", "ramp", "sine"])
        self.assertGreater(checkpoint["temperature"], 0)
        protocol = checkpoint["training_protocol"]
        self.assertEqual(protocol["split_rows"], {"train": 150, "selection": 150, "calibration": 150})
        split_seeds = {name: {row["seed"] for row in records} for name, records in protocol["splits"].items()}
        self.assertFalse(split_seeds["train"] & split_seeds["selection"])
        self.assertFalse(split_seeds["train"] & split_seeds["calibration"])
        self.assertFalse(split_seeds["calibration"] & set(COMPARISON_SEEDS))
        self.assertNotIn(COMPARISON_NAMESPACE, {row["namespace"] for rows in protocol["splits"].values() for row in rows})
        self.assertLessEqual(protocol["calibration"]["nll_after"], protocol["calibration"]["nll_before"] + 1e-12)
        withheld = {c.profile for c in comparison_conditions(config) if c.axis == "withheld_family"}
        fitted = {row["condition"]["profile"] for rows in protocol["splits"].values() for row in rows}
        self.assertFalse(withheld & fitted)

    def test_temperature_fit_reduces_overconfidence_without_changing_winning_class(self):
        import torch
        from robust_candidate import fit_temperature

        logits = np.tile([8., 0., 0.], (90, 1))
        labels = np.tile([0, 0, 1], 30)
        calibration = fit_temperature(logits, labels)
        self.assertGreater(calibration["temperature"], 1)
        self.assertLess(calibration["nll_after"], calibration["nll_before"])
        np.testing.assert_array_equal(logits.argmax(axis=1), (logits / calibration["temperature"]).argmax(axis=1))
        self.assertTrue(torch.isfinite(torch.tensor(calibration["nll_after"])))

    def test_calibration_is_applied_equally_in_gui_and_research_inference(self):
        import torch
        from inference import classify_traces
        from research_study import frozen_probabilities

        class FixedLogits(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([4., 0., 0.]).repeat(len(x), 1)

        model = FixedLogits()
        traces = np.zeros((3, 8), dtype=np.float32)
        traces[:, [2, 5]] = 30
        options = dict(normalization_mean=0., normalization_std=1., class_codes=list(CLASS_CODES),
                       class_names=list(CLASS_NAMES), minimum_spikes=2, confidence_threshold=.7,
                       device=torch.device("cpu"))
        original = classify_traces(traces, model, **options)
        calibrated = classify_traces(traces, model, **options, temperature=4.)
        self.assertTrue(all(row.determined for row in original))
        self.assertTrue(all(not row.determined for row in calibrated))
        self.assertEqual([row.class_code for row in original], [row.class_code for row in calibrated])
        probabilities = frozen_probabilities(traces, model, {"normalization": {"mean": 0., "std": 1.},
                                                            "class_codes": list(CLASS_CODES), "temperature": 4.})
        np.testing.assert_allclose(probabilities.max(axis=1), [row.confidence for row in calibrated])
        with self.assertRaises(ValueError):
            classify_traces(traces, model, **options, temperature=0)

    def test_gates_reject_reference_regression_and_coverage_collapse(self):
        from robust_candidate import adoption_decision

        good = self.hypothetical_results()
        self.assertTrue(adoption_decision(good)["superior"])
        regression = deepcopy(good)
        for row in regression:
            if row["condition"]["name"] == "reference":
                row["candidate"]["accuracy"] = .96
        decision = adoption_decision(regression)
        self.assertFalse(decision["superior"])
        self.assertFalse(decision["checks"]["reference_accuracy_floor"])
        collapse = deepcopy(good)
        for row in collapse:
            row["candidate"]["coverage"] = .1
            row["candidate"]["accepted_errors_per_requested_neuron"] = 0.
        self.assertFalse(adoption_decision(collapse)["checks"]["coverage_preserved"])

    def test_paired_comparison_retains_all_neurons_and_does_not_modify_weights(self):
        from research_study import file_digest
        from robust_candidate import compare_candidates

        before = file_digest(self.baseline), file_digest(self.candidate)
        output = self.root / "comparison"
        report = compare_candidates(self.baseline, self.candidate, self.config, output,
                                     samples_per_class=2, progress=lambda _: None)
        self.assertEqual((file_digest(self.baseline), file_digest(self.candidate)), before)
        self.assertEqual(len(report["results"]), 100)
        self.assertEqual(len(report["seeds"]), 5)
        self.assertEqual(report["status"], "complete")
        for row in report["results"]:
            self.assertEqual(row["baseline"]["total"], 6)
            self.assertEqual(row["candidate"]["total"], 6)
        saved = json.loads((output / "comparison.json").read_text())
        self.assertEqual(saved["decision"], report["decision"])
        with self.assertRaises(ValueError):
            compare_candidates(self.baseline, self.candidate, self.config, output, samples_per_class=2)

    def test_adoption_requires_passing_gate_hashes_and_keeps_exact_backup(self):
        from research_study import file_digest
        from robust_candidate import COMPARISON_NAMESPACE, COMPARISON_SEEDS, adopt_candidate, adoption_decision, comparison_conditions

        active = self.root / "active.pt"
        candidate = self.root / "candidate_for_adoption.pt"
        shutil.copyfile(self.baseline, active)
        shutil.copyfile(self.candidate, candidate)
        old = active.read_bytes()
        report = {"status": "complete", "baseline_sha256": file_digest(active),
                  "candidate_sha256": file_digest(candidate), "decision": {"superior": False}}
        with self.assertRaises(ValueError):
            adopt_candidate(active, candidate, report)
        self.assertEqual(active.read_bytes(), old)
        samples = self.hypothetical_results()
        report["results"] = []
        for condition in comparison_conditions(self.config):
            for seed in COMPARISON_SEEDS:
                row = deepcopy(samples[0 if condition.name == "reference" else 1])
                row["condition"] = {"name": condition.name, "axis": condition.axis}
                row["seed"] = seed
                report["results"].append(row)
        report["comparison_namespace"] = COMPARISON_NAMESPACE
        report["decision"] = adoption_decision(report["results"])
        original_hash = report["candidate_sha256"]
        report["candidate_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            adopt_candidate(active, candidate, report)
        self.assertEqual(active.read_bytes(), old)
        report["candidate_sha256"] = original_hash
        adopted = adopt_candidate(active, candidate, report)
        self.assertEqual(active.read_bytes(), candidate.read_bytes())
        self.assertEqual(Path(adopted["backup"]).read_bytes(), old)
        self.assertEqual(adopted["backup_sha256"], report["baseline_sha256"])


if __name__ == "__main__":
    unittest.main()
