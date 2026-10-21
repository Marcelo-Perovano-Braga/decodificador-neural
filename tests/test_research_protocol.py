import unittest
from dataclasses import replace

import numpy as np

from config import SimulationConfig
from network import NetworkConfig, build_sparse_network, simulate_connected_batch
from research_features import TimingCentroid, extract_timing_features, prediction_metrics, probability_metrics
from research_protocol import Condition, generate_cohort, make_protocol, stimulus
from simulation import IzhikevichBatch, simulate_izhikevich_batch


class StimulusTests(unittest.TestCase):
    def setUp(self):
        self.batch = IzhikevichBatch(
            np.array([.02, .02, .1]), np.full(3, .2), np.array([-65, -50, -65]),
            np.array([8, 2, 2]), np.array([15, 20, 25]),
        )

    def test_constant_matrix_preserves_both_solvers_exactly(self):
        matrix = np.repeat(self.batch.current[:, None], 300, axis=1)
        varying = replace(self.batch, current=matrix)
        expected = simulate_izhikevich_batch(self.batch, time_ms=30)
        np.testing.assert_array_equal(simulate_izhikevich_batch(varying, time_ms=30), expected)
        for weight in (0, .5):
            graph = build_sparse_network(np.arange(3), NetworkConfig(out_degree=2, weight=weight), 9)
            first, activity = simulate_connected_batch(self.batch, graph, time_ms=30)
            second, other = simulate_connected_batch(varying, graph, time_ms=30)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(activity.spike_counts, other.spike_counts)
            if weight == 0:
                np.testing.assert_array_equal(second, expected)

    def test_late_drive_does_not_change_earlier_samples(self):
        graph = build_sparse_network(np.arange(3), NetworkConfig(out_degree=2), 9)
        quiet = np.zeros((3, 300), dtype=np.float32)
        late = quiet.copy()
        late[:, 150:] = 25
        for solver in (
            lambda current: simulate_izhikevich_batch(replace(self.batch, current=current), time_ms=30),
            lambda current: simulate_connected_batch(replace(self.batch, current=current), graph, time_ms=30)[0],
        ):
            before, after = solver(quiet), solver(late)
            np.testing.assert_array_equal(before[:, :151], after[:, :151])
            self.assertFalse(np.array_equal(before[:, 151:], after[:, 151:]))

    def test_rejects_malformed_or_nonfinite_current_in_both_solvers(self):
        graph = build_sparse_network(np.arange(3), NetworkConfig(out_degree=0), 9)
        for current in (np.ones((3, 299)), np.ones((3, 1, 300)), np.full(3, np.nan)):
            batch = replace(self.batch, current=current)
            with self.assertRaises(ValueError):
                simulate_izhikevich_batch(batch, time_ms=30)
            with self.assertRaises(ValueError):
                simulate_connected_batch(batch, graph, time_ms=30)


class ResearchProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(time_ms=60, dt_ms=.2, current_min=15, current_max=30)
        self.conditions = {condition.name: condition for condition in make_protocol(self.config)}

    def test_held_out_currents_strictly_exclude_training_interval_and_profiles_differ(self):
        for name in ("current_below_range", "current_above_range"):
            cohort = generate_cohort(self.conditions[name], self.config, 12, 101)
            self.assertTrue(cohort.outside_recorded_drive_range.all())
            if name == "current_below_range":
                self.assertTrue(np.all(cohort.amplitudes < self.config.current_min))
            else:
                self.assertTrue(np.all(cohort.amplitudes > self.config.current_max))
            self.assertIsNotNone(self.conditions[name].held_out_reason)
        for name in ("pulse_onset_0.1", "ramp", "sine"):
            condition = self.conditions[name]
            drive = stimulus(np.array([15], dtype=np.float32), condition.profile, 300, .2)
            self.assertGreater(np.ptp(drive), 0)
            self.assertIsNotNone(condition.held_out_reason)

    def test_sweeps_pair_parameters_and_drives_and_noise_does_not_change_true_events(self):
        first = generate_cohort(self.conditions["reference"], self.config, 8, 101)
        noisy = generate_cohort(self.conditions["noise_10"], self.config, 8, 101)
        zero_network = generate_cohort(self.conditions["network_0"], self.config, 8, 101)
        repeated = generate_cohort(self.conditions["reference"], self.config, 8, 101)
        for name in first.parameters:
            np.testing.assert_array_equal(first.parameters[name], noisy.parameters[name])
        np.testing.assert_array_equal(first.amplitudes, noisy.amplitudes)
        np.testing.assert_array_equal(first.physical_spike_counts, noisy.physical_spike_counts)
        np.testing.assert_array_equal(first.traces, zero_network.traces)
        np.testing.assert_array_equal(first.physical_spike_counts, zero_network.physical_spike_counts)
        np.testing.assert_array_equal(first.traces, repeated.traces)
        self.assertFalse(np.array_equal(first.traces, noisy.traces))
        other_seed = generate_cohort(self.conditions["reference"], self.config, 8, 202)
        self.assertFalse(np.array_equal(first.amplitudes, other_seed.amplitudes))

    def test_quiet_examples_are_kept_and_labels_retain_parameter_family(self):
        condition = Condition("quiet", "test", 0, 0, 0, current_range=(-20, -10))
        cohort = generate_cohort(condition, self.config, 4, 101)
        self.assertEqual(cohort.traces.shape, (12, 300))
        self.assertEqual(cohort.physical_spike_counts.sum(), 0)
        np.testing.assert_array_equal(cohort.labels, np.repeat([0, 1, 2], 4))
        self.assertEqual(len({condition.name for condition in make_protocol(self.config)}), 23)


class TimingFeatureTests(unittest.TestCase):
    def test_contiguous_peaks_isi_features_and_missing_activity(self):
        traces = np.full((3, 20), -65., dtype=np.float32)
        traces[0, [1, 2, 5, 6, 13, 17]] = 30
        traces[2, 3] = np.nan
        features, counts, finite = extract_timing_features(traces, 1)
        np.testing.assert_array_equal(counts, [4, 0, 0])
        np.testing.assert_array_equal(finite, [True, True, False])
        np.testing.assert_allclose(features[0], [200, 4, np.std([4, 8, 4]) / np.mean([4, 8, 4]), 2 / 3, 1])
        np.testing.assert_array_equal(features[1:], 0)

    def test_centroids_use_training_scaling_and_do_not_refit_on_test_rows(self):
        features = np.zeros((6, 5))
        features[:, 0] = [0, 2, 10, 12, 20, 22]
        labels = np.repeat([0, 1, 2], 2)
        baseline = TimingCentroid.fit(features, labels, columns=(0,))
        before = baseline.to_dict()
        test = np.zeros((4, 5))
        test[:, 0] = [1, 11, 21, 1000]
        np.testing.assert_array_equal(baseline.predict(test), [0, 1, 2, 2])
        self.assertEqual(baseline.to_dict(), before)
        self.assertAlmostEqual(baseline.mean[0], 11)
        self.assertAlmostEqual(baseline.scale[0], features[:, 0].std())
        with self.assertRaises(ValueError):
            TimingCentroid.fit(features[:4], labels[:4])

    def test_metrics_count_missing_predictions_as_errors_and_use_all_rows(self):
        score = prediction_metrics(np.array([0, 1, 2, 2]), np.array([0, -1, 1, 2]))
        self.assertEqual(score["total"], 4)
        self.assertEqual(score["correct"], 2)
        self.assertEqual(score["accuracy"], .5)
        self.assertEqual(score["missing_predictions"], 1)
        self.assertEqual(score["confusion_matrix"], [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        self.assertIsNone(prediction_metrics(np.array([], dtype=int), np.array([], dtype=int))["accuracy"])

    def test_probability_scores_have_known_values_and_skip_nonfinite_output(self):
        labels = np.arange(3)
        perfect = probability_metrics(labels, np.eye(3))
        for metric in ("nll", "brier", "ece"):
            self.assertAlmostEqual(perfect[metric], 0)
        uniform = probability_metrics(labels, np.full((3, 3), 1 / 3))
        self.assertAlmostEqual(uniform["nll"], np.log(3))
        self.assertAlmostEqual(uniform["brier"], 2 / 3)
        self.assertAlmostEqual(uniform["ece"], 0)
        missing = probability_metrics(labels, np.full((3, 3), np.nan))
        self.assertEqual(missing["evaluated"], 0)
        self.assertIsNone(missing["nll"])
        with self.assertRaises(ValueError):
            probability_metrics(labels, np.full((3, 3), .5))


if __name__ == "__main__":
    unittest.main()
