import unittest
from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np

from config import SimulationConfig
from network import NetworkConfig, SparseNetwork, build_sparse_network, simulate_connected_batch
from population import PopulationRequest, build_raster_preview, simulate_population
from simulation import IzhikevichBatch, simulate_izhikevich_batch


def two_neurons(weight=0.5, delay_ms=1.0):
    """Neuron 0 drives a quiet neuron 1; neuron 1 has zero outgoing strength."""
    config = NetworkConfig(out_degree=1, weight=abs(weight), delay_ms=delay_ms)
    network = SparseNetwork(
        config, np.asarray([[1], [0]], dtype=np.int32),
        np.asarray([weight, 0], dtype=np.float32), np.asarray([weight < 0, False]),
        np.ones(2, dtype=np.int32), np.asarray([[1, 0, 0], [1, 0, 0], [0, 0, 0]]),
    )
    batch = IzhikevichBatch(
        np.asarray([0.02, 0.02]), np.asarray([0.2, 0.2]),
        np.asarray([-65, -65]), np.asarray([8, 8]), np.asarray([15, 0]),
    )
    return batch, network


class SparseNetworkTests(unittest.TestCase):
    def test_topology_has_exact_unique_out_degree_and_reproducible_roles(self):
        labels = np.repeat([0, 1, 2], 70)
        config = NetworkConfig(out_degree=40)
        first = build_sparse_network(labels, config, 7)
        second = build_sparse_network(labels, replace(config, weight=1.0), 7)
        np.testing.assert_array_equal(first.targets, second.targets)
        np.testing.assert_array_equal(first.inhibitory, second.inhibitory)
        np.testing.assert_array_equal(second.source_weights, first.source_weights * 2)
        self.assertEqual(first.targets.shape, (210, 40))
        self.assertEqual(first.synapse_count, 8400)
        self.assertEqual(first.inhibitory.sum(), 42)
        self.assertFalse(np.any(first.targets == np.arange(210)[:, None]))
        self.assertTrue(all(np.unique(row).size == 40 for row in first.targets))
        np.testing.assert_array_equal(first.in_degrees, np.bincount(first.targets.ravel(), minlength=210))
        expected = np.zeros((3, 3), dtype=np.int64)
        for source, row in enumerate(first.targets):
            for target in row:
                expected[labels[source], labels[target]] += 1
        np.testing.assert_array_equal(first.class_connections, expected)
        self.assertFalse(np.array_equal(first.targets, build_sparse_network(labels, config, 8).targets))
        # Each intrinsic class contains both synaptic roles in this seeded example.
        for label in range(3):
            self.assertTrue(np.any(first.inhibitory[labels == label]))
            self.assertTrue(np.any(~first.inhibitory[labels == label]))

    def test_single_neuron_and_small_network_clamp_without_self_connections(self):
        for size in (1, 2, 8):
            network = build_sparse_network(np.zeros(size, dtype=int), NetworkConfig(), 42)
            self.assertEqual(network.out_degree, size - 1)
            self.assertEqual(network.synapse_count, size * (size - 1))

    def test_zero_strength_and_zero_degree_preserve_independent_traces_and_noise(self):
        config = SimulationConfig(time_ms=30.0)
        original = PopulationRequest((4, 3, 5), current=15, seed=7)
        baseline = simulate_population(original, config)
        for network in (NetworkConfig(weight=0), NetworkConfig(out_degree=0)):
            population = simulate_population(replace(original, network=network), config)
            np.testing.assert_array_equal(population.traces, baseline.traces)
            np.testing.assert_array_equal(population.labels, baseline.labels)

    def test_delayed_excitatory_and_inhibitory_currents_have_correct_sign_and_decay(self):
        for weight in (0.5, -0.5):
            with self.subTest(weight=weight):
                batch, network = two_neurons(weight=weight, delay_ms=1.03)
                traces, activity = simulate_connected_batch(batch, network, time_ms=30.0, dt_ms=0.1)
                baseline = simulate_izhikevich_batch(batch, time_ms=30.0, dt_ms=0.1)
                emitted = activity.spike_steps[activity.spike_neurons == 0]
                arrival = int(emitted[0]) + 11
                self.assertAlmostEqual(activity.effective_delay_ms, 1.1)
                np.testing.assert_array_equal(traces[:, :arrival + 1], baseline[:, :arrival + 1])
                self.assertEqual(activity.mean_synaptic_current[:arrival].sum(), 0)
                self.assertAlmostEqual(float(activity.mean_synaptic_current[arrival]), weight / 2)
                self.assertAlmostEqual(
                    float(activity.mean_synaptic_current[arrival + 1]),
                    weight / 2 * np.exp(-0.1 / 5.0), places=6,
                )
                self.assertEqual(np.sign(traces[1, arrival + 1] - baseline[1, arrival + 1]), np.sign(weight))
                self.assertEqual(activity.delivered_events, int(np.count_nonzero(activity.spike_steps + 11 < 300)))

    def test_sparse_solver_matches_dense_delayed_reference_including_colliding_targets(self):
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        config = NetworkConfig(out_degree=3, weight=0.7, delay_ms=0.3, tau_ms=3)
        graph = build_sparse_network(labels, config, 9)
        batch = IzhikevichBatch(
            np.asarray([.02, .02, .02, .02, .1, .1]), np.full(6, .2),
            np.asarray([-65, -65, -50, -50, -65, -65]), np.asarray([8, 8, 2, 2, 2, 2]),
            np.asarray([15, 15, 20, 20, 25, 25]),
        )
        actual, activity = simulate_connected_batch(batch, graph, time_ms=20, dt_ms=.1)
        expected = np.empty_like(actual)
        expected[:, 0] = -65
        a, b, c, d, external = (value.astype(np.float32) for value in (
            batch.a, batch.b, batch.c, batch.d, batch.current,
        ))
        recovery = b * expected[:, 0]
        current = np.zeros(6, dtype=np.float32)
        history = np.zeros((200, 6), dtype=bool)
        expected_current = np.empty(200)
        for step in range(200):
            if step:
                current *= np.float32(np.exp(-.1 / 3))
            if step >= 3:
                for source in np.flatnonzero(history[step - 3]):
                    for target in graph.targets[source]:
                        current[target] += graph.source_weights[source]
            expected_current[step] = current.mean()
            voltage = expected[:, step]
            fired = voltage >= 30
            history[step] = fired
            if step == 199:
                break
            voltage[fired] = 30
            dv = .04 * voltage * voltage + 5 * voltage + 140 - recovery + external
            dv += current
            du = a * (b * voltage - recovery)
            expected[:, step + 1] = voltage + dv * .1
            next_recovery = recovery + du * .1
            expected[fired, step + 1] = c[fired]
            next_recovery[fired] = recovery[fired] + d[fired]
            recovery = next_recovery
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(activity.mean_synaptic_current, expected_current)
        np.testing.assert_array_equal(activity.spike_counts, history.sum(axis=0))
        steps, neurons = np.nonzero(history)
        np.testing.assert_array_equal(activity.spike_steps, steps)
        np.testing.assert_array_equal(activity.spike_neurons, neurons)

    def test_delay_longer_than_window_does_not_wrap_into_earlier_samples(self):
        batch, graph = two_neurons(delay_ms=20)
        connected, activity = simulate_connected_batch(batch, graph, time_ms=10)
        baseline = simulate_izhikevich_batch(batch, time_ms=10)
        np.testing.assert_array_equal(connected, baseline)
        self.assertGreater(activity.spike_neurons.size, 0)
        self.assertEqual(activity.delivered_events, 0)
        self.assertEqual(activity.mean_synaptic_current.sum(), 0)

    def test_observation_noise_does_not_create_synaptic_spikes_or_change_network_activity(self):
        request = PopulationRequest((3, 2, 4), network=NetworkConfig(out_degree=3))
        clean = simulate_population(request, SimulationConfig(time_ms=23.7, observation_noise_std=0))
        noisy = simulate_population(request, SimulationConfig(time_ms=23.7, observation_noise_std=100))
        np.testing.assert_array_equal(clean.spike_counts, noisy.spike_counts)
        np.testing.assert_array_equal(clean.network_activity.spike_steps, noisy.network_activity.spike_steps)
        np.testing.assert_array_equal(clean.network_activity.mean_synaptic_current, noisy.network_activity.mean_synaptic_current)
        self.assertFalse(np.array_equal(clean.traces, noisy.traces))
        preview = build_raster_preview(noisy, None, limit=4)
        selected = np.linspace(0, 8, 4, dtype=int)
        expected_count = np.isin(clean.network_activity.spike_neurons, selected).sum()
        self.assertEqual(sum(series.times.size for series in preview.series), expected_count)
        activity = noisy.network_activity
        widths = np.minimum(5, 23.7 - np.arange(activity.rate_hz.size) * 5)
        self.assertAlmostEqual(float(np.sum(activity.rate_hz * 9 * widths / 1000)), int(noisy.spike_counts.sum()))

    def test_rejects_invalid_parameters_and_oversized_network_before_allocation(self):
        for kwargs in ({"out_degree": -1}, {"out_degree": 1001}, {"weight": float("nan")},
                       {"tau_ms": 0}, {"delay_ms": 0}, {"inhibitory_fraction": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NetworkConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "limite"):
            PopulationRequest((100_000, 100_000, 100_000), network=NetworkConfig())

    def test_cancellation_during_topology_and_stepping(self):
        checks = 0

        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 3

        with self.assertRaises(CancelledError):
            build_sparse_network(np.zeros(3000, dtype=int), NetworkConfig(), 42, cancelled=cancelled)
        batch, graph = two_neurons()
        checks = 0
        with self.assertRaises(CancelledError):
            simulate_connected_batch(batch, graph, cancelled=cancelled)


if __name__ == "__main__":
    unittest.main()
