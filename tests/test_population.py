import unittest
from concurrent.futures import CancelledError
from types import SimpleNamespace

import numpy as np

from config import CLASS_CODES, IZHIKEVICH_PARAMETERS, SimulationConfig
from population import (
    MAX_NEURONS_PER_CLASS,
    MAX_NEURONS_TOTAL,
    Population,
    PopulationRequest,
    build_raster_preview,
    simulate_population,
    summarize_population,
)
from simulation import simulate_izhikevich_single


class PopulationTests(unittest.TestCase):
    def test_unvaried_population_matches_individual_simulations(self):
        config = SimulationConfig(time_ms=60.0, dt_ms=0.1)
        request = PopulationRequest((2, 1, 2), current=15.0, varied=False)
        population = simulate_population(request, config)
        np.testing.assert_array_equal(population.labels, [0, 0, 1, 2, 2])
        self.assertEqual(population.traces.shape, (5, 600))
        for index, label in enumerate(population.labels):
            params = IZHIKEVICH_PARAMETERS[CLASS_CODES[label]]
            timeline, trace = simulate_izhikevich_single(
                **params, current=15.0, time_ms=60.0, dt_ms=0.1,
            )
            np.testing.assert_array_equal(population.traces[index], trace)
            np.testing.assert_allclose(population.timeline, timeline)

    def test_variability_is_reproducible_and_distinguishes_individuals(self):
        config = SimulationConfig(time_ms=60.0)
        request = PopulationRequest((3, 0, 0), seed=7)
        first = simulate_population(request, config)
        second = simulate_population(request, config)
        np.testing.assert_array_equal(first.traces, second.traces)
        self.assertFalse(np.array_equal(first.traces[0], first.traces[1]))
        other = simulate_population(PopulationRequest((3, 0, 0), seed=8), config)
        self.assertFalse(np.array_equal(first.traces, other.traces))

    @staticmethod
    def example_population():
        return Population(
            PopulationRequest((2, 1, 2)), 10.0,
            np.full((5, 10), -65, dtype=np.float32),
            np.asarray([0, 0, 1, 2, 2]), np.asarray([2, 3, 4, 5, 6]),
        )

    def test_statistics_separate_coverage_accuracy_and_undetermined_hypotheses(self):
        population = self.example_population()
        results = [
            SimpleNamespace(determined=True, class_code="RS", confidence=0.9),
            SimpleNamespace(determined=False, class_code="CH", confidence=0.4),
            SimpleNamespace(determined=True, class_code="FS", confidence=0.8),
            SimpleNamespace(determined=True, class_code="FS", confidence=0.7),
            SimpleNamespace(determined=False, class_code=None, confidence=0.0),
        ]
        stats = summarize_population(population, results)
        self.assertEqual(stats.expected_counts, {"RS": 2, "CH": 1, "FS": 2})
        self.assertEqual(stats.predicted_counts, {"RS": 1, "CH": 0, "FS": 2})
        self.assertEqual((stats.classified, stats.correct, stats.indeterminate), (3, 2, 2))
        self.assertAlmostEqual(stats.coverage, 3 / 5)
        self.assertAlmostEqual(stats.accuracy, 2 / 3)
        self.assertAlmostEqual(stats.mean_confidence, 0.8)
        self.assertEqual(stats.total_spikes, 20)
        self.assertEqual(stats.mean_rate_hz, 40.0)
        np.testing.assert_array_equal(stats.confusion_matrix, [[1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]])

    def test_missing_model_is_not_reported_as_wrong_or_indeterminate(self):
        stats = summarize_population(self.example_population(), None)
        self.assertEqual(stats.unclassified, 5)
        self.assertEqual(stats.indeterminate, 0)
        self.assertIsNone(stats.accuracy)
        self.assertIsNone(stats.coverage)
        self.assertEqual(stats.confusion_matrix.sum(), 0)

    def test_raster_is_bounded_and_uses_predictions_not_expected_classes(self):
        traces = np.full((300, 20), -65, dtype=np.float32)
        traces[:, [4, 12]] = 30
        population = Population(
            PopulationRequest((100, 100, 100)), 0.1, traces,
            np.repeat([0, 1, 2], 100), np.full(300, 2),
        )
        results = [SimpleNamespace(determined=True, class_code="FS")] * 300
        preview = build_raster_preview(population, results, limit=17)
        self.assertEqual(preview.neuron_count, 17)
        fs = next(series for series in preview.series if series.class_code == "FS")
        self.assertEqual(fs.times.size, 34)
        np.testing.assert_array_equal(np.unique(fs.neuron_ids), np.linspace(0, 299, 17, dtype=int) + 1)
        self.assertEqual(sum(series.times.size for series in preview.series if series.class_code != "FS"), 0)
        self.assertEqual(summarize_population(population, [
            SimpleNamespace(determined=True, class_code="FS", confidence=0.9)
        ] * 300).classified, 300)

    def test_population_rejects_empty_negative_and_oversized_counts(self):
        for counts in [(0, 0, 0), (-1, 1, 1), (MAX_NEURONS_PER_CLASS + 1, 0, 0), (MAX_NEURONS_TOTAL, 1, 0)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                PopulationRequest(counts)

    def test_population_accepts_a_million_total_in_balanced_and_single_class_requests(self):
        # Validate the boundary without allocating voltage traces or topology.
        for counts in ((333_333, 333_333, 333_334), (1_000_000, 0, 0), (0, 1_000_000, 0), (0, 0, 1_000_000)):
            self.assertEqual(sum(PopulationRequest(counts).counts), MAX_NEURONS_TOTAL)

    def test_cancelled_population_does_not_start(self):
        with self.assertRaises(CancelledError):
            simulate_population(PopulationRequest(), SimulationConfig(), cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
