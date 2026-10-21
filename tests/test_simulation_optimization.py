"""Numerical compatibility with the original float32 stepping and RNG stream."""

import unittest
from concurrent.futures import CancelledError
from unittest.mock import patch

import numpy as np

from config import CLASS_CODES, IZHIKEVICH_PARAMETERS
from simulation import (
    IzhikevichBatch,
    add_observation_noise,
    count_spikes,
    simulate_izhikevich_batch,
    simulate_lif_batch,
)


def original_izhikevich(batch, steps, dt):
    """Frozen pre-optimization reference, including spike clipping order."""
    a, b, c, d, current = (
        np.asarray(value, dtype=np.float32)
        for value in (batch.a, batch.b, batch.c, batch.d, batch.current)
    )
    traces = np.empty((a.size, steps), dtype=np.float32)
    traces[:, 0] = -65.0
    recovery = b * traces[:, 0]
    for step in range(steps - 1):
        voltage = traces[:, step]
        spiking = voltage >= 30.0
        if np.any(spiking):
            voltage[spiking] = 30.0
        dv = 0.04 * voltage * voltage + 5.0 * voltage + 140.0 - recovery + current
        du = a * (b * voltage - recovery)
        next_voltage = voltage + dv * dt
        next_recovery = recovery + du * dt
        if np.any(spiking):
            next_voltage[spiking] = c[spiking]
            next_recovery[spiking] = recovery[spiking] + d[spiking]
        traces[:, step + 1] = next_voltage
        recovery = next_recovery
    return traces


class SimulationOptimizationTests(unittest.TestCase):
    def batch(self, size=37):
        rng = np.random.default_rng(71)
        codes = [CLASS_CODES[i % 3] for i in range(size)]
        parameters = {
            name: np.asarray([
                IZHIKEVICH_PARAMETERS[code][name] * rng.normal(1.0, 0.03)
                for code in codes
            ], dtype=np.float32)
            for name in ("a", "b", "c", "d")
        }
        return IzhikevichBatch(**parameters, current=np.linspace(-2, 35, size, dtype=np.float32))

    def test_izhikevich_is_bitwise_identical_to_original_solver(self):
        batch = self.batch()
        for duration, dt in ((200.0, 0.1), (90.0, 0.2)):
            with self.subTest(duration=duration, dt=dt):
                expected = original_izhikevich(batch, round(duration / dt), dt)
                actual = simulate_izhikevich_batch(batch, time_ms=duration, dt_ms=dt)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(count_spikes(actual), count_spikes(expected))

    def test_lif_is_bitwise_identical_to_original_solver(self):
        drives = np.linspace(0, 40, 37, dtype=np.float32)
        expected = np.empty((drives.size, 2000), dtype=np.float32)
        expected[:, 0] = -70
        for step in range(1999):
            voltage = expected[:, step]
            next_voltage = voltage + ((-70.0 - voltage + drives) / 10.0) * 0.1
            spiking = next_voltage >= -50.0
            expected[:, step + 1] = np.where(spiking, -70.0, next_voltage)
            expected[spiking, step] = -50.0
        np.testing.assert_array_equal(simulate_lif_batch(drives), expected)

    def test_blocked_noise_preserves_values_and_rng_state(self):
        source_rng = np.random.default_rng(29)
        for shape in ((1025, 37), (2049,), (), (513, 2, 3)):
            values = source_rng.uniform(-120, 45, size=shape).astype(np.float32)
            for order in ("C", "F"):
                for std in (0.0, 0.2):
                    with self.subTest(shape=shape, order=order, std=std):
                        values = np.array(values, order=order, copy=True)
                        before = values.copy()
                        expected_rng = np.random.default_rng(19)
                        actual_rng = np.random.default_rng(19)
                        expected = values.copy() if std == 0 else np.clip(
                            values + expected_rng.normal(0.0, std, size=shape).astype(np.float32),
                            -110.0, 35.0,
                        ).astype(np.float32)
                        actual = add_observation_noise(values, std, actual_rng)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(values, before)
                        self.assertFalse(np.shares_memory(actual, values))
                        np.testing.assert_array_equal(actual_rng.random(7), expected_rng.random(7))
                        inplace = add_observation_noise(values, std, np.random.default_rng(19), out=values)
                        self.assertIs(inplace, values)
                        np.testing.assert_array_equal(inplace, expected)

    def test_blocked_counts_match_full_array_with_nonfinite_values(self):
        values = np.random.default_rng(8).uniform(-80, 35, size=(1025, 73)).astype(np.float32)
        values[0, :3] = [np.nan, np.inf, -np.inf]
        for order in ("C", "F"):
            traces = np.array(values, order=order)
            np.testing.assert_array_equal(count_spikes(traces), np.count_nonzero(traces >= 29, axis=1))
        self.assertEqual(count_spikes(np.empty((0, 20))).size, 0)

    def test_solver_cancels_before_finishing_all_timesteps(self):
        checks = 0

        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 4

        with patch("simulation.np.any", wraps=np.any) as reduction:
            with self.assertRaises(CancelledError):
                simulate_izhikevich_batch(self.batch(), cancelled=cancelled)
            self.assertGreater(reduction.call_count, 0)
            self.assertLess(reduction.call_count, 100)

    def test_noise_cancels_between_blocks(self):
        values = np.zeros((1025, 20), dtype=np.float32)
        target = np.full_like(values, np.nan)
        checks = 0

        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 3

        with self.assertRaises(CancelledError):
            add_observation_noise(values, 0.2, np.random.default_rng(42), out=target, cancelled=cancelled)
        self.assertTrue(np.isfinite(target[:512]).all())
        self.assertTrue(np.isnan(target[512:]).all())

    def test_noise_rejects_partially_overlapping_output(self):
        backing = np.zeros((1026, 20), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "partial overlap"):
            add_observation_noise(backing[:-1], 0.2, np.random.default_rng(42), out=backing[1:])


if __name__ == "__main__":
    unittest.main()
