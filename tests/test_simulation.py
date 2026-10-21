import unittest

import numpy as np

from simulation import (
    IzhikevichBatch,
    count_spikes,
    simulate_izhikevich_batch,
    simulate_izhikevich_single,
    simulate_lif_batch,
)


class SimulationTests(unittest.TestCase):
    def test_izhikevich_batch_shape_and_finite_values(self):
        batch = IzhikevichBatch(
            a=np.asarray([0.02, 0.10], dtype=np.float32),
            b=np.asarray([0.20, 0.20], dtype=np.float32),
            c=np.asarray([-65.0, -65.0], dtype=np.float32),
            d=np.asarray([8.0, 2.0], dtype=np.float32),
            current=np.asarray([10.0, 10.0], dtype=np.float32),
        )
        traces = simulate_izhikevich_batch(batch, time_ms=50.0, dt_ms=0.1)
        self.assertEqual(traces.shape, (2, 500))
        self.assertTrue(np.isfinite(traces).all())

    def test_single_matches_batch(self):
        timeline, single = simulate_izhikevich_single(
            0.02, 0.20, -65.0, 8.0, 12.0, time_ms=100.0, dt_ms=0.1
        )
        batch = IzhikevichBatch(
            a=np.asarray([0.02]),
            b=np.asarray([0.20]),
            c=np.asarray([-65.0]),
            d=np.asarray([8.0]),
            current=np.asarray([12.0]),
        )
        batched = simulate_izhikevich_batch(batch, time_ms=100.0, dt_ms=0.1)[0]
        np.testing.assert_allclose(single, batched)
        self.assertEqual(timeline.shape, single.shape)

    def test_active_trace_has_detectable_spikes(self):
        _, trace = simulate_izhikevich_single(
            0.02, 0.20, -65.0, 8.0, 15.0, time_ms=200.0, dt_ms=0.1
        )
        self.assertGreaterEqual(int(count_spikes(trace)[0]), 2)

    def test_lif_batch_shape(self):
        traces = simulate_lif_batch(
            np.asarray([21.0, 30.0], dtype=np.float32),
            time_ms=50.0,
            dt_ms=0.1,
        )
        self.assertEqual(traces.shape, (2, 500))
        self.assertTrue(np.isfinite(traces).all())


if __name__ == "__main__":
    unittest.main()

