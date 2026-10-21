import unittest

from benchmark_models import benchmark_models
from config import SimulationConfig


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_has_both_models_and_positive_timings(self):
        report = benchmark_models(
            samples=6,
            repeats=1,
            seed=3,
            config=SimulationConfig(time_ms=20.0, dt_ms=0.2),
        )
        self.assertIn("lif", report)
        self.assertIn("izhikevich", report)
        self.assertGreater(report["lif"]["median_seconds"], 0)
        self.assertGreater(report["izhikevich"]["median_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
