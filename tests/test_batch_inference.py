import importlib.util
import unittest
from concurrent.futures import CancelledError
from threading import Event

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class BatchInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    @staticmethod
    def options():
        import torch
        return dict(
            normalization_mean=0.0, normalization_std=1.0,
            class_codes=["RS", "CH", "FS"], class_names=["RS", "CH", "FS"],
            minimum_spikes=2, confidence_threshold=0.7, device=torch.device("cpu"),
        )

    @staticmethod
    def counting_model(cancel=None):
        import torch

        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, inputs):
                self.batch_sizes.append(inputs.shape[0])
                logits = torch.full((inputs.shape[0], 3), -5.0)
                logits.scatter_(1, inputs[:, 0, 0].long().reshape(-1, 1), 5.0)
                if cancel is not None:
                    cancel.set()
                return logits

        return FixtureModel()

    def test_batched_rows_remain_aligned_when_other_rows_are_rejected(self):
        from inference import classify_traces
        traces = np.zeros((7, 8), dtype=np.float32)
        traces[:, 0] = [2, 1, 0, 1, 2, 0, 1]
        traces[:, [2, 5]] = 30
        traces[1, 2:] = 0
        traces[6, 3] = np.nan
        currents = np.asarray([10, 10, 10, 10, 50, 10, 10])
        model = self.counting_model()
        results = classify_traces(
            traces, model, **self.options(), currents=currents,
            trained_current_range=(5.0, 30.0), batch_size=2,
        )
        self.assertEqual(model.batch_sizes, [2, 2])
        self.assertEqual([item.class_code for item in results], ["FS", None, "RS", "CH", None, "RS", None])
        self.assertIn("atividade insuficiente", results[1].reason)
        self.assertIn("fora do domínio", results[4].reason)
        self.assertIn("não finitos", results[6].reason)

    def test_population_cnn_inference_matches_single_neuron_inference(self):
        import torch
        from config import SimulationConfig
        from inference import classify_trace, classify_traces
        from models import NeuronClassifier1D
        from population import PopulationRequest, simulate_population
        torch.manual_seed(23)
        model = NeuronClassifier1D()
        population = simulate_population(PopulationRequest((2, 2, 2), current=15), SimulationConfig())
        options = self.options()
        options.update(normalization_mean=-53.0, normalization_std=21.0, confidence_threshold=0.0)
        batched = classify_traces(population.traces, model, **options, batch_size=4)
        singles = [classify_trace(trace, model, **options) for trace in population.traces]
        self.assertEqual([item.class_code for item in batched], [item.class_code for item in singles])
        np.testing.assert_allclose(
            [item.confidence for item in batched], [item.confidence for item in singles], atol=1e-6,
        )

    def test_cancel_stops_between_bounded_batches(self):
        from inference import classify_traces
        cancel = Event()
        model = self.counting_model(cancel)
        traces = np.zeros((130, 8), dtype=np.float32)
        traces[:, [2, 5]] = 30
        with self.assertRaises(CancelledError):
            classify_traces(traces, model, **self.options(), batch_size=16, cancelled=cancel.is_set)
        self.assertEqual(model.batch_sizes, [16])

    def test_silent_population_does_not_call_model(self):
        from inference import classify_traces
        model = self.counting_model()
        results = classify_traces(np.full((150, 8), -65, dtype=np.float32), model, **self.options())
        self.assertEqual(model.batch_sizes, [])
        self.assertEqual(len(results), 150)
        self.assertTrue(all(not result.determined for result in results))


if __name__ == "__main__":
    unittest.main()
