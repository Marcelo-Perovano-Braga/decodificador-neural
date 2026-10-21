import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_classifier_output_shape(self):
        import torch

        from models import NeuronClassifier1D

        model = NeuronClassifier1D(num_classes=3)
        output = model(torch.zeros(4, 1, 2000))
        self.assertEqual(tuple(output.shape), (4, 3))

    def test_classifier_rejects_wrong_shape(self):
        import torch

        from models import NeuronClassifier1D

        model = NeuronClassifier1D(num_classes=3)
        with self.assertRaises(ValueError):
            model(torch.zeros(4, 2000))

    def test_silent_trace_is_indeterminate(self):
        import numpy as np
        import torch

        from inference import classify_trace
        from models import NeuronClassifier1D

        result = classify_trace(
            np.full(2000, -65.0, dtype=np.float32),
            NeuronClassifier1D(num_classes=3),
            normalization_mean=-60.0,
            normalization_std=10.0,
            class_codes=["RS", "CH", "FS"],
            class_names=["Regular Spiking", "Chattering", "Fast Spiking"],
            minimum_spikes=2,
            confidence_threshold=0.70,
            device=torch.device("cpu"),
            current=5.0,
            trained_current_range=(5.0, 30.0),
        )
        self.assertFalse(result.determined)
        self.assertIn("atividade insuficiente", result.reason)


if __name__ == "__main__":
    unittest.main()
