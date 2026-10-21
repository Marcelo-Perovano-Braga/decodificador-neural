import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_inference_fusion_preserves_outputs_and_source_checkpoint(self):
        from copy import deepcopy
        import torch
        from models import NeuronClassifier1D

        torch.manual_seed(77)
        model = NeuronClassifier1D()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm1d):
                    layer.running_mean.normal_()
                    layer.running_var.uniform_(0.2, 2.0)
                    layer.weight.normal_()
                    layer.bias.normal_()
        state = deepcopy(model.state_dict())
        reference = deepcopy(model).eval()
        optimized = model.for_inference()
        self.assertTrue(model.training)
        self.assertFalse(optimized.training)
        self.assertTrue(any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model.modules()))
        self.assertFalse(any(isinstance(layer, torch.nn.BatchNorm1d) for layer in optimized.modules()))
        self.assertEqual(state.keys(), model.state_dict().keys())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        with torch.inference_mode():
            for steps in (513, 2000):
                values = torch.randn(7, 1, steps)
                torch.testing.assert_close(optimized(values), reference(values), rtol=1e-5, atol=1e-6)

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
