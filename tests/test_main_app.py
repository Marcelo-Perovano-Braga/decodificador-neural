import tempfile
import unittest
from pathlib import Path

import numpy as np

from gui_support import GUI_AVAILABLE, application, wait_until


@unittest.skipUnless(GUI_AVAILABLE, "Qt UI dependencies are not installed")
class PopulationInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from config import CLASS_CODES, CLASS_NAMES, SimulationConfig
        from models import NeuronClassifier1D
        cls.app = application()
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint_path = Path(cls.directory.name) / "test-classifier.pt"
        # A deterministic untrained fixture checks UI accounting, not scientific accuracy.
        model = NeuronClassifier1D()
        with torch.no_grad():
            model.classifier[-1].weight.zero_()
            model.classifier[-1].bias.copy_(torch.tensor([8.0, 0.0, 0.0]))
        torch.save({
            "model_state": model.state_dict(),
            "normalization": {"mean": -53.0, "std": 21.0},
            "simulation": SimulationConfig().to_dict(),
            "class_codes": list(CLASS_CODES), "class_names": list(CLASS_NAMES),
            "confidence_threshold": 0.7,
        }, cls.checkpoint_path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def window(self, with_model=True):
        from main_app import IzhikevichApp
        path = self.checkpoint_path if with_model else Path(self.directory.name) / "missing.pt"
        window = IzhikevichApp(checkpoint_path=path)
        self.addCleanup(window.close)
        return window

    @staticmethod
    def configure(window, counts):
        for spin, value in zip(window.count_spins, counts, strict=True):
            spin.setValue(value)
        window.variation_check.setChecked(False)
        window.slider_current.setValue(150)
        window._submit_request()
        wait_until(lambda: window.result is not None and window.result.population.request.counts == counts)

    def test_all_neurons_are_classified_and_selection_only_changes_the_view(self):
        window = self.window()
        self.configure(window, (2, 1, 1))
        result = window.result
        self.assertEqual(window.table_model.rowCount(), 4)
        self.assertEqual(result.statistics.classified, 4)
        self.assertEqual(result.statistics.correct, 2)
        self.assertEqual(window.counts_table.item(0, 1).text(), "4")
        self.assertEqual(window.confusion_table.item(1, 0).text(), "1")
        window.neuron_selector.setValue(4)
        np.testing.assert_array_equal(window.curve.yData, result.population.traces[3])
        self.assertIn("Fast Spiking", window.trace_subtitle.text())
        window._select_table_neuron(window.table_model.index(2, 0))
        self.assertEqual(window.neuron_selector.value(), 3)
        self.assertIs(window.result, result)
        self.assertFalse(window.runner.busy)

    def test_out_of_domain_stimulus_is_indeterminate_for_every_neuron(self):
        window = self.window()
        self.configure(window, (1, 1, 1))
        window.slider_current.setValue(0)
        window._submit_request()
        wait_until(lambda: window.result.population.request.current == 0)
        stats = window.result.statistics
        self.assertEqual(stats.classified, 0)
        self.assertEqual(stats.indeterminate, 3)
        self.assertIsNone(stats.accuracy)
        self.assertEqual(stats.confusion_matrix[:, -1].tolist(), [1, 1, 1])
        self.assertTrue(all("fora do domínio" in item.reason for item in window.result.predictions))

    def test_empty_population_clears_stale_data_and_can_recover(self):
        window = self.window()
        self.configure(window, (1, 1, 1))
        for spin in window.count_spins:
            spin.setValue(0)
        window._submit_request()
        self.assertIsNone(window.result)
        self.assertEqual(window.table_model.rowCount(), 0)
        self.assertIn("pelo menos um", window.status_label.text())
        self.assertFalse(window.neuron_selector.isEnabled())
        self.configure(window, (1, 0, 0))
        self.assertEqual(window.result.population.size, 1)

    def test_missing_checkpoint_still_simulates_without_fabricating_predictions(self):
        window = self.window(with_model=False)
        self.configure(window, (1, 2, 1))
        self.assertIsNone(window.result.predictions)
        self.assertEqual(window.result.statistics.unclassified, 4)
        self.assertEqual(window.result.statistics.indeterminate, 0)
        self.assertEqual(window.table_model.data(window.table_model.index(0, 5)), "Sem modelo")
        self.assertEqual(window.counts_table.item(0, 1).text(), "—")
        self.assertEqual(window.confusion_table.item(0, 0).text(), "—")

    def test_connected_controls_results_and_switch_back_to_independent(self):
        window = self.window()
        self.configure(window, (3, 2, 3))
        window.network_check.setChecked(True)
        window.out_degree_spin.setValue(4)
        window._submit_request()
        wait_until(lambda: window.result.population.request.network is not None)
        result = window.result
        graph = result.population.network_activity.network
        self.assertEqual(graph.synapse_count, 32)
        self.assertEqual(result.statistics.total, 8)
        self.assertEqual(window.table_model.rowCount(), 8)
        self.assertIn("experimental", window.status_label.text())
        self.assertEqual(window.accuracy_card.title_label.text(), "Concordância experimental")
        self.assertIsNotNone(window.rate_curve.yData)
        self.assertIn("32 sinapses", window.network_detail.text())
        window.neuron_selector.setValue(8)
        self.assertIn("Neurônio 8", window.connection_detail.text())
        np.testing.assert_array_equal(window.curve.yData, result.population.traces[7])
        self.assertIs(window.result, result)
        window.network_check.setChecked(False)
        window._submit_request()
        wait_until(lambda: window.result.population.request.network is None)
        self.assertIsNone(window.result.population.network_activity)
        self.assertIn("Modo independente", window.network_detail.text())
        self.assertTrue(window.rate_curve.yData is None or not window.rate_curve.yData.size)
        self.assertEqual(window.accuracy_card.title_label.text(), "Acerto dos classificados")

    def test_population_budget_error_can_recover_to_independent_mode(self):
        window = self.window(with_model=False)
        window.update_timer.stop()
        for spin in window.count_spins:
            spin.setValue(1_000_000)
        window.network_check.setChecked(True)
        window._submit_request()
        self.assertIsNone(window.result)
        self.assertIn("limite", window.status_label.text())
        window.network_check.setChecked(False)
        self.configure(window, (2, 1, 1))
        self.assertEqual(window.result.population.size, 4)

    def test_million_neuron_controls_submit_the_billion_synapse_boundary(self):
        from unittest.mock import patch
        window = self.window(with_model=False)
        # Capture the request; this test must not allocate a billion targets.
        with patch.object(window.runner, "submit") as submit:
            for spin, value in zip(window.count_spins, (333_333, 333_333, 333_334), strict=True):
                self.assertEqual(spin.maximum(), 1_000_000)
                spin.setValue(value)
            self.assertEqual(window.out_degree_spin.value(), 100)
            window.network_check.setChecked(True)
            window.out_degree_spin.setValue(1000)
            window._submit_request()
            request = submit.call_args.args[0]
            self.assertEqual(sum(request.counts), 1_000_000)
            self.assertEqual(request.network.out_degree, 1000)


if __name__ == "__main__":
    unittest.main()
