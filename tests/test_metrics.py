import unittest

import numpy as np

from metrics import classification_report, confusion_matrix


class MetricsTests(unittest.TestCase):
    def test_confusion_matrix_orientation(self):
        targets = np.asarray([0, 0, 1, 1, 2])
        predictions = np.asarray([0, 1, 1, 1, 2])
        matrix = confusion_matrix(targets, predictions, 3)
        np.testing.assert_array_equal(
            matrix,
            np.asarray([[1, 1, 0], [0, 2, 0], [0, 0, 1]]),
        )

    def test_classification_report(self):
        report = classification_report(
            np.asarray([0, 1, 2, 2]),
            np.asarray([0, 1, 1, 2]),
            ["RS", "CH", "FS"],
        )
        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertEqual(report["total"], 4)
        self.assertEqual(report["per_class"]["FS"]["support"], 2)


if __name__ == "__main__":
    unittest.main()

