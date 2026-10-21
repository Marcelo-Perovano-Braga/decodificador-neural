"""Small event-loop helpers shared by offscreen Qt integration tests."""

import importlib.util
import os
from time import monotonic


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
GUI_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "PyQt6", "pyqtgraph", "qfluentwidgets",
))


def application():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait_until(predicate, timeout=5.0):
    from PyQt6.QtTest import QTest
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        QTest.qWait(5)
    if not predicate():
        raise AssertionError("Timed out waiting for the Qt worker or UI.")
