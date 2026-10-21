import unittest
from concurrent.futures import CancelledError
from threading import Event

from gui_support import GUI_AVAILABLE, application, wait_until
from population import PopulationRequest


@unittest.skipUnless(GUI_AVAILABLE, "Qt UI dependencies are not installed")
class PopulationWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = application()

    def runner(self, processor):
        from population_worker import PopulationRunner
        runner = PopulationRunner(processor)
        self.addCleanup(runner.shutdown)
        return runner

    def test_latest_request_wins_without_blocking_event_loop(self):
        from PyQt6.QtCore import QThread, QTimer
        from PyQt6.QtTest import QTest
        started, release = Event(), Event()
        calls, received, threads, ticks = [], [], [], []

        def processor(request, cancelled):
            calls.append(request.current)
            threads.append(QThread.currentThread())
            if request.current == 1:
                started.set()
                release.wait(2.0)  # Deliberately return a stale result even after cancellation.
            return request.current

        runner = self.runner(processor)
        runner.ready.connect(received.append)
        heartbeat = QTimer()
        heartbeat.setInterval(5)
        heartbeat.timeout.connect(lambda: ticks.append(1))
        heartbeat.start()
        self.addCleanup(heartbeat.stop)
        runner.submit(PopulationRequest(current=1))
        wait_until(started.is_set)
        for value in range(2, 101):
            runner.submit(PopulationRequest(current=value))
        QTest.qWait(40)
        self.assertGreaterEqual(len(ticks), 3)
        self.assertEqual(received, [])
        release.set()
        wait_until(lambda: received == [100])
        self.assertEqual(calls, [1, 100])
        self.assertTrue(all(thread != self.app.thread() for thread in threads))
        self.assertFalse(runner.busy)

    def test_invalidation_discards_results_before_another_request_is_submitted(self):
        started, release = Event(), Event()

        def processor(request, cancelled):
            started.set()
            release.wait(2.0)
            return request.current

        runner = self.runner(processor)
        received = []
        runner.ready.connect(received.append)
        runner.submit(PopulationRequest())
        wait_until(started.is_set)
        runner.invalidate()
        release.set()
        wait_until(lambda: not runner.busy)
        self.assertEqual(received, [])

    def test_error_is_reported_and_next_request_can_succeed(self):
        def processor(request, cancelled):
            if request.current == 1:
                raise RuntimeError("test failure")
            return request.current

        runner = self.runner(processor)
        errors, received = [], []
        runner.failed.connect(errors.append)
        runner.ready.connect(received.append)
        runner.submit(PopulationRequest(current=1))
        wait_until(lambda: errors == ["test failure"])
        runner.submit(PopulationRequest(current=2))
        wait_until(lambda: received == [2])

    def test_shutdown_cancels_running_work_and_drops_pending_request(self):
        started = Event()
        pause = Event()
        calls = []

        def processor(request, cancelled):
            calls.append(request.current)
            started.set()
            while not cancelled():
                pause.wait(0.001)
            raise CancelledError()

        runner = self.runner(processor)
        runner.submit(PopulationRequest(current=1))
        wait_until(started.is_set)
        runner.submit(PopulationRequest(current=2))
        runner.shutdown()
        self.assertFalse(runner._thread.isRunning())
        self.assertFalse(runner.busy)
        self.assertEqual(calls, [1])
        runner.shutdown()
        with self.assertRaises(RuntimeError):
            runner.submit(PopulationRequest())


if __name__ == "__main__":
    unittest.main()
