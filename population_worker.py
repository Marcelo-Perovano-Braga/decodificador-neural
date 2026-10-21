"""One background worker with a single replaceable pending population request."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from threading import Event
from time import perf_counter

import numpy as np
import torch
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from torch import nn

from config import SimulationConfig
from inference import InferenceResult, classify_traces
from population import (
    Population,
    PopulationRequest,
    PopulationStatistics,
    RasterPreview,
    build_raster_preview,
    check_cancelled,
    simulate_population,
    summarize_population,
)


@dataclass(frozen=True)
class PopulationResult:
    population: Population
    predictions: tuple[InferenceResult, ...] | None
    statistics: PopulationStatistics
    raster: RasterPreview
    elapsed_ms: float


def process_population(
    request: PopulationRequest,
    cancelled: Callable[[], bool],
    *,
    config: SimulationConfig,
    model: nn.Module | None,
    checkpoint: dict[str, object] | None,
    device: torch.device,
) -> PopulationResult:
    started = perf_counter()
    population = simulate_population(request, config, cancelled=cancelled)
    predictions = None
    if model is not None and checkpoint is not None:
        normalization = checkpoint["normalization"]
        predictions = tuple(classify_traces(
            population.traces,
            model,
            normalization_mean=float(normalization["mean"]),
            normalization_std=float(normalization["std"]),
            class_codes=checkpoint["class_codes"],
            class_names=checkpoint["class_names"],
            minimum_spikes=config.minimum_spikes,
            confidence_threshold=float(checkpoint.get("confidence_threshold", 0.70)),
            device=device,
            currents=np.full(population.size, request.current, dtype=np.float32),
            trained_current_range=(config.current_min, config.current_max),
            cancelled=cancelled,
        ))
    check_cancelled(cancelled)
    statistics = summarize_population(population, predictions)
    raster = build_raster_preview(population, predictions)
    check_cancelled(cancelled)
    return PopulationResult(
        population, predictions, statistics, raster, (perf_counter() - started) * 1000.0,
    )


@dataclass(frozen=True)
class _Task:
    revision: int
    request: PopulationRequest
    cancel: Event = field(default_factory=Event)


class _Worker(QObject):
    completed = pyqtSignal(int, object, object)

    def __init__(self, processor: Callable) -> None:
        super().__init__()
        self.processor = processor

    @pyqtSlot(object)
    def process(self, task: _Task) -> None:
        result = error = None
        try:
            result = self.processor(task.request, task.cancel.is_set)
        except CancelledError:
            pass
        except Exception as exc:
            error = str(exc)
        self.completed.emit(task.revision, result, error)


class PopulationRunner(QObject):
    """Serialize model access and suppress results from obsolete settings.

    Submission never creates another thread or queues more than the newest
    pending request. All public methods and result signals belong to the GUI
    thread. Only the supplied processor executes in the worker thread.
    """

    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    busyChanged = pyqtSignal(bool)
    _execute = pyqtSignal(object)

    def __init__(self, processor: Callable, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._revision = 0
        self._active: _Task | None = None
        self._pending: PopulationRequest | None = None
        self._closed = False
        self._thread = QThread(self)
        self._worker = _Worker(processor)
        self._worker.moveToThread(self._thread)
        self._execute.connect(self._worker.process)
        self._worker.completed.connect(self._on_completed)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._active is not None

    def invalidate(self) -> None:
        self._revision += 1
        self._pending = None
        if self._active is not None:
            self._active.cancel.set()

    def submit(self, request: PopulationRequest) -> None:
        if self._closed:
            raise RuntimeError("The population runner is closed.")
        self.invalidate()
        self._pending = request
        if self._active is None:
            self._start_pending()

    def _start_pending(self) -> None:
        request, self._pending = self._pending, None
        if request is None or self._closed:
            return
        self._active = _Task(self._revision, request)
        self.busyChanged.emit(True)
        self._execute.emit(self._active)

    @pyqtSlot(int, object, object)
    def _on_completed(self, revision: int, result: object, error: object) -> None:
        if self._closed or self._active is None or self._active.revision != revision:
            return
        self._active = None
        if self._pending is not None:
            self._start_pending()
        else:
            self.busyChanged.emit(False)
        if revision == self._revision:
            if error is not None:
                self.failed.emit(str(error))
            elif result is not None:
                self.ready.emit(result)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.invalidate()
        self._thread.quit()
        # The processor checks cancellation between bounded inference batches.
        # Wait before destroying its model/QObjects; never terminate the thread.
        self._thread.wait()
        self._active = None
