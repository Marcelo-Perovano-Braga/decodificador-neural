"""Independent-neuron populations and statistics for the desktop demonstration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from config import CLASS_CODES, IZHIKEVICH_PARAMETERS, SimulationConfig
from simulation import IzhikevichBatch, add_observation_noise, count_spikes, simulate_izhikevich_batch

if TYPE_CHECKING:
    from inference import InferenceResult


MAX_NEURONS_PER_CLASS = 100_000
RASTER_PREVIEW_SIZE = 200


@dataclass(frozen=True)
class PopulationRequest:
    counts: tuple[int, int, int] = (10, 10, 10)
    current: float = 10.0
    seed: int = 42
    varied: bool = True

    def __post_init__(self) -> None:
        if len(self.counts) != len(CLASS_CODES) or any(
            not isinstance(count, int) or not 0 <= count <= MAX_NEURONS_PER_CLASS
            for count in self.counts
        ):
            raise ValueError(f"Use de 0 a {MAX_NEURONS_PER_CLASS} neurônios por classe.")
        if sum(self.counts) == 0:
            raise ValueError("Selecione pelo menos um neurônio.")
        if not np.isfinite(self.current):
            raise ValueError("O estímulo deve ser finito.")
        if self.seed < 0:
            raise ValueError("A semente não pode ser negativa.")


@dataclass(frozen=True)
class Population:
    request: PopulationRequest
    dt_ms: float
    traces: np.ndarray
    labels: np.ndarray
    spike_counts: np.ndarray

    @property
    def size(self) -> int:
        return int(self.labels.size)

    @property
    def timeline(self) -> np.ndarray:
        return np.arange(self.traces.shape[1], dtype=np.float32) * self.dt_ms


@dataclass(frozen=True)
class PopulationStatistics:
    total: int
    expected_counts: dict[str, int]
    predicted_counts: dict[str, int]
    classified: int
    correct: int
    indeterminate: int
    unclassified: int
    accuracy: float | None
    coverage: float | None
    mean_confidence: float | None
    total_spikes: int
    mean_rate_hz: float
    confusion_matrix: np.ndarray


@dataclass(frozen=True)
class RasterSeries:
    class_code: str | None
    times: np.ndarray
    neuron_ids: np.ndarray


@dataclass(frozen=True)
class RasterPreview:
    neuron_count: int
    series: tuple[RasterSeries, ...]


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise CancelledError()


def simulate_population(
    request: PopulationRequest,
    config: SimulationConfig,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> Population:
    """Simulate a population without introducing connections or changing labels."""

    check_cancelled(cancelled)
    labels = np.repeat(np.arange(len(CLASS_CODES)), request.counts).astype(np.int64)
    rng = np.random.default_rng(request.seed)
    parameters = {}
    for name in ("a", "b", "c", "d"):
        centers = np.asarray(
            [IZHIKEVICH_PARAMETERS[code][name] for code in CLASS_CODES],
            dtype=np.float32,
        )
        values = centers[labels]
        if request.varied and config.parameter_jitter:
            values = values * rng.normal(1.0, config.parameter_jitter, size=labels.size)
        parameters[name] = np.asarray(values, dtype=np.float32)

    batch = IzhikevichBatch(
        **parameters,
        current=np.full(labels.size, request.current, dtype=np.float32),
    )
    traces = simulate_izhikevich_batch(
        batch, time_ms=config.time_ms, dt_ms=config.dt_ms, cancelled=cancelled,
    )
    check_cancelled(cancelled)
    if request.varied and config.observation_noise_std:
        add_observation_noise(
            traces, config.observation_noise_std, rng, out=traces, cancelled=cancelled,
        )
    check_cancelled(cancelled)
    return Population(request, config.dt_ms, traces, labels, count_spikes(traces, cancelled=cancelled))


def summarize_population(
    population: Population,
    results: Sequence[InferenceResult] | None,
) -> PopulationStatistics:
    """Keep prediction coverage separate from accuracy among classified neurons."""

    expected = np.bincount(population.labels, minlength=len(CLASS_CODES))
    expected_counts = dict(zip(CLASS_CODES, map(int, expected), strict=True))
    predicted_counts = dict.fromkeys(CLASS_CODES, 0)
    matrix = np.zeros((len(CLASS_CODES), len(CLASS_CODES) + 1), dtype=np.int64)
    classified = correct = indeterminate = 0
    confidences = []
    if results is not None:
        if len(results) != population.size:
            raise ValueError("Each neuron must have exactly one inference result.")
        for expected_label, result in zip(population.labels, results, strict=True):
            if not result.determined:
                indeterminate += 1
                matrix[expected_label, -1] += 1
                continue
            if result.class_code not in predicted_counts:
                raise ValueError(f"Unknown predicted class: {result.class_code}")
            predicted_label = CLASS_CODES.index(result.class_code)
            predicted_counts[result.class_code] += 1
            matrix[expected_label, predicted_label] += 1
            classified += 1
            correct += int(predicted_label == expected_label)
            confidences.append(result.confidence)

    seconds = population.traces.shape[1] * population.dt_ms / 1000.0
    return PopulationStatistics(
        total=population.size,
        expected_counts=expected_counts,
        predicted_counts=predicted_counts,
        classified=classified,
        correct=correct,
        indeterminate=indeterminate,
        unclassified=population.size if results is None else 0,
        accuracy=correct / classified if classified else None,
        coverage=classified / population.size if results is not None else None,
        mean_confidence=float(np.mean(confidences)) if confidences else None,
        total_spikes=int(population.spike_counts.sum()),
        mean_rate_hz=float(population.spike_counts.mean() / seconds),
        confusion_matrix=matrix,
    )


def build_raster_preview(
    population: Population,
    results: Sequence[InferenceResult] | None,
    *,
    limit: int = RASTER_PREVIEW_SIZE,
) -> RasterPreview:
    """Bound drawing cost while retaining all traces for inference and inspection."""

    if limit < 1:
        raise ValueError("The raster preview limit must be positive.")
    if results is not None and len(results) != population.size:
        raise ValueError("Each neuron must have exactly one inference result.")
    selected = np.linspace(0, population.size - 1, min(limit, population.size), dtype=int)
    rows, steps = np.nonzero(population.traces[selected] >= 29.0)
    neuron_indices = selected[rows]
    groups = np.full(population.size, -1, dtype=np.int8)
    if results is not None:
        for index, result in enumerate(results):
            if result.determined:
                groups[index] = CLASS_CODES.index(result.class_code)

    series = []
    for group, code in [*enumerate(CLASS_CODES), (-1, None)]:
        mask = groups[neuron_indices] == group
        series.append(RasterSeries(
            code,
            steps[mask].astype(np.float32) * population.dt_ms,
            neuron_indices[mask].astype(np.float32) + 1,
        ))
    return RasterPreview(int(selected.size), tuple(series))
