"""Deterministic, vectorized neuron simulators used by training and the UI."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass

import numpy as np


_ROW_BLOCK_SIZE = 512
_CANCELLATION_STEPS = 32


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise CancelledError()


@dataclass(frozen=True)
class IzhikevichBatch:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    current: np.ndarray

    def validate(self) -> None:
        lengths = {np.asarray(value).shape[0] for value in (
            self.a,
            self.b,
            self.c,
            self.d,
            self.current,
        )}
        if len(lengths) != 1:
            raise ValueError("All parameter arrays must have the same length.")
        if next(iter(lengths), 0) < 1:
            raise ValueError("At least one simulated neuron is required.")


def _step_count(time_ms: float, dt_ms: float) -> int:
    if time_ms <= 0 or dt_ms <= 0:
        raise ValueError("time_ms and dt_ms must be positive.")
    steps = int(round(time_ms / dt_ms))
    if steps < 2:
        raise ValueError("Simulation must contain at least two time steps.")
    return steps


def simulate_izhikevich_batch(
    batch: IzhikevichBatch,
    *,
    time_ms: float = 200.0,
    dt_ms: float = 0.1,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Simulate independent Izhikevich neurons.

    Returns a float32 array shaped ``(samples, time_steps)``. The neurons are
    independent instances; this is not a connected multi-neuron network.
    """

    batch.validate()
    steps = _step_count(time_ms, dt_ms)
    _check_cancelled(cancelled)

    a = np.asarray(batch.a, dtype=np.float32)
    b = np.asarray(batch.b, dtype=np.float32)
    c = np.asarray(batch.c, dtype=np.float32)
    d = np.asarray(batch.d, dtype=np.float32)
    current = np.asarray(batch.current, dtype=np.float32)
    sample_count = a.shape[0]

    # Each timestep touches every neuron. Keep those columns contiguous;
    # callers still receive the same (neurons, time_steps) array and values.
    traces = np.empty((sample_count, steps), dtype=np.float32, order="F")
    recovery = np.empty((sample_count,), dtype=np.float32)
    traces[:, 0] = -65.0
    recovery[:] = b * traces[:, 0]

    for step in range(steps - 1):
        if step % _CANCELLATION_STEPS == 0:
            _check_cancelled(cancelled)
        voltage = traces[:, step]
        spiking = voltage >= 30.0
        has_spikes = np.any(spiking)
        if has_spikes:
            voltage[spiking] = 30.0

        dv = 0.04 * voltage * voltage + 5.0 * voltage + 140.0 - recovery + current
        du = a * (b * voltage - recovery)
        next_voltage = voltage + dv * dt_ms
        next_recovery = recovery + du * dt_ms

        if has_spikes:
            next_voltage[spiking] = c[spiking]
            next_recovery[spiking] = recovery[spiking] + d[spiking]

        traces[:, step + 1] = next_voltage
        recovery = next_recovery

    _check_cancelled(cancelled)
    return traces


def simulate_izhikevich_single(
    a: float,
    b: float,
    c: float,
    d: float,
    current: float,
    *,
    time_ms: float = 200.0,
    dt_ms: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    batch = IzhikevichBatch(
        a=np.asarray([a], dtype=np.float32),
        b=np.asarray([b], dtype=np.float32),
        c=np.asarray([c], dtype=np.float32),
        d=np.asarray([d], dtype=np.float32),
        current=np.asarray([current], dtype=np.float32),
    )
    trace = simulate_izhikevich_batch(batch, time_ms=time_ms, dt_ms=dt_ms)[0]
    timeline = np.arange(trace.shape[0], dtype=np.float32) * np.float32(dt_ms)
    return timeline, trace


def simulate_lif_batch(
    input_drive: np.ndarray,
    *,
    time_ms: float = 200.0,
    dt_ms: float = 0.1,
    resting_voltage: float = -70.0,
    threshold_voltage: float = -50.0,
    reset_voltage: float = -70.0,
    tau_ms: float = 10.0,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Simulate independent LIF neurons for the computational baseline.

    ``input_drive`` is a voltage-equivalent drive in model units. This avoids
    incorrectly labelling the value as an electrical current measured in mV.
    """

    drives = np.asarray(input_drive, dtype=np.float32).reshape(-1)
    if drives.size < 1:
        raise ValueError("At least one input drive is required.")
    if tau_ms <= 0:
        raise ValueError("tau_ms must be positive.")

    steps = _step_count(time_ms, dt_ms)
    _check_cancelled(cancelled)
    traces = np.empty((drives.size, steps), dtype=np.float32, order="F")
    traces[:, 0] = resting_voltage

    for step in range(steps - 1):
        if step % _CANCELLATION_STEPS == 0:
            _check_cancelled(cancelled)
        voltage = traces[:, step]
        dv = (resting_voltage - voltage + drives) / tau_ms
        next_voltage = voltage + dv * dt_ms
        spiking = next_voltage >= threshold_voltage
        traces[:, step + 1] = np.where(spiking, reset_voltage, next_voltage)
        traces[spiking, step] = threshold_voltage

    _check_cancelled(cancelled)
    return traces


def count_spikes(
    traces: np.ndarray,
    threshold: float = 29.0,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    values = np.asarray(traces)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("traces must have shape (samples, time_steps) or (time_steps,).")
    # Bound the comparison temporary, even for million-row populations.
    counts = np.empty(values.shape[0], dtype=np.int32)
    _check_cancelled(cancelled)
    for start in range(0, values.shape[0], _ROW_BLOCK_SIZE):
        _check_cancelled(cancelled)
        stop = start + _ROW_BLOCK_SIZE
        counts[start:stop] = np.count_nonzero(values[start:stop] >= threshold, axis=1)
    _check_cancelled(cancelled)
    return counts


def add_observation_noise(
    traces: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
    *,
    out: np.ndarray | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Add exactly the same seeded noise using bounded row buffers.

    By default the input is preserved. ``out`` must be independent storage or
    exactly the input view. The population worker supplies its own writable
    trace array to avoid retaining a second full array.
    """
    values = np.asarray(traces, dtype=np.float32)
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")
    _check_cancelled(cancelled)
    if out is None:
        out = np.empty_like(values)
    elif out.shape != values.shape or out.dtype != np.float32 or not out.flags.writeable:
        raise ValueError("out must be writable float32 storage with the same shape as traces.")
    elif np.shares_memory(out, values) and (
        out.strides != values.strides
        or out.__array_interface__["data"][0] != values.__array_interface__["data"][0]
    ):
        raise ValueError("out must be the input view or independent storage, without partial overlap.")
    # Keep scalar and higher-dimensional inputs supported as before.
    source = values.reshape(1) if values.ndim == 0 else values
    target = out.reshape(1) if out.ndim == 0 else out
    for start in range(0, source.shape[0], _ROW_BLOCK_SIZE):
        _check_cancelled(cancelled)
        stop = start + _ROW_BLOCK_SIZE
        block = target[start:stop]
        if noise_std == 0:
            np.copyto(block, source[start:stop])
        else:
            # Row order and float64 generation before float32 conversion match
            # the original full-array draw, including the RNG state afterward.
            noise = rng.normal(0.0, noise_std, size=source[start:stop].shape).astype(np.float32)
            np.add(source[start:stop], noise, out=block)
            np.clip(block, -110.0, 35.0, out=block)
    _check_cancelled(cancelled)
    return out

