"""Deterministic, vectorized neuron simulators used by training and the UI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
) -> np.ndarray:
    """Simulate independent Izhikevich neurons.

    Returns a float32 array shaped ``(samples, time_steps)``. The neurons are
    independent instances; this is not a connected multi-neuron network.
    """

    batch.validate()
    steps = _step_count(time_ms, dt_ms)

    a = np.asarray(batch.a, dtype=np.float32)
    b = np.asarray(batch.b, dtype=np.float32)
    c = np.asarray(batch.c, dtype=np.float32)
    d = np.asarray(batch.d, dtype=np.float32)
    current = np.asarray(batch.current, dtype=np.float32)
    sample_count = a.shape[0]

    traces = np.empty((sample_count, steps), dtype=np.float32)
    recovery = np.empty((sample_count,), dtype=np.float32)
    traces[:, 0] = -65.0
    recovery[:] = b * traces[:, 0]

    for step in range(steps - 1):
        voltage = traces[:, step]
        spiking = voltage >= 30.0
        if np.any(spiking):
            voltage[spiking] = 30.0

        dv = 0.04 * voltage * voltage + 5.0 * voltage + 140.0 - recovery + current
        du = a * (b * voltage - recovery)
        next_voltage = voltage + dv * dt_ms
        next_recovery = recovery + du * dt_ms

        if np.any(spiking):
            next_voltage[spiking] = c[spiking]
            next_recovery[spiking] = recovery[spiking] + d[spiking]

        traces[:, step + 1] = next_voltage
        recovery = next_recovery

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
    traces = np.empty((drives.size, steps), dtype=np.float32)
    traces[:, 0] = resting_voltage

    for step in range(steps - 1):
        voltage = traces[:, step]
        dv = (resting_voltage - voltage + drives) / tau_ms
        next_voltage = voltage + dv * dt_ms
        spiking = next_voltage >= threshold_voltage
        traces[:, step + 1] = np.where(spiking, reset_voltage, next_voltage)
        traces[spiking, step] = threshold_voltage

    return traces


def count_spikes(traces: np.ndarray, threshold: float = 29.0) -> np.ndarray:
    values = np.asarray(traces)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("traces must have shape (samples, time_steps) or (time_steps,).")
    return np.count_nonzero(values >= threshold, axis=1).astype(np.int32)


def add_observation_noise(
    traces: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(traces, dtype=np.float32)
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")
    if noise_std == 0:
        return values.copy()
    noisy = values + rng.normal(0.0, noise_std, size=values.shape).astype(np.float32)
    return np.clip(noisy, -110.0, 35.0).astype(np.float32, copy=False)

