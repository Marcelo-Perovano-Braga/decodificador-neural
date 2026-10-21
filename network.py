"""Sparse, delayed current-based synapses for the exploratory network mode.

RS/CH/FS parameters describe intrinsic responses. Excitatory/inhibitory roles
are assigned separately; they are not inferred from those response labels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import numpy as np

from simulation import IzhikevichBatch, _check_cancelled, _step_count, _validate_current_shape


MAX_OUT_DEGREE = 1_000
MAX_SYNAPSES = 10_000_000
_BLOCK_SIZE = 256


@dataclass(frozen=True)
class NetworkConfig:
    out_degree: int = 100
    weight: float = 0.5
    inhibitory_fraction: float = 0.2
    inhibitory_gain: float = 4.0
    delay_ms: float = 1.0
    tau_ms: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.out_degree, int) or not 0 <= self.out_degree <= MAX_OUT_DEGREE:
            raise ValueError(f"Use de 0 a {MAX_OUT_DEGREE} conexões por neurônio.")
        values = (self.weight, self.inhibitory_fraction, self.inhibitory_gain, self.delay_ms, self.tau_ms)
        if not np.isfinite(values).all():
            raise ValueError("Os parâmetros da rede devem ser finitos.")
        if not 0 <= self.weight <= 5 or not 0 <= self.inhibitory_gain <= 10:
            raise ValueError("Use força entre 0 e 5 e ganho inibitório entre 0 e 10.")
        if not 0 <= self.inhibitory_fraction <= 1:
            raise ValueError("A fração inibitória deve estar entre 0 e 1.")
        if not 0 < self.delay_ms <= 20 or not 0.5 <= self.tau_ms <= 50:
            raise ValueError("Use atraso entre 0 e 20 ms e decaimento entre 0,5 e 50 ms.")

    def degree_for(self, size: int) -> int:
        if size < 1:
            raise ValueError("Selecione pelo menos um neurônio.")
        degree = min(self.out_degree, size - 1)
        if size * degree > MAX_SYNAPSES:
            raise ValueError(
                f"Rede com {size * degree:,} sinapses; limite de {MAX_SYNAPSES:,}. "
                "Reduza a população ou as conexões por neurônio."
            )
        return degree


@dataclass(frozen=True)
class SparseNetwork:
    config: NetworkConfig
    targets: np.ndarray
    source_weights: np.ndarray
    inhibitory: np.ndarray
    in_degrees: np.ndarray
    class_connections: np.ndarray

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])

    @property
    def out_degree(self) -> int:
        return int(self.targets.shape[1])

    @property
    def synapse_count(self) -> int:
        return int(self.targets.size)


@dataclass(frozen=True)
class NetworkActivity:
    network: SparseNetwork
    spike_neurons: np.ndarray
    spike_steps: np.ndarray
    spike_counts: np.ndarray
    mean_synaptic_current: np.ndarray
    effective_delay_ms: float
    delivered_events: int
    rate_times_ms: np.ndarray
    rate_hz: np.ndarray


def build_sparse_network(
    labels: np.ndarray,
    config: NetworkConfig,
    seed: int,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> SparseNetwork:
    """Uniform fixed out-degree, without self-connections or duplicate targets.

    A separate random stream leaves the existing parameter/noise draws intact.
    Storage is O(N*K); construction never allocates an N-by-N matrix.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("A rede deve usar os rótulos RS, CH e FS.")
    degree = config.degree_for(labels.size)
    _check_cancelled(cancelled)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53594E]))
    inhibitory = np.zeros(labels.size, dtype=bool)
    inhibitory[rng.choice(labels.size, round(labels.size * config.inhibitory_fraction), replace=False)] = True
    weights = np.where(inhibitory, -config.weight * config.inhibitory_gain, config.weight).astype(np.float32)
    targets = np.empty((labels.size, degree), dtype=np.int32)
    in_degrees = np.zeros(labels.size, dtype=np.int32)
    class_connections = np.zeros((3, 3), dtype=np.int64)
    for start in range(0, labels.size, _BLOCK_SIZE):
        _check_cancelled(cancelled)
        stop = min(start + _BLOCK_SIZE, labels.size)
        if not degree:
            continue
        for source in range(start, stop):
            choices = rng.choice(labels.size - 1, degree, replace=False)
            targets[source] = choices + (choices >= source)
        block = targets[start:stop]
        np.add.at(in_degrees, block.ravel(), 1)
        for label in range(3):
            destinations = block[labels[start:stop] == label]
            class_connections[label] += np.bincount(labels[destinations].ravel(), minlength=3)
    _check_cancelled(cancelled)
    return SparseNetwork(config, targets, weights, inhibitory, in_degrees, class_connections)


def _deliver_spikes(
    network: SparseNetwork,
    sources: np.ndarray,
    current: np.ndarray,
    cancelled: Callable[[], bool] | None,
) -> None:
    for start in range(0, sources.size, _BLOCK_SIZE):
        _check_cancelled(cancelled)
        indices = sources[start:start + _BLOCK_SIZE]
        targets = network.targets[indices]
        weights = np.broadcast_to(network.source_weights[indices, None], targets.shape)
        # add.at sums repeated destinations; ordinary fancy += loses events.
        np.add.at(current, targets.ravel(), weights.ravel())


def simulate_connected_batch(
    batch: IzhikevichBatch,
    network: SparseNetwork,
    *,
    time_ms: float = 200.0,
    dt_ms: float = 0.1,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, NetworkActivity]:
    """Keep the original Euler/reset convention, adding delayed synaptic drive.

    At time t, decay the previous synaptic current, add events emitted at t-d,
    and integrate using I_external + I_synaptic. Each spike adds its source's
    signed weight; current decays exactly by exp(-dt/tau). Delay rounds up to
    whole timesteps. Spike detection precedes observation noise.
    """
    batch.validate()
    steps = _step_count(time_ms, dt_ms)
    _check_cancelled(cancelled)
    size = len(batch.a)
    if network.size != size or network.source_weights.shape != (size,):
        raise ValueError("A rede deve conter todos os neurônios da população.")
    a, b, c, d, external = (np.asarray(values, dtype=np.float32) for values in (
        batch.a, batch.b, batch.c, batch.d, batch.current,
    ))
    _validate_current_shape(external, size, steps)
    traces = np.empty((size, steps), dtype=np.float32, order="F")
    traces[:, 0] = -65.0
    recovery = b * traces[:, 0]
    synaptic = np.zeros(size, dtype=np.float32)
    mean_current = np.empty(steps, dtype=np.float32)
    spike_counts = np.zeros(size, dtype=np.int32)
    delay_steps = max(1, ceil(network.config.delay_ms / dt_ms - 1e-10))
    pending: list[np.ndarray | None] = [None] * min(delay_steps + 1, steps)
    decay = np.float32(np.exp(-dt_ms / network.config.tau_ms))
    neuron_events, time_events = [], []
    delivered = 0
    try:
        with np.errstate(over="raise", invalid="raise"):
            for step in range(steps):
                if step % 32 == 0:
                    _check_cancelled(cancelled)
                if step:
                    synaptic *= decay
                arriving = pending[step % len(pending)]
                pending[step % len(pending)] = None
                if arriving is not None:
                    _deliver_spikes(network, arriving, synaptic, cancelled)
                    delivered += arriving.size * network.out_degree
                mean_current[step] = np.mean(synaptic)
                voltage = traces[:, step]
                spiking = voltage >= 30.0
                has_spikes = np.any(spiking)
                if has_spikes:
                    sources = np.flatnonzero(spiking).astype(np.int32)
                    neuron_events.append(sources)
                    time_events.append(np.full(sources.size, step, dtype=np.int32))
                    spike_counts[sources] += 1
                    if network.out_degree and step + delay_steps < steps:
                        pending[(step + delay_steps) % len(pending)] = sources
                if step == steps - 1:
                    break
                if has_spikes:
                    voltage[spiking] = 30.0
                drive = external[:, step] if external.ndim == 2 else external
                dv = 0.04 * voltage * voltage + 5.0 * voltage + 140.0 - recovery + drive
                dv += synaptic
                du = a * (b * voltage - recovery)
                next_voltage = voltage + dv * dt_ms
                next_recovery = recovery + du * dt_ms
                if has_spikes:
                    next_voltage[spiking] = c[spiking]
                    next_recovery[spiking] = recovery[spiking] + d[spiking]
                traces[:, step + 1] = next_voltage
                recovery = next_recovery
    except FloatingPointError as exc:
        raise ValueError("Rede numericamente instável. Reduza a força ou o ganho das sinapses.") from exc

    _check_cancelled(cancelled)
    spike_neurons = np.concatenate(neuron_events) if neuron_events else np.empty(0, dtype=np.int32)
    spike_steps = np.concatenate(time_events) if time_events else np.empty(0, dtype=np.int32)
    duration_ms = steps * dt_ms
    bins = max(1, ceil(duration_ms / 5.0))
    widths_ms = np.minimum(5.0, duration_ms - np.arange(bins) * 5.0)
    rate_times = np.arange(bins) * 5.0 + widths_ms / 2
    bin_indices = np.minimum((spike_steps * dt_ms / 5.0).astype(np.int64), bins - 1)
    rate_hz = np.bincount(bin_indices, minlength=bins) / (size * widths_ms / 1000.0)
    activity = NetworkActivity(
        network, spike_neurons, spike_steps, spike_counts, mean_current,
        delay_steps * dt_ms, int(delivered), rate_times, rate_hz,
    )
    return traces, activity
