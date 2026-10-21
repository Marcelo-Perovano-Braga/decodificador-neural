"""Paired reproducible conditions, including types/ranges withheld from fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import IZHIKEVICH_PARAMETERS, CLASS_CODES, SimulationConfig
from network import NetworkConfig, build_sparse_network, simulate_connected_batch
from simulation import IzhikevichBatch, add_observation_noise


@dataclass(frozen=True)
class Condition:
    name: str
    axis: str
    value: float
    noise_std: float
    jitter: float
    profile: str = "constant"
    current_range: tuple[float, float] | None = None
    onset_fraction: float = 0.1
    network_weight: float | None = None
    held_out_reason: str | None = None
    pulse_duration_fraction: float = 0.5
    frequency_hz: float = 20.0


@dataclass(frozen=True)
class ResearchCohort:
    traces: np.ndarray
    labels: np.ndarray
    parameters: dict[str, np.ndarray]
    amplitudes: np.ndarray
    physical_spike_counts: np.ndarray
    outside_recorded_drive_range: np.ndarray
    synapses: int


def make_protocol(config: SimulationConfig, *, training_profiles=("constant",)) -> list[Condition]:
    conditions = [Condition("reference", "reference", 0, config.observation_noise_std, config.parameter_jitter)]
    for value in sorted({0.0, config.observation_noise_std, 1.0, 2.0, 5.0, 10.0}):
        conditions.append(Condition(f"noise_{value:g}", "noise_std", value, value, config.parameter_jitter))
    for value in sorted({0.0, config.parameter_jitter, 0.06, 0.10, 0.15}):
        conditions.append(Condition(f"jitter_{value:g}", "parameter_jitter", value, config.observation_noise_std, value))
    for value in (0.0, 0.1, 0.5, 1.0):
        conditions.append(Condition(
            f"network_{value:g}", "network_weight", value,
            config.observation_noise_std, config.parameter_jitter, network_weight=value,
        ))
    for value in (0.1, 0.3, 0.5):
        conditions.append(Condition(
            f"pulse_onset_{value:g}", "pulse_onset_fraction", value,
            config.observation_noise_std, config.parameter_jitter,
            profile="pulse", onset_fraction=value,
            held_out_reason="pulse stimulus type excluded from training" if "pulse" not in training_profiles else None,
        ))
    for profile in ("ramp", "sine"):
        conditions.append(Condition(
            profile, "stimulus_type", 0, config.observation_noise_std, config.parameter_jitter,
            profile=profile, held_out_reason=f"{profile} stimulus type excluded from training" if profile not in training_profiles else None,
        ))
    span = config.current_max - config.current_min
    low_edge = float(np.nextafter(np.float32(config.current_min), np.float32(-np.inf)))
    high_edge = float(np.nextafter(np.float32(config.current_max), np.float32(np.inf)))
    for name, bounds in (
        ("current_below_range", (config.current_min - span / 2, low_edge)),
        ("current_above_range", (high_edge, config.current_max + span / 2)),
    ):
        conditions.append(Condition(
            name, "held_out_current", float(np.mean(bounds)), config.observation_noise_std,
            config.parameter_jitter, current_range=bounds,
            held_out_reason="strictly outside checkpoint-recorded constant-current training range",
        ))
    return conditions


def stimulus(amplitudes: np.ndarray, profile: str, steps: int, dt_ms: float, onset_fraction=0.1,
             pulse_duration_fraction=0.5, frequency_hz=20.0, *, rng=None) -> np.ndarray:
    if profile == "constant":
        return amplitudes
    phase = np.arange(steps, dtype=np.float32) / steps
    if profile == "pulse":
        envelope = ((phase >= onset_fraction) & (phase < onset_fraction + pulse_duration_fraction)).astype(np.float32)
    elif profile == "ramp":
        envelope = np.linspace(0.25, 1.75, steps, dtype=np.float32)
    elif profile == "sine":
        time_seconds = np.arange(steps, dtype=np.float32) * dt_ms / 1000
        envelope = 1 + 0.5 * np.sin(2 * np.pi * frequency_hz * time_seconds)
    elif profile == "double_pulse":
        envelope = (((phase >= .1) & (phase < .3)) | ((phase >= .6) & (phase < .8))).astype(np.float32)
    elif profile == "triangle":
        envelope = 0.25 + 1.5 * (1 - np.abs(2 * phase - 1))
    elif profile == "ramp_down":
        envelope = np.linspace(1.75, .25, steps, dtype=np.float32)
    elif profile.startswith("random_"):
        if rng is None:
            raise ValueError("Randomized stimuli require an explicit random generator.")
        envelopes = np.empty((len(amplitudes), steps), dtype=np.float32)
        for row in range(len(amplitudes)):
            if profile == "random_pulses":
                envelope = np.zeros(steps, dtype=np.float32)
                for _ in range(int(rng.integers(1, 5))):
                    onset = rng.uniform(0, .8)
                    duration = rng.uniform(.12, min(.6, 1 - onset))
                    envelope[(phase >= onset) & (phase < onset + duration)] = rng.uniform(.65, 1.75)
            elif profile in ("random_steps", "random_smooth"):
                knots = np.r_[0., np.sort(rng.uniform(.05, .95, int(rng.integers(2, 8)))), 1.]
                heights = rng.uniform(.15, 1.75, len(knots))
                if profile == "random_smooth":
                    envelope = np.interp(phase, knots, heights)
                else:
                    envelope = heights[np.searchsorted(knots[1:], phase, side="right")]
            elif profile == "random_ramp":
                envelope = np.linspace(*rng.uniform(.15, 1.9, 2), steps)
            elif profile == "random_triangle":
                peak = rng.uniform(.2, .8)
                envelope = np.interp(phase, [0, peak, 1],
                                     [rng.uniform(.05, .5), rng.uniform(1.25, 1.9), rng.uniform(.05, .5)])
            elif profile == "random_sine":
                frequency, offset = rng.uniform(5, 45), rng.uniform(0, 2 * np.pi)
                envelope = np.maximum(.05, rng.uniform(.8, 1.1) + rng.uniform(.3, .7) *
                                      np.sin(2 * np.pi * frequency * phase * steps * dt_ms / 1000 + offset))
            else:
                raise ValueError(f"Unknown randomized stimulus profile: {profile}")
            envelopes[row] = envelope
        return np.asfortranarray(amplitudes[:, None] * envelopes, dtype=np.float32)
    elif profile == "chirp":
        # Frequency sweeps linearly from 5 to 65 Hz over the observation window.
        duration_seconds = steps * dt_ms / 1000
        time_seconds = phase * duration_seconds
        envelope = 1 + .65 * np.sin(2 * np.pi * (5 * time_seconds + 30 * time_seconds ** 2 / duration_seconds))
    elif profile == "exponential":
        envelope = .25 + 1.5 * np.exp(-4 * phase)
    elif profile == "cosine_burst":
        local = np.clip((phase - .15) / .7, 0, 1)
        envelope = .25 + .75 * (1 - np.cos(2 * np.pi * local))
    else:
        raise ValueError(f"Unknown stimulus profile: {profile}")
    return np.asfortranarray(amplitudes[:, None] * envelope[None, :], dtype=np.float32)


def generate_cohort(
    condition: Condition, config: SimulationConfig, samples_per_class: int, seed: int,
    *, out_degree: int = 100, namespace: int = 1,
) -> ResearchCohort:
    if not 1 <= samples_per_class <= 10_000 or seed < 0:
        raise ValueError("Use 1–10,000 samples per class and a nonnegative seed.")
    if not np.isfinite([condition.noise_std, condition.jitter, condition.onset_fraction,
                        condition.pulse_duration_fraction, condition.frequency_hz]).all():
        raise ValueError("Condition parameters must be finite.")
    if condition.noise_std < 0 or not 0 <= condition.jitter <= 0.3:
        raise ValueError("Use nonnegative noise and parameter jitter between 0 and 0.3.")
    if not 0 <= condition.onset_fraction < 1 or not 0 < condition.pulse_duration_fraction <= 1 - condition.onset_fraction:
        raise ValueError("Pulse must start and end within the observation window.")
    if not 0 < condition.frequency_hz < 500 / config.dt_ms:
        raise ValueError("Stimulus frequency must be positive and below the Nyquist frequency.")
    labels = np.repeat(np.arange(3), samples_per_class)
    parameters = {}
    # Separate streams and unconditional parameter draws keep sweeps paired,
    # including zero-noise/zero-jitter controls and connected/independent runs.
    for index, name in enumerate(("a", "b", "c", "d")):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53545544, namespace, index]))
        centers = np.asarray([IZHIKEVICH_PARAMETERS[code][name] for code in CLASS_CODES])[labels]
        parameters[name] = (centers * (1 + condition.jitter * rng.standard_normal(labels.size))).astype(np.float32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53545544, namespace, 4]))
    low, high = condition.current_range or (config.current_min, config.current_max)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Current range must be finite and ordered.")
    amplitudes = rng.uniform(low, high, labels.size).astype(np.float32)
    drive_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53545544, namespace, 6]))
    drive = stimulus(amplitudes, condition.profile, config.window_size, config.dt_ms,
                     condition.onset_fraction, condition.pulse_duration_fraction, condition.frequency_hz,
                     rng=drive_rng)
    network_config = NetworkConfig(
        out_degree=out_degree if condition.network_weight is not None else 0,
        weight=condition.network_weight or 0.0,
    )
    graph = build_sparse_network(labels, network_config, seed)
    # A zero-degree graph uses the same equations and event counter as connected
    # conditions, retaining bitwise voltage equivalence with the legacy solver.
    traces, activity = simulate_connected_batch(
        IzhikevichBatch(**parameters, current=drive), graph,
        time_ms=config.time_ms, dt_ms=config.dt_ms,
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53545544, namespace, 5]))
    add_observation_noise(traces, condition.noise_std, rng, out=traces)
    low_drive = drive.min(axis=1) if drive.ndim == 2 else drive
    high_drive = drive.max(axis=1) if drive.ndim == 2 else drive
    return ResearchCohort(
        traces, labels, parameters, amplitudes, activity.spike_counts,
        (low_drive < config.current_min) | (high_drive > config.current_max),
        graph.synapse_count,
    )
