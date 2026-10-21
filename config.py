"""Shared scientific and training configuration.

The labels represent simulated firing-pattern phenotypes, not biological
diagnoses or measured human neurons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


CLASS_NAMES: Final[tuple[str, ...]] = (
    "Regular Spiking",
    "Chattering",
    "Fast Spiking",
)

CLASS_CODES: Final[tuple[str, ...]] = ("RS", "CH", "FS")

# Canonical parameterizations commonly used to demonstrate the three
# Izhikevich response patterns. Input current is expressed in model units.
IZHIKEVICH_PARAMETERS: Final[dict[str, dict[str, float]]] = {
    "RS": {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0},
    "CH": {"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.0},
    "FS": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0},
}


@dataclass(frozen=True)
class SimulationConfig:
    time_ms: float = 200.0
    dt_ms: float = 0.1
    current_min: float = 5.0
    current_max: float = 30.0
    parameter_jitter: float = 0.03
    observation_noise_std: float = 0.20
    minimum_spikes: int = 2

    def __post_init__(self) -> None:
        if self.time_ms <= 0 or self.dt_ms <= 0:
            raise ValueError("time_ms and dt_ms must be positive.")
        if self.current_min >= self.current_max:
            raise ValueError("current_min must be lower than current_max.")
        if self.parameter_jitter < 0 or self.observation_noise_std < 0:
            raise ValueError("jitter and noise cannot be negative.")
        if self.minimum_spikes < 1:
            raise ValueError("minimum_spikes must be at least one.")

    @property
    def window_size(self) -> int:
        return int(round(self.time_ms / self.dt_ms))

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["window_size"] = self.window_size
        return data
