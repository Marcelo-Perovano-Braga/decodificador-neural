"""Validated classifier inference shared by the desktop UI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from simulation import count_spikes


@dataclass(frozen=True)
class InferenceResult:
    determined: bool
    class_index: int | None
    class_code: str | None
    class_name: str | None
    confidence: float
    spike_count: int
    reason: str | None = None


def classify_trace(
    trace: np.ndarray,
    model: nn.Module,
    *,
    normalization_mean: float,
    normalization_std: float,
    class_codes: Sequence[str],
    class_names: Sequence[str],
    minimum_spikes: int,
    confidence_threshold: float,
    device: torch.device,
    current: float | None = None,
    trained_current_range: tuple[float, float] | None = None,
) -> InferenceResult:
    values = np.asarray(trace, dtype=np.float32).reshape(-1)
    spikes = int(count_spikes(values)[0])
    if normalization_std <= 0:
        raise ValueError("normalization_std must be positive.")
    if len(class_codes) != len(class_names):
        raise ValueError("class_codes and class_names must have the same length.")

    if current is not None and trained_current_range is not None:
        low, high = trained_current_range
        if current < low or current > high:
            return InferenceResult(
                determined=False,
                class_index=None,
                class_code=None,
                class_name=None,
                confidence=0.0,
                spike_count=spikes,
                reason="estímulo fora do domínio de treinamento",
            )

    if spikes < minimum_spikes:
        return InferenceResult(
            determined=False,
            class_index=None,
            class_code=None,
            class_name=None,
            confidence=0.0,
            spike_count=spikes,
            reason="atividade insuficiente para distinguir o padrão",
        )

    normalized = (values - normalization_mean) / normalization_std
    tensor = torch.from_numpy(normalized).view(1, 1, -1).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence_tensor, index_tensor = torch.max(probabilities, dim=0)
    confidence = float(confidence_tensor.item())
    index = int(index_tensor.item())

    if confidence < confidence_threshold:
        return InferenceResult(
            determined=False,
            class_index=index,
            class_code=class_codes[index],
            class_name=class_names[index],
            confidence=confidence,
            spike_count=spikes,
            reason="confiança abaixo do limiar",
        )

    return InferenceResult(
        determined=True,
        class_index=index,
        class_code=class_codes[index],
        class_name=class_names[index],
        confidence=confidence,
        spike_count=spikes,
    )

