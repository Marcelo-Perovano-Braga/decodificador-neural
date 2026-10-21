"""Validated single-trace and batched inference for the desktop UI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass

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


def classify_traces(
    traces: np.ndarray,
    model: nn.Module,
    *,
    normalization_mean: float,
    normalization_std: float,
    class_codes: Sequence[str],
    class_names: Sequence[str],
    minimum_spikes: int,
    confidence_threshold: float,
    device: torch.device,
    currents: np.ndarray | None = None,
    trained_current_range: tuple[float, float] | None = None,
    batch_size: int = 64,
    cancelled: Callable[[], bool] | None = None,
) -> list[InferenceResult]:
    """Classify independent traces in bounded batches, preserving row order.

    Neither expected class labels nor generator parameters are passed to the
    model. Invalid, silent and out-of-domain rows do not enter inference.
    """

    values = np.asarray(traces, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("Expected traces shaped (neurons, time_steps), with at least two steps.")
    if not np.isfinite(normalization_std) or normalization_std <= 0:
        raise ValueError("normalization_std must be finite and positive.")
    if not np.isfinite(normalization_mean):
        raise ValueError("normalization_mean must be finite.")
    if len(class_codes) == 0 or len(class_codes) != len(class_names):
        raise ValueError("class_codes and class_names must have the same nonzero length.")
    if batch_size < 1 or minimum_spikes < 1:
        raise ValueError("batch_size and minimum_spikes must be positive.")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one.")
    if currents is not None:
        currents = np.asarray(currents, dtype=np.float32)
        if currents.shape != (values.shape[0],):
            raise ValueError("currents must contain one value per neuron.")
    if trained_current_range is not None:
        low, high = trained_current_range
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError("trained_current_range must be finite and ordered.")

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise CancelledError()

    check_cancelled()
    spikes = count_spikes(values)
    finite = np.isfinite(values).all(axis=1)
    results = []
    eligible = []
    for index, spike_count in enumerate(spikes):
        reason = None
        if not finite[index]:
            reason = "sinal contém valores não finitos"
        elif currents is not None and not np.isfinite(currents[index]):
            reason = "estímulo não finito"
        elif currents is not None and trained_current_range is not None:
            low, high = trained_current_range
            if currents[index] < low or currents[index] > high:
                reason = "estímulo fora do domínio de treinamento"
        if reason is None and spike_count < minimum_spikes:
            reason = "atividade insuficiente para distinguir o padrão"
        results.append(InferenceResult(False, None, None, None, 0.0, int(spike_count), reason))
        if reason is None:
            eligible.append(index)

    if not eligible:
        return results

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(eligible), batch_size):
            check_cancelled()
            indices = eligible[start:start + batch_size]
            normalized = (values[indices] - normalization_mean) / normalization_std
            tensor = torch.from_numpy(normalized).unsqueeze(1).to(device)
            logits = model(tensor)
            if tuple(logits.shape) != (len(indices), len(class_codes)):
                raise ValueError("Model output does not match the configured classes.")
            probabilities = torch.softmax(logits, dim=1)
            confidence_tensors, index_tensors = probabilities.max(dim=1)
            confidences = confidence_tensors.cpu().tolist()
            predictions = index_tensors.cpu().tolist()
            for row, confidence, prediction in zip(indices, confidences, predictions, strict=True):
                if not np.isfinite(confidence):
                    results[row] = InferenceResult(
                        False, None, None, None, 0.0, int(spikes[row]),
                        "previsão contém valores não finitos",
                    )
                    continue
                determined = confidence >= confidence_threshold
                results[row] = InferenceResult(
                    determined, prediction, class_codes[prediction], class_names[prediction],
                    float(confidence), int(spikes[row]),
                    None if determined else "confiança abaixo do limiar",
                )
            check_cancelled()
    return results


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
    return classify_traces(
        np.asarray(trace, dtype=np.float32).reshape(1, -1), model,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        class_codes=class_codes,
        class_names=class_names,
        minimum_spikes=minimum_spikes,
        confidence_threshold=confidence_threshold,
        device=device,
        currents=np.asarray([current]) if current is not None else None,
        trained_current_range=trained_current_range,
    )[0]
