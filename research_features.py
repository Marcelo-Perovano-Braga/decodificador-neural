"""Small timing-feature baselines, fitted exclusively on the training split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = ("rate_hz", "median_isi_ms", "isi_cv", "short_isi_fraction", "adaptation_ratio")


def extract_timing_features(
    traces: np.ndarray, dt_ms: float, *, burst_isi_ms: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use observed voltages only, never generator parameters or clean events.

    A contiguous region >=29 mV counts once, at its rising edge. Undefined ISI
    features are zero: <2 peaks for median/fraction, <3 for CV/adaptation.
    short_isi_fraction is an explicit timing proxy, not a biological diagnosis.
    """
    traces = np.asarray(traces)
    if traces.ndim != 2 or traces.shape[1] < 2:
        raise ValueError("Expected a matrix of voltage traces.")
    if not np.isfinite([dt_ms, burst_isi_ms]).all() or dt_ms <= 0 or burst_isi_ms <= 0:
        raise ValueError("Time step and burst interval must be finite and positive.")
    features = np.zeros((traces.shape[0], len(FEATURE_NAMES)), dtype=np.float64)
    counts = np.zeros(traces.shape[0], dtype=np.int32)
    finite = np.zeros(traces.shape[0], dtype=bool)
    duration_seconds = traces.shape[1] * dt_ms / 1000
    for row, trace in enumerate(traces):
        if not np.isfinite(trace).all():
            continue
        finite[row] = True
        above = trace >= 29.0
        events = np.flatnonzero(above & np.r_[True, ~above[:-1]])
        counts[row] = events.size
        features[row, 0] = events.size / duration_seconds
        if events.size >= 2:
            isi = np.diff(events) * dt_ms
            features[row, 1] = np.median(isi)
            features[row, 3] = np.mean(isi <= burst_isi_ms)
            if events.size >= 3:
                features[row, 2] = np.std(isi) / np.mean(isi)
                features[row, 4] = isi[-1] / isi[0]
    return features, counts, finite


@dataclass(frozen=True)
class TimingCentroid:
    columns: tuple[int, ...]
    mean: np.ndarray
    scale: np.ndarray
    centroids: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, *, columns=(0, 1, 2, 3, 4)) -> TimingCentroid:
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or labels.shape != (features.shape[0],) or not features.shape[0]:
            raise ValueError("Training features and labels must contain aligned nonempty rows.")
        if not columns or len(set(columns)) != len(columns) or any(index not in range(len(FEATURE_NAMES)) for index in columns):
            raise ValueError("Choose distinct timing-feature columns.")
        features = features[:, columns]
        if not np.isfinite(features).all() or set(labels.tolist()) != {0, 1, 2}:
            raise ValueError("Baseline training requires finite features and all three classes.")
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardized = (features - mean) / scale
        centroids = np.stack([standardized[labels == label].mean(axis=0) for label in range(3)])
        return cls(tuple(columns), mean, scale, centroids)

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features)[:, self.columns] - self.mean) / self.scale
        distance = np.sum((standardized[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        return distance.argmin(axis=1)

    def to_dict(self) -> dict:
        return {
            "method": "training-standardized nearest centroid; no confidence calibration",
            "features": [FEATURE_NAMES[index] for index in self.columns],
            "columns": list(self.columns), "mean": self.mean.tolist(),
            "scale": self.scale.tolist(), "centroids": self.centroids.tolist(),
        }


def prediction_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Count missing predictions as errors, with an explicit fourth column."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("Labels and predictions must be aligned vectors.")
    if not np.isin(labels, (0, 1, 2)).all() or not np.isin(predictions, (-1, 0, 1, 2)).all():
        raise ValueError("Unknown target or predicted class.")
    labels, predictions = labels.astype(int), predictions.astype(int)
    matrix = np.zeros((3, 4), dtype=np.int64)
    np.add.at(matrix, (labels, np.where(predictions < 0, 3, predictions)), 1)
    correct = int(sum(matrix[index, index] for index in range(3)))
    precision, recall, f1 = [], [], []
    for index in range(3):
        tp, predicted, support = matrix[index, index], matrix[:, index].sum(), matrix[index].sum()
        p = float(tp / predicted) if predicted else 0.0
        r = float(tp / support) if support else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "total": int(labels.size), "correct": correct,
        "accuracy": correct / labels.size if labels.size else None,
        "macro_f1": float(np.mean(f1)) if labels.size else None,
        "macro_precision": float(np.mean(precision)) if labels.size else None,
        "macro_recall": float(np.mean(recall)) if labels.size else None,
        "missing_predictions": int(np.count_nonzero(predictions < 0)),
        "confusion_matrix": matrix.tolist(),
        "confusion_columns": ["RS", "CH", "FS", "missing"],
    }


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Multiclass NLL/Brier and 10-bin top-label ECE, without fitting a calibrator."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != (labels.size, 3) or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Expected canonical labels and three aligned probability columns.")
    labels = labels.astype(int)
    valid = np.isfinite(probabilities).all(axis=1)
    labels, probabilities = labels[valid], probabilities[valid]
    if not labels.size:
        return {"evaluated": 0, "nll": None, "brier": None, "ece": None}
    if np.any((probabilities < 0) | (probabilities > 1)) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Finite probabilities must lie in [0, 1] and sum to one.")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == index) * abs(np.mean(confidence[bins == index]) - np.mean(correct[bins == index]))
        for index in range(10) if np.any(bins == index)
    )
    return {
        "evaluated": int(labels.size),
        "nll": float(-np.log(np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1)).mean()),
        "brier": float(np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1).mean()),
        "ece": float(ece),
    }
