"""Generate data, train the classifier and produce held-out evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from benchmark_models import benchmark_models
from config import CLASS_CODES, CLASS_NAMES, SimulationConfig
from data_pipeline import generate_dataset, load_split, read_metadata
from metrics import classification_report
from models import NeuronClassifier1D


PROJECT_ROOT = Path(__file__).resolve().parent


class TraceDataset(Dataset):
    def __init__(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        mean: float,
        std: float,
    ) -> None:
        normalized = (np.asarray(traces, dtype=np.float32) - mean) / std
        self.traces = torch.from_numpy(normalized[:, np.newaxis, :])
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.traces[index], self.labels[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            targets.append(labels.detach().cpu().numpy())
            predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    return (
        total_loss / max(total_examples, 1),
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_CODES])
        for code, row in zip(CLASS_CODES, matrix, strict=True):
            writer.writerow([code, *row])


def _write_results_summary(
    path: Path,
    test_report: dict[str, object],
    benchmark_report: dict[str, object],
) -> None:
    target_status = "ATINGIDA" if test_report["target_met"] else "NÃO ATINGIDA"
    lines = [
        "# Resumo automático dos resultados",
        "",
        "> Este texto é gerado a partir do conjunto de teste retido. Deve ser ",
        "> revisado e incorporado ao relatório final, não copiado sem contexto.",
        "",
        "## Classificação Izhikevich",
        "",
        f"- Acurácia no teste: **{test_report['accuracy']:.2%}**.",
        f"- Meta institucional: **{test_report['target_accuracy']:.2%} — {target_status}**.",
        f"- F1 macro: **{test_report['macro_f1']:.3f}**.",
        f"- Amostras de teste: **{test_report['total']}**.",
        f"- Faixa de estímulo observada no teste: **{test_report['held_out_current_range'][0]:.2f}–"
        f"{test_report['held_out_current_range'][1]:.2f} unidades do modelo**.",
        "",
        "### Resultados por classe",
        "",
        "| Classe | Precisão | Recall | F1 | Suporte |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CLASS_NAMES:
        values = test_report["per_class"][name]
        lines.append(
            f"| {name} | {values['precision']:.3f} | {values['recall']:.3f} | "
            f"{values['f1']:.3f} | {values['support']} |"
        )
    lines.extend([
        "",
        "## Custo computacional da simulação",
        "",
        f"- LIF: **{benchmark_report['lif']['microseconds_per_trace']:.2f} µs/traço**.",
        f"- Izhikevich: **{benchmark_report['izhikevich']['microseconds_per_trace']:.2f} µs/traço**.",
        f"- Razão Izhikevich/LIF: **{benchmark_report['izhikevich_to_lif_runtime_ratio']:.2f}×**.",
        "",
        "O benchmark mede custo computacional, não equivalência biológica. Os modelos "
        "possuem equações, estados e escalas de entrada diferentes.",
        "",
        "## Limitação obrigatória",
        "",
        "Os sinais são sintéticos e a avaliação mede generalização dentro da distribuição "
        "simulada. O resultado não demonstra funcionamento com registros neurais reais.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[SYSTEM] Training device: {device}")

    simulation_config = SimulationConfig(
        time_ms=args.time_ms,
        dt_ms=args.dt_ms,
        current_min=args.current_min,
        current_max=args.current_max,
        parameter_jitter=args.parameter_jitter,
        observation_noise_std=args.noise_std,
        minimum_spikes=args.minimum_spikes,
    )

    if args.regenerate or not args.database.exists():
        print(f"[DATA] Generating SQLite dataset at {args.database}")
        generate_dataset(
            args.database,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            config=simulation_config,
            overwrite=args.regenerate,
        )

    dataset_metadata = read_metadata(args.database)
    stored_simulation = dataset_metadata["simulation"]
    simulation_config = SimulationConfig(
        time_ms=float(stored_simulation["time_ms"]),
        dt_ms=float(stored_simulation["dt_ms"]),
        current_min=float(stored_simulation["current_min"]),
        current_max=float(stored_simulation["current_max"]),
        parameter_jitter=float(stored_simulation["parameter_jitter"]),
        observation_noise_std=float(stored_simulation["observation_noise_std"]),
        minimum_spikes=int(stored_simulation["minimum_spikes"]),
    )
    train_traces, train_labels, _ = load_split(args.database, "train")
    validation_traces, validation_labels, _ = load_split(args.database, "validation")
    test_traces, test_labels, test_sample_metadata = load_split(args.database, "test")

    train_mean = float(train_traces.mean(dtype=np.float64))
    train_std = float(train_traces.std(dtype=np.float64))
    if train_std < 1e-6:
        raise RuntimeError("Training traces have near-zero variance.")

    datasets = {
        "train": TraceDataset(train_traces, train_labels, train_mean, train_std),
        "validation": TraceDataset(validation_traces, validation_labels, train_mean, train_std),
        "test": TraceDataset(test_traces, test_labels, train_mean, train_std),
    }
    pin_memory = device.type == "cuda"
    loaders = {
        name: _loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            workers=args.workers,
            seed=args.seed + index,
            pin_memory=pin_memory,
        )
        for index, (name, dataset) in enumerate(datasets.items())
    }

    model = NeuronClassifier1D(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
        min_lr=1e-6,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print("[TRAIN] Starting mini-batch optimization")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_targets, train_predictions = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_targets, validation_predictions = _run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
        )
        scheduler.step(validation_loss)
        train_accuracy = float(np.mean(train_targets == train_predictions))
        validation_accuracy = float(np.mean(validation_targets == validation_predictions))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.3f} | "
            f"val loss {validation_loss:.4f} acc {validation_accuracy:.3f}"
        )

        if validation_loss < best_validation_loss - args.minimum_delta:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[TRAIN] Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")
    model.load_state_dict(best_state)

    test_loss, test_targets, test_predictions = _run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
    )
    test_report = classification_report(test_targets, test_predictions, CLASS_NAMES)
    test_report["loss"] = test_loss
    test_report["target_accuracy"] = args.target_accuracy
    test_report["target_met"] = bool(test_report["accuracy"] >= args.target_accuracy)
    test_report["held_out_current_range"] = [
        float(test_sample_metadata["input_current"].min()),
        float(test_sample_metadata["input_current"].max()),
    ]

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "architecture": "NeuronClassifier1D",
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "class_codes": list(CLASS_CODES),
        "class_names": list(CLASS_NAMES),
        "normalization": {"mean": train_mean, "std": train_std},
        "simulation": dataset_metadata["simulation"],
        "confidence_threshold": args.confidence_threshold,
        "test_metrics": test_report,
        "seed": args.seed,
    }
    temporary_checkpoint = args.weights.with_suffix(args.weights.suffix + ".tmp")
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, args.weights)

    benchmark_report = benchmark_models(
        samples=args.benchmark_samples,
        repeats=args.benchmark_repeats,
        seed=args.seed,
        config=simulation_config,
    )

    args.artifacts.mkdir(parents=True, exist_ok=True)
    _write_history(args.artifacts / "training_history.csv", history)
    _write_confusion_matrix(args.artifacts / "confusion_matrix.csv", test_report["confusion_matrix"])
    (args.artifacts / "test_metrics.json").write_text(
        json.dumps(test_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.artifacts / "model_benchmark.json").write_text(
        json.dumps(benchmark_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_results_summary(
        args.artifacts / "results_summary.md",
        test_report,
        benchmark_report,
    )

    final_report = {
        "checkpoint": str(args.weights),
        "database": str(args.database),
        "epochs_completed": len(history),
        "best_validation_loss": best_validation_loss,
        "test": test_report,
        "benchmark": benchmark_report,
    }
    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    if not test_report["target_met"]:
        print(
            f"[WARNING] Held-out accuracy {test_report['accuracy']:.3f} did not "
            f"reach the required {args.target_accuracy:.3f}."
        )
    return final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "neural_patterns.sqlite",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "classifier.pt",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-ms", type=float, default=200.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--current-min", type=float, default=5.0)
    parser.add_argument("--current-max", type=float, default=30.0)
    parser.add_argument("--parameter-jitter", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=0.20)
    parser.add_argument("--minimum-spikes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--target-accuracy", type=float, default=0.75)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--benchmark-samples", type=int, default=256)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
