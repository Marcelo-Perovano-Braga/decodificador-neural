"""Frozen-checkpoint robustness studies and leakage-conscious baseline fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import asdict
from html import escape
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from config import CLASS_CODES, SimulationConfig
from data_pipeline import load_split, read_metadata
from inference import _gather_traces
from models import NeuronClassifier1D
from network import NetworkConfig
from research_features import FEATURE_NAMES, TimingCentroid, extract_timing_features, prediction_metrics, probability_metrics
from research_protocol import Condition, generate_cohort


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_digests(traces: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.asarray(row, dtype="<f4").tobytes()).hexdigest() for row in traces}


def load_frozen_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    codes = checkpoint["class_codes"]
    if len(codes) != 3 or set(codes) != set(CLASS_CODES):
        raise ValueError("Checkpoint must contain exactly RS, CH and FS.")
    normalization = checkpoint["normalization"]
    if not np.isfinite([normalization["mean"], normalization["std"]]).all() or normalization["std"] <= 0:
        raise ValueError("Checkpoint normalization must be finite with positive standard deviation.")
    missing_settings = set(SimulationConfig.__dataclass_fields__) - set(checkpoint["simulation"])
    if missing_settings:
        raise ValueError("Checkpoint lacks recorded simulation settings: " + ", ".join(sorted(missing_settings)))
    config = SimulationConfig(**{
        name: value for name, value in checkpoint["simulation"].items()
        if name in SimulationConfig.__dataclass_fields__
    })
    if not np.isfinite(list(config.to_dict().values())).all() or config.window_size < 2:
        raise ValueError("Invalid checkpoint simulation settings.")
    if checkpoint["simulation"].get("window_size", config.window_size) != config.window_size:
        raise ValueError("Checkpoint window length disagrees with its simulation settings.")
    threshold = float(checkpoint.get("confidence_threshold", 0.7))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint confidence threshold.")
    temperature = float(checkpoint.get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid checkpoint calibration temperature.")
    model = NeuronClassifier1D().eval()
    model.load_state_dict(checkpoint["model_state"])
    model = model.for_inference()
    model.requires_grad_(False)
    return model, checkpoint, config


def frozen_probabilities(traces, model, checkpoint, *, batch_size=64) -> np.ndarray:
    """Diagnostic inference: no current-range, activity or confidence rejection.

    Columns are remapped to canonical RS/CH/FS order. Nonfinite input/output rows
    stay missing. No labels, clean spikes, parameters or stimulus metadata enter
    either predictor.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    probabilities = np.full((len(traces), 3), np.nan)
    finite = np.asarray([np.isfinite(row).all() for row in traces])
    indices = np.flatnonzero(finite).tolist()
    order = [checkpoint["class_codes"].index(code) for code in CLASS_CODES]
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            inputs = _gather_traces(traces, rows)
            inputs -= float(checkpoint["normalization"]["mean"])
            inputs /= float(checkpoint["normalization"]["std"])
            logits = model(torch.from_numpy(inputs).unsqueeze(1))
            if tuple(logits.shape) != (len(rows), 3):
                raise ValueError("Unexpected frozen model output shape.")
            probabilities[rows] = torch.softmax(logits / float(checkpoint.get("temperature", 1.0)), dim=1).cpu().numpy()[:, order]
    return probabilities


def fit_baselines(config, *, database: Path | None, synthetic_samples: int, burst_isi_ms: float):
    if database is not None:
        database_hash = file_digest(database)
        metadata = read_metadata(database)
        if tuple(metadata["class_codes"]) != CLASS_CODES:
            raise ValueError("Database class order differs from the canonical order.")
        if any(metadata["simulation"].get(name) != value for name, value in config.to_dict().items()):
            raise ValueError("Database settings differ from checkpoint settings; use the original training database.")
        traces, labels, _ = load_split(database, "train")
        source = {
            "kind": "sqlite_train_split_only", "path": str(database.resolve()),
            "sha256": database_hash,
            "provenance_limit": "Recorded settings match. Legacy checkpoints do not store a training-database hash; original dataset identity is supplied by the operator.",
        }
    else:
        condition = Condition("baseline_fit", "fit", 0, config.observation_noise_std, config.parameter_jitter)
        cohort = generate_cohort(condition, config, synthetic_samples, 17, namespace=0)
        traces, labels = cohort.traces, cohort.labels
        source = {
            "kind": "explicit_synthetic_fit", "seed": 17, "namespace": 0,
            "provenance_limit": "New constant-current training cohort; not the original CNN training rows. Comparison is exploratory, not identical-data model selection.",
        }
    features, counts, finite = extract_timing_features(traces, config.dt_ms, burst_isi_ms=burst_isi_ms)
    usable = finite if database is not None else finite & (counts >= config.minimum_spikes)
    baselines = {
        "rate_centroid": TimingCentroid.fit(features[usable], labels[usable], columns=(0,)),
        "timing_centroid": TimingCentroid.fit(features[usable], labels[usable]),
    }
    source.update({
        "rows_loaded": int(len(labels)), "rows_fitted": int(usable.sum()),
        "class_counts_fitted": np.bincount(labels[usable], minlength=3).tolist(),
        "training_profiles": ["constant"],
        "training_current_range": [config.current_min, config.current_max],
    })
    # Include all loaded training traces, even rows excluded from feature fitting.
    return baselines, source, trace_digests(traces)


def score_condition(cohort, features, observed_counts, finite, probabilities, baselines, config, threshold):
    valid_probabilities = np.isfinite(probabilities).all(axis=1)
    predictions = np.full(len(cohort.labels), -1, dtype=int)
    confidence = np.full(len(cohort.labels), np.nan)
    predictions[valid_probabilities] = probabilities[valid_probabilities].argmax(axis=1)
    confidence[valid_probabilities] = probabilities[valid_probabilities].max(axis=1)
    active = finite & (observed_counts >= config.minimum_spikes)
    outputs = {"cnn_frozen": predictions}
    prediction_seconds = {}
    for name, baseline in baselines.items():
        started = perf_counter()
        outputs[name] = baseline.predict(features)
        outputs[name][~finite] = -1
        prediction_seconds[name] = perf_counter() - started
    scores = {}
    for name, output in outputs.items():
        score = prediction_metrics(cohort.labels, output)
        score["active_only"] = prediction_metrics(cohort.labels[active], output[active])
        if name == "cnn_frozen":
            accepted = active & valid_probabilities & (confidence >= threshold)
            score["selective"] = {
                "policy": "finite observed trace, minimum observed peaks, confidence >= checkpoint threshold; no prior current-range rejection",
                "coverage": float(accepted.mean()),
                "accepted": int(accepted.sum()),
                "metrics": prediction_metrics(cohort.labels[accepted], output[accepted]),
            }
            score["probability_metrics"] = probability_metrics(cohort.labels, probabilities)
        scores[name] = score
    return scores, outputs, confidence, active, prediction_seconds


def aggregate_results(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        for model, score in result["scores"].items():
            groups[(result["condition"]["name"], model)].append((result, score))
    aggregates = []
    for (name, model), values in groups.items():
        accuracies = [score["accuracy"] for _, score in values]
        correct = sum(score["correct"] for _, score in values)
        total = sum(score["total"] for _, score in values)
        aggregate = {
            "condition": name, "model": model, "replicate_seeds": [row["seed"] for row, _ in values],
            "correct": correct, "total": total, "pooled_accuracy": correct / total,
            "mean_seed_accuracy": float(np.mean(accuracies)),
            "std_seed_accuracy": float(np.std(accuracies, ddof=1)) if len(values) > 1 else None,
            "mean_seed_macro_f1": float(np.mean([score["macro_f1"] for _, score in values])),
        }
        if model == "cnn_frozen":
            accepted = sum(score["selective"]["accepted"] for _, score in values)
            accepted_correct = sum(score["selective"]["metrics"]["correct"] for _, score in values)
            aggregate["selective_coverage"] = accepted / total
            aggregate["selective_accuracy"] = accepted_correct / accepted if accepted else None
        aggregates.append(aggregate)
    return aggregates


def write_reports(output: Path, report: dict) -> None:
    (output / "study.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["condition", "seed", "axis", "value", "profile", "held_out_reason", "model", "correct", "total", "accuracy", "macro_f1", "active_accuracy", "active_coverage", "selective_accuracy", "selective_coverage", "physical_spikes", "synapses", "inference_seconds", "nll", "brier", "ece"]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in report["results"]:
            for model, score in row["scores"].items():
                writer.writerow({
                    "condition": row["condition"]["name"], "seed": row["seed"],
                    **{key: row["condition"][key] for key in ("axis", "value", "profile", "held_out_reason")},
                    "model": model, **{key: score[key] for key in ("correct", "total", "accuracy", "macro_f1")},
                    "active_accuracy": score["active_only"]["accuracy"], "active_coverage": row["active_coverage"],
                    "selective_accuracy": score.get("selective", {}).get("metrics", {}).get("accuracy"),
                    "selective_coverage": score.get("selective", {}).get("coverage"),
                    **{key: row[key] for key in ("physical_spikes", "synapses")},
                    "inference_seconds": row["cnn_seconds"] if model == "cnn_frozen" else row["feature_extraction_seconds"] + row["baseline_prediction_seconds"][model],
                    **{key: score.get("probability_metrics", {}).get(key) for key in ("nll", "brier", "ece")},
                })
    def percent(value):
        return f"{value:.4%}" if value is not None else "unavailable"

    aggregate_index = {(row["condition"], row["model"]): row for row in report["aggregates"]}
    condition_index = {row["condition"]["name"]: row["condition"] for row in report["results"]}
    reference = aggregate_index.get(("reference", "cnn_frozen"))
    comparison_header = ["Condition", "Protocol", "CNN", "Rate baseline", "Timing baseline", "CNN change vs reference (pp)", "CNN coverage", "CNN accepted accuracy"]
    comparison_rows = []
    for name, condition in condition_index.items():
        cnn = aggregate_index[(name, "cnn_frozen")]
        difference = 100 * (cnn["pooled_accuracy"] - reference["pooled_accuracy"]) if reference else None
        comparison_rows.append([
            name, "held out" if condition["held_out_reason"] else condition["axis"],
            percent(cnn["pooled_accuracy"]), percent(aggregate_index[(name, "rate_centroid")]["pooled_accuracy"]),
            percent(aggregate_index[(name, "timing_centroid")]["pooled_accuracy"]),
            f"{difference:+.4f}" if difference is not None else "unavailable",
            percent(cnn["selective_coverage"]), percent(cnn["selective_accuracy"]),
        ])
    lines = [
        "# Frozen model robustness study", "", report["label_meaning"], "",
        f"Checkpoint SHA-256: `{report['checkpoint_sha256']}`", "",
        "Accuracy below counts every requested neuron, including silent examples. Missing predictions count as errors.",
        "Confidence/activity-based rejection is reported separately in study.json and summary.csv.",
        "Standard deviation is across seed replicates; neurons within a connected network are not independent replicates.",
        "", f"Baseline source: **{report['baseline_source']['kind']}**. {report['baseline_source']['provenance_limit']}",
        "", "## Robustness map", "",
        "| " + " | ".join(comparison_header) + " |",
        "|" + "---|" * len(comparison_header),
        *["| " + " | ".join(row) + " |" for row in comparison_rows],
        "", "## Exact counts and seed variation", "",
        "| Condition | Model | Correct / total | Accuracy | Seed SD | Macro F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    table_rows = []
    for aggregate in report["aggregates"]:
        sd = aggregate["std_seed_accuracy"]
        sd_text = f"{sd:.4%}" if sd is not None else "unavailable (one seed)"
        values = [aggregate["condition"], aggregate["model"], f"{aggregate['correct']}/{aggregate['total']}",
                  f"{aggregate['pooled_accuracy']:.4%}", sd_text, f"{aggregate['mean_seed_macro_f1']:.6f}"]
        lines.append("| " + " | ".join(values) + " |")
        table_rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>")
    lines.extend(["", "## Interpretation and held-out status", ""] + [f"- {note}" for note in report["interpretation"]])
    notes = "".join(f"<li>{escape(note)}</li>" for note in report["interpretation"])
    comparison_html = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in comparison_rows)
    held_out_html = "".join(f"<li><b>{escape(name)}</b>: {escape(condition['held_out_reason'])}</li>" for name, condition in condition_index.items() if condition["held_out_reason"])
    if held_out_html:
        lines.extend(["", "## Held-out conditions", ""] + [f"- {name}: {condition['held_out_reason']}" for name, condition in condition_index.items() if condition["held_out_reason"]])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    html = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>Frozen model robustness study</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#233044;background:#f5f7fb}"
        "table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border:1px solid #cdd6df;text-align:left}"
        "th{background:#dff3ec}tr:nth-child(even){background:#f2f6f9}code{overflow-wrap:anywhere}.table{overflow:auto}li{margin:8px 0}</style>"
        "<h1>Frozen model robustness study</h1>"
        f"<p>{escape(report['label_meaning'])}</p><p>Checkpoint: <code>{report['checkpoint_sha256']}</code></p>"
        f"<p>Baseline: <b>{escape(report['baseline_source']['kind'])}</b>. {escape(report['baseline_source']['provenance_limit'])}</p>"
        "<p>Accuracy includes all requested neurons. Confidence rejection is separate. Seed SD measures variation between seed runs.</p>"
        "<h2>Robustness map</h2><p>All-model accuracy uses the full population. Accepted accuracy uses only the CNN's accepted subset; read it alongside coverage. Changes are percentage points relative to the paired reference.</p>"
        "<div class='table'><table><thead><tr>" + "".join(f"<th>{escape(value)}</th>" for value in comparison_header)
        + "</tr></thead><tbody>" + comparison_html + "</tbody></table></div>"
        + (f"<h2>Held-out conditions</h2><ul>{held_out_html}</ul>" if held_out_html else "")
        + "<h2>Exact counts and seed variation</h2><div class='table'><table><thead><tr><th>Condition</th><th>Model</th><th>Correct / total</th><th>Accuracy</th><th>Seed SD</th><th>Macro F1</th></tr></thead><tbody>"
        + "".join(table_rows) + f"</tbody></table></div><h2>Interpretation</h2><ul>{notes}</ul></html>"
    )
    (output / "report.html").write_text(html, encoding="utf-8")


def run_study(
    *, checkpoint_path: Path, database: Path | None, output: Path,
    conditions: list[Condition], seeds: list[int], samples_per_class=100,
    synthetic_samples=500, out_degree=100, batch_size=64, burst_isi_ms=5.0,
    progress=print,
) -> dict:
    if not conditions or len({condition.name for condition in conditions}) != len(conditions):
        raise ValueError("Choose nonempty uniquely named conditions.")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Choose distinct nonnegative seeds.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty; existing research files are preserved.")
    checkpoint_hash = file_digest(checkpoint_path)
    model, checkpoint, config = load_frozen_checkpoint(checkpoint_path)
    baselines, source, training_hashes = fit_baselines(
        config, database=database, synthetic_samples=synthetic_samples, burst_isi_ms=burst_isi_ms,
    )
    if checkpoint.get("training_protocol"):
        source["provenance_limit"] += " This baseline-fit cohort is separate from the robust CNN's augmented training corpus; this is not an identical-training-data architecture comparison."
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status_path.write_text(json.dumps({"status": "running"}) + "\n")
    try:
        results = []
        with (output / "predictions.csv.partial").open("w", newline="", encoding="utf-8") as handle:
            fields = ["condition", "seed", "neuron", "label", "a", "b", "c", "d", "amplitude", "physical_spikes", "observed_peaks", "outside_recorded_drive_range", *FEATURE_NAMES, "cnn_prediction", "cnn_confidence", "cnn_accepted", "rate_centroid", "timing_centroid"]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for condition in conditions:
                for seed in seeds:
                    progress(f"[{len(results) + 1}/{len(conditions) * len(seeds)}] {condition.name} · seed {seed}")
                    cohort = generate_cohort(condition, config, samples_per_class, seed, out_degree=out_degree)
                    if trace_digests(cohort.traces) & training_hashes:
                        raise ValueError("Exact training-trace overlap detected. Choose a different evaluation seed.")
                    started = perf_counter()
                    features, counts, finite = extract_timing_features(cohort.traces, config.dt_ms, burst_isi_ms=burst_isi_ms)
                    feature_seconds = perf_counter() - started
                    started = perf_counter()
                    probabilities = frozen_probabilities(cohort.traces, model, checkpoint, batch_size=batch_size)
                    cnn_seconds = perf_counter() - started
                    scores, predictions, confidence, active, baseline_seconds = score_condition(
                        cohort, features, counts, finite, probabilities, baselines, config,
                        float(checkpoint.get("confidence_threshold", 0.7)),
                    )
                    accepted_threshold = float(checkpoint.get("confidence_threshold", 0.7))
                    for row, label in enumerate(cohort.labels):
                        writer.writerow({
                            "condition": condition.name, "seed": seed, "neuron": row + 1, "label": CLASS_CODES[label],
                            **{name: float(values[row]) for name, values in cohort.parameters.items()},
                            "amplitude": float(cohort.amplitudes[row]),
                            "physical_spikes": int(cohort.physical_spike_counts[row]), "observed_peaks": int(counts[row]),
                            "outside_recorded_drive_range": bool(cohort.outside_recorded_drive_range[row]),
                            **dict(zip(FEATURE_NAMES, features[row].tolist(), strict=True)),
                            "cnn_prediction": CLASS_CODES[predictions["cnn_frozen"][row]] if predictions["cnn_frozen"][row] >= 0 else "missing",
                            "cnn_confidence": float(confidence[row]) if np.isfinite(confidence[row]) else "",
                            "cnn_accepted": bool(active[row] and np.isfinite(confidence[row]) and confidence[row] >= accepted_threshold),
                            **{name: CLASS_CODES[predictions[name][row]] if predictions[name][row] >= 0 else "missing" for name in baselines},
                        })
                    handle.flush()
                    condition_record = asdict(condition)
                    if condition.current_range is not None:
                        condition_record["current_range"] = list(condition.current_range)
                    results.append({
                        "condition": condition_record, "seed": seed, "scores": scores,
                        "active_coverage": float(active.mean()),
                        "physical_spikes": int(cohort.physical_spike_counts.sum()),
                        "physical_mean_rate_hz": float(cohort.physical_spike_counts.mean() / (cohort.traces.shape[1] * config.dt_ms / 1000)),
                        "observed_peaks": int(counts.sum()), "synapses": cohort.synapses,
                        "outside_recorded_drive_range": int(cohort.outside_recorded_drive_range.sum()),
                        "cnn_seconds": cnn_seconds, "feature_extraction_seconds": feature_seconds,
                        "baseline_prediction_seconds": baseline_seconds,
                    })
        if file_digest(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during the study; results cannot be marked frozen.")
        if database is not None and file_digest(database) != source["sha256"]:
            raise RuntimeError("Training database changed during the study.")
        report = {
            "schema_version": 1, "status": "complete", "checkpoint_sha256": checkpoint_hash,
            "checkpoint_path": str(checkpoint_path.resolve()), "baseline_source": source,
            "cnn_normalization": checkpoint["normalization"],
            "cnn_confidence_threshold": float(checkpoint.get("confidence_threshold", 0.7)),
            "cnn_temperature": float(checkpoint.get("temperature", 1.0)),
            "cnn_training_profiles": checkpoint.get("training_profiles", ["constant"]),
            "cnn_class_order": list(checkpoint["class_codes"]),
            "config": config.to_dict(), "seeds": seeds, "samples_per_class": samples_per_class,
            "out_degree_requested": out_degree, "burst_isi_ms": burst_isi_ms,
            "protocol_version": 1, "evaluation_rng_namespace": 1,
            "network_settings": asdict(NetworkConfig(out_degree=out_degree)),
            "cpu_threads": torch.get_num_threads(),
            "source_sha256": {
                name: file_digest(Path(__file__).resolve().parent / name)
                for name in ("evaluate_research.py", "research_study.py", "research_protocol.py", "research_features.py",
                             "simulation.py", "network.py", "models.py", "config.py", "data_pipeline.py", "inference.py")
            },
            "label_meaning": "Agreement with the RS/CH/FS parameter family used to generate each neuron; not a biological cell diagnosis or a label of its response under every input.",
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
            "baseline_models": {name: baseline.to_dict() for name, baseline in baselines.items()},
            "interpretation": [
                "CNN weights and training normalization are frozen. No calibration, fine-tuning or test-based model selection is performed.",
                "Held-out stimulus families are determined using checkpoint training_profiles (legacy default: constant). External current ranges refer to constant-current base amplitudes; augmented waveforms may reach those instantaneous values. Historical checkpoint provenance beyond its metadata cannot be proven by this tool.",
                "Gaussian parameter jitter has unbounded support: larger jitter is a distribution shift, not a strictly disjoint held-out parameter band.",
                "Noise, jitter and network-strength sweeps use paired latent draws within each seed. Pulse-onset comparisons are within a withheld pulse family, not one-factor comparisons with a constant stimulus.",
                "The current CLI/UI prior domain-rejection policy is bypassed for diagnostic CNN predictions. Secondary coverage uses observed activity and checkpoint confidence only; these predictions are not operational validation.",
                "Every requested neuron is retained, including quiet/ambiguous examples. Ground-truth spike events use the same pre-noise event counter for all conditions; predictors see observed voltages only.",
                "Baseline scaling and centroids use training rows only. Exact trace overlap with those rows is checked. This check does not certify all possible forms of leakage.",
                "Seed replicates, not neurons in a recurrent network, are the units for reported seed variation. No row-independent confidence interval or biological-realism claim is made.",
                "Baseline distances are not calibrated probabilities; selective confidence metrics are provided only for the CNN.",
            ],
            "results": results, "aggregates": aggregate_results(results),
        }
        write_reports(output, report)
        (output / "predictions.csv.partial").rename(output / "predictions.csv")
        status_path.write_text(json.dumps({"status": "complete", "checkpoint_sha256": checkpoint_hash}) + "\n")
        return report
    except BaseException as exc:
        status_path.write_text(json.dumps({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": str(exc)}) + "\n")
        raise
