"""Train a separate robust CNN and gate adoption using paired, fresh comparisons."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import CLASS_CODES, CLASS_NAMES, SimulationConfig
from inference import _gather_traces
from models import NeuronClassifier1D
from research_features import prediction_metrics, probability_metrics
from research_protocol import Condition, generate_cohort
from robust_protocols import (EXPANDED_NAMESPACE, EXPANDED_PROFILES, EXPANDED_SEEDS, EXPANDED_WITHHELD,
                              expanded_comparison_conditions, expanded_training_conditions)
from research_study import file_digest, frozen_probabilities, load_frozen_checkpoint, trace_digests
from simulation import count_spikes
from train_pipeline import TraceDataset, _loader, _run_epoch, seed_everything


# This protocol and these gates are fixed before generating the comparison data.
# They are engineering adoption criteria, not a statistical significance test.
COMPARISON_SEEDS = (6101, 6203, 6301, 6407, 6503)
COMPARISON_NAMESPACE = 23
POLICY = {
    "minimum_mean_accuracy_gain": .02,
    "minimum_pulse_accuracy_gain": .10,
    "minimum_reference_accuracy": .99,
    "maximum_reference_accuracy_drop": .01,
    "maximum_single_condition_accuracy_drop": .03,
    "maximum_mean_coverage_drop": .05,
    "minimum_relative_accepted_error_reduction": .15,
    "require_nll_improvement": True,
    "require_brier_nonregression": True,
    "require_positive_gain_each_seed": True,
}


def training_conditions(config: SimulationConfig, protocol_version=1) -> list[Condition]:
    if protocol_version not in (1, 2):
        raise ValueError("Unknown candidate protocol version.")
    conditions = []
    for index, (noise, jitter) in enumerate(((config.observation_noise_std, config.parameter_jitter),
                                           (2., .06), (5., .06), (10., .06), (.2, .12))):
        conditions.append(Condition(f"constant_{index}", "fit_constant", noise, noise, jitter))
    for index, (onset, duration, noise, jitter) in enumerate((
        (.05, .25, .2, .03), (.25, .5, .2, .03), (.45, .5, .2, .03),
        (0., .75, 2., .06), (.2, .25, 5., .06), (.4, .5, 10., .06),
        (.5, .5, .2, .03), (.7, .3, 2., .06), (.35, .65, 5., .06), (.1, .9, .2, .03),
    )):
        conditions.append(Condition(f"pulse_{index}", "fit_pulse", onset, noise, jitter,
                                    profile="pulse", onset_fraction=onset, pulse_duration_fraction=duration))
    for index, (noise, jitter) in enumerate(((.2, .03), (5., .06), (10., .06))):
        conditions.append(Condition(f"ramp_{index}", "fit_ramp", noise, noise, jitter, profile="ramp"))
    for index, noise in enumerate((.2, 5.)):
        conditions.append(Condition(f"sine_{index}", "fit_sine", noise, noise, .03, profile="sine"))
    for value, jitter in ((.15, .03), (.4, .06), (.8, .06)):
        conditions.append(Condition(f"network_{value:g}", "fit_network", value, .2, jitter, network_weight=value))
    conditions.append(Condition("pulse_network", "fit_network", .3, 2., .06, profile="pulse",
                                onset_fraction=.25, network_weight=.3))
    conditions.append(Condition("ramp_network", "fit_network", .4, .2, .03, profile="ramp", network_weight=.4))
    return expanded_training_conditions(conditions) if protocol_version == 2 else conditions


def comparison_conditions(config: SimulationConfig, protocol_version=1) -> list[Condition]:
    if protocol_version not in (1, 2):
        raise ValueError("Unknown candidate protocol version.")
    n, j = config.observation_noise_std, config.parameter_jitter
    conditions = [Condition("reference", "reference", 0, n, j)]
    conditions.extend(Condition(f"noise_{v:g}", "noise", v, v, j) for v in (2., 5., 10.))
    conditions.extend(Condition(f"jitter_{v:g}", "jitter", v, n, v) for v in (.06, .1, .15))
    conditions.extend(Condition(f"network_{v:g}", "network", v, n, j, network_weight=v) for v in (.1, .5, 1.))
    conditions.extend(Condition(f"pulse_{v:g}", "pulse", v, n, j, profile="pulse", onset_fraction=v)
                      for v in (.1, .3, .5))
    conditions.append(Condition("pulse_short_late", "pulse", .55, n, j, profile="pulse",
                                onset_fraction=.55, pulse_duration_fraction=.35))
    for profile in ("ramp", "sine"):
        conditions.append(Condition(profile, "familiar_stimulus", 0, n, j, profile=profile))
    for profile in ("double_pulse", "triangle"):
        conditions.append(Condition(profile, "withheld_family", 0, n, j, profile=profile,
                                    held_out_reason="stimulus family absent from candidate fitting and calibration"))
    span = config.current_max - config.current_min
    for name, bounds in (
        ("current_below", (config.current_min - span / 2, float(np.nextafter(np.float32(config.current_min), np.float32(-np.inf))))),
        ("current_above", (float(np.nextafter(np.float32(config.current_max), np.float32(np.inf))), config.current_max + span / 2)),
    ):
        conditions.append(Condition(name, "withheld_amplitude", 0, n, j, current_range=bounds,
                                    held_out_reason="constant-current amplitude outside candidate fit range"))
    return expanded_comparison_conditions(config, conditions) if protocol_version == 2 else conditions


def comparison_plan(config, protocol_version=1):
    """Serializable reservation made before fitting; no generated test examples."""
    conditions = comparison_conditions(config, protocol_version)
    return {"protocol_version": protocol_version, "config": config.to_dict(), "policy": dict(POLICY),
            "seeds": list(EXPANDED_SEEDS if protocol_version == 2 else COMPARISON_SEEDS),
            "namespace": EXPANDED_NAMESPACE if protocol_version == 2 else COMPARISON_NAMESPACE,
            "conditions": json.loads(json.dumps([asdict(condition) for condition in conditions])),
            "withheld_profiles": list(EXPANDED_WITHHELD if protocol_version == 2 else ("double_pulse", "triangle"))}


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Corpus:
    traces: np.ndarray
    labels: np.ndarray
    conditions: list[dict]


def generate_corpus(config, samples_per_class, seed, split, *, out_degree=100, protocol_version=1, progress=print):
    if split not in ("train", "selection", "calibration"):
        raise ValueError("Unknown fitting split.")
    namespace = {"train": 20, "selection": 21, "calibration": 22}[split] + (10 if protocol_version == 2 else 0)
    traces, labels, records = [], [], []
    conditions = training_conditions(config, protocol_version)
    for index, condition in enumerate(conditions):
        cohort_seed = int(np.random.SeedSequence([seed, namespace, index]).generate_state(1)[0])
        progress(f"[DATA] {split} {index + 1}/{len(conditions)}: {condition.name}")
        cohort = generate_cohort(condition, config, samples_per_class, cohort_seed,
                                 out_degree=out_degree, namespace=namespace)
        if not np.isfinite(cohort.traces).all():
            raise ValueError(f"Nonfinite simulation in {split}/{condition.name}.")
        traces.append(np.asarray(cohort.traces, order="C"))
        labels.append(cohort.labels)
        records.append({"condition": asdict(condition), "seed": cohort_seed, "namespace": namespace,
                        "rows": len(cohort.labels), "synapses": cohort.synapses,
                        "quiet_rows": int(np.count_nonzero(cohort.physical_spike_counts < config.minimum_spikes))})
    return Corpus(np.concatenate(traces), np.concatenate(labels), records)


def batch_logits(traces, model, normalization, batch_size=96):
    model.eval()
    output = np.empty((len(traces), 3), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(traces), batch_size):
            end = min(start + batch_size, len(traces))
            inputs = _gather_traces(traces, list(range(start, end)))
            inputs -= normalization["mean"]
            inputs /= normalization["std"]
            output[start:end] = model(torch.from_numpy(inputs).unsqueeze(1)).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite candidate logits.")
    return output


def fit_temperature(logits, labels):
    """Fit one positive scalar on a separate calibration split; leave argmax intact."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if logits.shape != (len(labels), 3) or not torch.isfinite(logits).all() or not len(labels):
        raise ValueError("Invalid calibration examples.")
    temperatures = np.unique(np.r_[1., np.exp(np.linspace(np.log(.1), np.log(10.), 121))])
    losses = [float(nn.functional.cross_entropy(logits / value, labels)) for value in temperatures]
    best = int(np.argmin(losses))
    return {"temperature": float(temperatures[best]),
            "nll_before": float(nn.functional.cross_entropy(logits, labels)),
            "nll_after": losses[best], "method": "fixed log-temperature grid, calibration split only",
            "rows": len(labels), "bounds": [.1, 10.]}


def selection_summary(logits, corpus):
    """Score development data by mean, worst condition and worst class NLL.

    A clean-reference accuracy floor is checked before ranking eligible epochs.
    None of these examples comes from the reserved confirmation protocol.
    """
    losses = nn.functional.cross_entropy(torch.as_tensor(logits, dtype=torch.float64),
                                         torch.as_tensor(corpus.labels), reduction="none").numpy()
    predicted = np.asarray(logits).argmax(axis=1)
    rows, offset = [], 0
    for record in corpus.conditions:
        end = offset + record["rows"]
        labels, guesses = corpus.labels[offset:end], predicted[offset:end]
        class_nll = [float(losses[offset:end][labels == code].mean()) for code in range(3)]
        recalls = [float(np.mean(guesses[labels == code] == code)) for code in range(3)]
        rows.append({"condition": record["condition"]["name"], "nll": float(losses[offset:end].mean()),
                     "accuracy": float(np.mean(labels == guesses)), "class_nll": class_nll, "class_recall": recalls})
        offset = end
    if offset != len(corpus.labels) or not rows or any(not np.isfinite(row["class_nll"]).all() for row in rows):
        raise ValueError("Invalid grouped selection corpus.")
    reference = next(row for row in rows if row["condition"] == "constant_0")["accuracy"]
    mean_nll = float(losses.mean())
    worst_nll = max(row["nll"] for row in rows)
    worst_class_nll = max(max(row["class_nll"]) for row in rows)
    return {"nll": mean_nll, "accuracy": float(np.mean(predicted == corpus.labels)),
            "reference_accuracy": reference, "reference_eligible": reference + 1e-12 >= .99,
            "minimum_condition_accuracy": min(row["accuracy"] for row in rows),
            "minimum_condition_class_recall": min(min(row["class_recall"]) for row in rows),
            "worst_condition_nll": worst_nll, "worst_condition_class_nll": worst_class_nll,
            "score": mean_nll + .5 * worst_nll + .25 * worst_class_nll + 10 * max(0., .99 - reference),
            "conditions": rows}


def train_candidate(config, output, *, train_per_class=200, validation_per_class=50,
                    epochs=40, patience=8, seed=17011, out_degree=100, batch_size=96,
                    protocol_version=1, progress=print):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be new or empty.")
    if min(train_per_class, validation_per_class, epochs, patience, batch_size) < 1 or seed < 0:
        raise ValueError("Training counts must be positive and seed nonnegative.")
    output.mkdir(parents=True, exist_ok=True)
    reserved = comparison_plan(config, protocol_version)
    (output / "reserved_comparison.json").write_text(json.dumps(reserved, indent=2) + "\n", encoding="utf-8")
    source_hashes = {name: file_digest(Path(__file__).parent / name) for name in
                     ("robust_candidate.py", "robust_protocols.py", "train_robust.py", "models.py", "train_pipeline.py",
                      "research_protocol.py", "network.py", "simulation.py")}
    seed_everything(seed)
    corpora = {name: generate_corpus(config, train_per_class if name == "train" else validation_per_class,
                                     seed, name, out_degree=out_degree, protocol_version=protocol_version, progress=progress)
               for name in ("train", "selection", "calibration")}
    hashes = {name: trace_digests(corpus.traces) for name, corpus in corpora.items()}
    if any(hashes[first] & hashes[second] for first, second in (("train", "selection"), ("train", "calibration"), ("selection", "calibration"))):
        raise ValueError("Exact trace overlap between fitting splits.")
    normalization = {"mean": float(corpora["train"].traces.mean(dtype=np.float64)),
                     "std": float(corpora["train"].traces.std(dtype=np.float64))}
    if normalization["std"] < 1e-6:
        raise ValueError("Degenerate training normalization.")
    loaders = {name: _loader(TraceDataset(corpora[name].traces, corpora[name].labels, **normalization),
                             batch_size=batch_size, shuffle=name == "train", workers=0,
                             seed=seed, pin_memory=False) for name in ("train", "selection")}
    model = NeuronClassifier1D()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3, min_lr=1e-5)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    best_eligible, best_selection = False, None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, target, predicted = _run_epoch(model, loaders["train"], nn.CrossEntropyLoss(label_smoothing=.03),
                                                  torch.device("cpu"), optimizer)
        grouped = None
        if protocol_version == 2:
            logits = batch_logits(corpora["selection"].traces, model, normalization, batch_size)
            grouped = selection_summary(logits, corpora["selection"])
            val_loss, selection_accuracy = grouped["nll"], grouped["accuracy"]
            selection_loss, eligible = grouped["score"], grouped["reference_eligible"]
        else:
            val_loss, val_target, val_predicted = _run_epoch(model, loaders["selection"], nn.CrossEntropyLoss(), torch.device("cpu"))
            selection_accuracy = float(np.mean(val_target == val_predicted))
            selection_loss, eligible = val_loss, True
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Training or selection loss became nonfinite.")
        scheduler.step(selection_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "selection_nll": val_loss,
               "train_accuracy": float(np.mean(target == predicted)),
               "selection_accuracy": selection_accuracy,
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        if grouped:
            row.update({f"selection_{key}": grouped[key] for key in ("score", "reference_accuracy", "minimum_condition_accuracy",
                                                                    "minimum_condition_class_recall", "worst_condition_nll", "worst_condition_class_nll")})
        history.append(row)
        progress(f"[TRAIN] {epoch}/{epochs}: train={row['train_accuracy']:.3%}, selection={row['selection_accuracy']:.3%}, NLL={val_loss:.5f}" +
                 (f", worst={grouped['minimum_condition_accuracy']:.3%}, score={selection_loss:.5f}" if grouped else ""))
        if (eligible and not best_eligible) or (eligible == best_eligible and selection_loss < best_loss - .0001):
            best_loss, best_state, best_epoch, stale = selection_loss, copy.deepcopy(model.state_dict()), epoch, 0
            best_eligible, best_selection = eligible, grouped
            # A real training checkpoint allows recovery after an interrupted run.
            torch.save({"epoch": epoch, "model_state": best_state, "normalization": normalization}, output / "best_training_state.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    calibration = fit_temperature(batch_logits(corpora["calibration"].traces, model.for_inference(), normalization, batch_size),
                                  corpora["calibration"].labels)
    profiles = list(EXPANDED_PROFILES) if protocol_version == 2 else ["constant", "pulse", "ramp", "sine"]
    protocol = {
        "version": protocol_version, "seed": seed, "train_per_class_per_condition": train_per_class,
        "validation_per_class_per_condition": validation_per_class,
        "splits": {name: corpus.conditions for name, corpus in corpora.items()},
        "split_rows": {name: len(corpus.labels) for name, corpus in corpora.items()},
        "quiet_rows_retained": True, "training_profiles": profiles,
        "withheld_profiles": reserved["withheld_profiles"], "reserved_comparison_sha256": plan_digest(reserved),
        "out_degree": out_degree, "batch_size": batch_size, "maximum_epochs": epochs,
        "best_epoch": best_epoch, "best_selection_nll": best_selection["nll"] if best_selection else best_loss,
        "selection_rule": "clean reference >=99%, then mean NLL + 0.5 worst-condition NLL + 0.25 worst-condition-class NLL + 10 reference deficit" if protocol_version == 2 else "minimum mean NLL",
        "best_selection": best_selection,
        "initialization": "from scratch", "label_smoothing": .03,
        "calibration": calibration,
        "source_sha256": source_hashes,
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
    }
    checkpoint = {
        "format_version": 2, "architecture": "NeuronClassifier1D", "model_state": best_state,
        "class_codes": list(CLASS_CODES), "class_names": list(CLASS_NAMES),
        "normalization": normalization, "simulation": config.to_dict(), "confidence_threshold": .7,
        "temperature": calibration["temperature"], "training_profiles": profiles,
        "training_protocol": protocol,
        "fit_trace_hashes": sorted(set.union(*hashes.values())),
    }
    if any(file_digest(Path(__file__).parent / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("Training source changed during fitting; refusing to record ambiguous provenance.")
    checkpoint_path = output / "candidate.pt"
    torch.save(checkpoint, checkpoint_path)
    protocol["checkpoint_sha256"] = file_digest(checkpoint_path)
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (output / "training.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return checkpoint_path


def score_predictions(labels, probabilities, active, threshold):
    valid = np.isfinite(probabilities).all(axis=1)
    predicted = np.full(len(labels), -1, dtype=int)
    predicted[valid] = probabilities[valid].argmax(axis=1)
    accepted = active & valid & (np.nan_to_num(probabilities, nan=-1).max(axis=1) >= threshold)
    result = prediction_metrics(labels, predicted)
    result["probabilities"] = probability_metrics(labels, probabilities)
    result["coverage"] = float(accepted.mean())
    result["accepted"] = prediction_metrics(labels[accepted], predicted[accepted])
    result["accepted_errors_per_requested_neuron"] = float(np.count_nonzero(accepted & (predicted != labels)) / len(labels))
    return result


def adoption_decision(results, policy=None):
    policy = dict(POLICY if policy is None else policy)
    names = tuple(dict.fromkeys(row["condition"]["name"] for row in results))
    if not names or "reference" not in names or not any(row["condition"]["axis"] == "pulse" for row in results):
        raise ValueError("Adoption requires reference and pulse comparisons.")
    aggregates = {}
    for name in names:
        rows = [row for row in results if row["condition"]["name"] == name]
        aggregates[name] = {model: {
            key: float(np.mean([row[model][key] for row in rows]))
            for key in ("accuracy", "macro_f1", "coverage", "accepted_errors_per_requested_neuron")
        } for model in ("baseline", "candidate")}
        for model in ("baseline", "candidate"):
            for key in ("nll", "brier", "ece"):
                aggregates[name][model][key] = float(np.mean([row[model]["probabilities"][key] for row in rows]))
    means = {model: {key: float(np.mean([row[model][key] for row in aggregates.values()]))
                     for key in ("accuracy", "macro_f1", "coverage", "accepted_errors_per_requested_neuron", "nll", "brier", "ece")}
             for model in ("baseline", "candidate")}
    pulse = [row for row in results if row["condition"]["axis"] == "pulse"]
    pulse_gain = float(np.mean([row["candidate"]["accuracy"] - row["baseline"]["accuracy"] for row in pulse]))
    seed_gains = {str(seed): float(np.mean([row["candidate"]["accuracy"] - row["baseline"]["accuracy"]
                                           for row in results if row["seed"] == seed])) for seed in sorted({row["seed"] for row in results})}
    reference = aggregates["reference"]
    eps = 1e-12
    checks = {
        "mean_accuracy_gain": means["candidate"]["accuracy"] - means["baseline"]["accuracy"] + eps >= policy["minimum_mean_accuracy_gain"],
        "pulse_accuracy_gain": pulse_gain + eps >= policy["minimum_pulse_accuracy_gain"],
        "reference_accuracy_floor": reference["candidate"]["accuracy"] + eps >= policy["minimum_reference_accuracy"],
        "reference_nonregression": reference["candidate"]["accuracy"] + eps >= reference["baseline"]["accuracy"] - policy["maximum_reference_accuracy_drop"],
        "no_large_condition_regression": all(row["candidate"]["accuracy"] + eps >= row["baseline"]["accuracy"] - policy["maximum_single_condition_accuracy_drop"] for row in aggregates.values()),
        "coverage_preserved": means["candidate"]["coverage"] + eps >= means["baseline"]["coverage"] - policy["maximum_mean_coverage_drop"],
        "accepted_error_reduction": means["candidate"]["accepted_errors_per_requested_neuron"] <= means["baseline"]["accepted_errors_per_requested_neuron"] * (1 - policy["minimum_relative_accepted_error_reduction"]) + eps,
        "nll_improved": not policy["require_nll_improvement"] or means["candidate"]["nll"] < means["baseline"]["nll"],
        "brier_nonregression": not policy["require_brier_nonregression"] or means["candidate"]["brier"] <= means["baseline"]["brier"] + eps,
        "positive_gain_each_seed": not policy["require_positive_gain_each_seed"] or all(gain > 0 for gain in seed_gains.values()),
    }
    return {"superior": all(checks.values()), "checks": checks, "policy": policy, "means": means,
            "pulse_accuracy_gain": pulse_gain, "seed_accuracy_gains": seed_gains, "conditions": aggregates}


def compare_candidates(baseline_path, candidate_path, config, output, *, samples_per_class=100, progress=print):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be new or empty; a completed test cannot be overwritten.")
    if samples_per_class < 1:
        raise ValueError("Comparison sample count must be positive.")
    baseline_model, baseline, baseline_config = load_frozen_checkpoint(baseline_path)
    candidate_model, candidate, candidate_config = load_frozen_checkpoint(candidate_path)
    if baseline_config != config or candidate_config != config:
        raise ValueError("Comparison simulation settings must match both checkpoints.")
    if not candidate.get("training_protocol"):
        raise ValueError("Candidate lacks its recorded training protocol.")
    version = candidate["training_protocol"]["version"]
    plan = comparison_plan(config, version)
    baseline_hash, candidate_hash = file_digest(baseline_path), file_digest(candidate_path)
    fit_hashes = set(candidate["fit_trace_hashes"])
    if set(candidate["training_protocol"]["withheld_profiles"]) != set(plan["withheld_profiles"]):
        raise ValueError("Candidate was not fitted under this comparison protocol.")
    if version == 2 and candidate["training_protocol"].get("reserved_comparison_sha256") != plan_digest(plan):
        raise ValueError("Reserved comparison protocol has changed since training.")
    fitted_profiles = {record["condition"]["profile"] for split in candidate["training_protocol"]["splits"].values() for record in split}
    if fitted_profiles & set(plan["withheld_profiles"]):
        raise ValueError("A reserved stimulus family was used during fitting.")
    degree = int(candidate["training_protocol"]["out_degree"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "status.json").write_text(json.dumps({"status": "running"}) + "\n")
    results = []
    try:
        for condition in comparison_conditions(config, version):
            for seed in plan["seeds"]:
                progress(f"[COMPARE] {condition.name}, seed {seed}")
                cohort = generate_cohort(condition, config, samples_per_class, seed,
                                         out_degree=degree, namespace=plan["namespace"])
                if fit_hashes & trace_digests(cohort.traces):
                    raise ValueError("Comparison overlaps candidate fitting or calibration data.")
                # Match the existing application's activity gate for both models.
                # This is its legacy count of >=29 mV samples, not clean spike events.
                active = np.isfinite(cohort.traces).all(axis=1) & (count_spikes(cohort.traces) >= config.minimum_spikes)
                row = {"condition": asdict(condition), "seed": seed, "synapses": cohort.synapses}
                for name, model, checkpoint in (("baseline", baseline_model, baseline), ("candidate", candidate_model, candidate)):
                    probabilities = frozen_probabilities(cohort.traces, model, checkpoint)
                    row[name] = score_predictions(cohort.labels, probabilities, active,
                                                  float(checkpoint.get("confidence_threshold", .7)))
                results.append(row)
        if file_digest(baseline_path) != baseline_hash or file_digest(candidate_path) != candidate_hash:
            raise RuntimeError("A checkpoint changed during comparison.")
        report = {"schema_version": 1, "protocol_version": version, "status": "complete", "baseline_sha256": baseline_hash,
                  "candidate_sha256": candidate_hash, "config": config.to_dict(),
                  "comparison_namespace": plan["namespace"], "seeds": plan["seeds"],
                  "reserved_comparison_sha256": plan_digest(plan),
                  "samples_per_class": samples_per_class, "results": results,
                  "decision": adoption_decision(results),
                  "scope": "Engineering gate on paired synthetic tests; not biological validation or a statistical significance claim.",
                  "coverage_policy": "finite trace, legacy GUI >=29 mV sample count, checkpoint confidence; prior current-range rejection bypassed for diagnostic comparisons",
                  "withheld_profiles": plan["withheld_profiles"]}
        (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["condition", "seed", "model", "correct", "total", "accuracy", "macro_f1", "coverage", "accepted_accuracy", "accepted_errors_per_requested_neuron", "nll", "brier", "ece"]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for row in results:
                for model in ("baseline", "candidate"):
                    score = row[model]
                    writer.writerow({"condition": row["condition"]["name"], "seed": row["seed"], "model": model,
                                     **{key: score[key] for key in ("correct", "total", "accuracy", "macro_f1", "coverage", "accepted_errors_per_requested_neuron")},
                                     "accepted_accuracy": score["accepted"]["accuracy"],
                                     **{key: score["probabilities"][key] for key in ("nll", "brier", "ece")}})
        lines = ["# Candidate comparison", "", f"Superior under every adoption gate: **{report['decision']['superior']}**", "",
                 "| Condition | Baseline accuracy | Candidate accuracy | Gain (pp) |", "|---|---:|---:|---:|"]
        for name, scores in report["decision"]["conditions"].items():
            lines.append(f"| {name} | {scores['baseline']['accuracy']:.3%} | {scores['candidate']['accuracy']:.3%} | {100 * (scores['candidate']['accuracy'] - scores['baseline']['accuracy']):+.3f} |")
        lines.extend(["", "## Adoption checks", ""] + [f"- {name}: {passed}" for name, passed in report["decision"]["checks"].items()])
        lines.extend(["", report["scope"], "", "A reference checkpoint remains active if any gate fails. No test-based training adjustment is performed by this command."])
        (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (output / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
        return report
    except BaseException as exc:
        (output / "status.json").write_text(json.dumps({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": str(exc)}) + "\n")
        raise


def adopt_candidate(baseline_path, candidate_path, report):
    """Promote only the compared bytes; keep a verified backup and replace atomically."""
    if report.get("status") != "complete" or not report.get("decision", {}).get("superior"):
        raise ValueError("Adoption rejected: candidate has not passed all comparison gates.")
    if baseline_path.resolve() == candidate_path.resolve():
        raise ValueError("Candidate and active checkpoint must be different files.")
    if file_digest(baseline_path) != report["baseline_sha256"] or file_digest(candidate_path) != report["candidate_sha256"]:
        raise ValueError("Checkpoint hashes differ from the completed comparison.")
    _, checkpoint, config = load_frozen_checkpoint(candidate_path)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Candidate does not contain the application's class names.")
    version = checkpoint["training_protocol"]["version"]
    plan = comparison_plan(config, version)
    expected_pairs = {(condition.name, seed) for condition in comparison_conditions(config, version) for seed in plan["seeds"]}
    actual_pairs = [(row["condition"]["name"], row["seed"]) for row in report.get("results", [])]
    if (report.get("comparison_namespace") != plan["namespace"] or set(actual_pairs) != expected_pairs
            or len(actual_pairs) != len(expected_pairs) or report["decision"].get("policy") != POLICY):
        raise ValueError("Adoption requires the complete predeclared comparison and unchanged policy.")
    if version == 2:
        if (report.get("protocol_version") != version or report.get("reserved_comparison_sha256") != plan_digest(plan)
                or checkpoint["training_protocol"].get("reserved_comparison_sha256") != plan_digest(plan)):
            raise ValueError("Adoption requires the comparison reserved before training.")
        expected_conditions = {condition["name"]: condition for condition in plan["conditions"]}
        if any(json.loads(json.dumps(row["condition"])) != expected_conditions[row["condition"]["name"]] for row in report["results"]):
            raise ValueError("Comparison conditions differ from the reserved protocol.")
    if not adoption_decision(report["results"])["superior"]:
        raise ValueError("Recomputed comparison does not pass the adoption gates.")
    backup = baseline_path.with_name(f"{baseline_path.stem}.original_{report['baseline_sha256'][:16]}{baseline_path.suffix}")
    if backup.exists():
        if file_digest(backup) != report["baseline_sha256"]:
            raise ValueError("Existing backup has different content; preserving it.")
    else:
        with baseline_path.open("rb") as source, backup.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if file_digest(backup) != report["baseline_sha256"]:
            raise RuntimeError("Backup verification failed.")
    handle, temporary_name = tempfile.mkstemp(prefix=".candidate_", suffix=".pt", dir=baseline_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as destination, candidate_path.open("rb") as source:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        if file_digest(temporary) != report["candidate_sha256"] or file_digest(baseline_path) != report["baseline_sha256"]:
            raise RuntimeError("Checkpoint changed before promotion.")
        os.replace(temporary, baseline_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"active": str(baseline_path.resolve()), "backup": str(backup.resolve()),
            "active_sha256": file_digest(baseline_path), "backup_sha256": file_digest(backup)}
