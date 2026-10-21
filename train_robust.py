"""Train, compare and optionally adopt a separate robust candidate checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from config import SimulationConfig
from inference import default_cpu_threads
from research_study import load_frozen_checkpoint
from robust_candidate import adopt_candidate, compare_candidates, train_candidate


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "weights/classifier.pt")
    parser.add_argument("--candidate", type=Path, help="Compare this already trained candidate instead of training again")
    parser.add_argument("--context-study", type=Path, help="Use recorded configuration for train-only when baseline is unavailable")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--adopt-if-superior", action="store_true")
    parser.add_argument("--protocol", type=int, choices=(1, 2), default=1,
                        help="Training protocol: 1 reproduces the earlier study; 2 adds randomized envelopes and grouped selection")
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--validation-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--out-degree", type=int, default=100)
    parser.add_argument("--threads", type=int, default=default_cpu_threads())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.train_per_class, args.validation_per_class, args.test_per_class, args.epochs,
           args.patience, args.batch_size, args.threads) < 1 or args.seed < 0:
        parser.error("Counts/threads must be positive and seed nonnegative.")
    if args.train_only and (args.candidate or args.adopt_if_superior):
        parser.error("--train-only cannot compare or adopt a candidate.")
    if not args.baseline.is_file() and not (args.train_only and args.context_study):
        parser.error("The actual trained --baseline is required for a valid comparison. Legacy .pth files and earlier metric summaries cannot replace it.")
    torch.set_num_threads(args.threads)
    output = args.output or ROOT / "artifacts/research" / datetime.now().strftime("candidate_%Y%m%d_%H%M%S_%f")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose a new or empty output directory; existing results are preserved.")
    try:
        if args.baseline.is_file():
            _, _, config = load_frozen_checkpoint(args.baseline)
        else:
            study = json.loads(args.context_study.read_text(encoding="utf-8"))
            config = SimulationConfig(**{key: value for key, value in study["config"].items()
                                         if key in SimulationConfig.__dataclass_fields__})
        output.mkdir(parents=True, exist_ok=True)
        (output / "status.json").write_text(json.dumps({"status": "running"}) + "\n")
        candidate = args.candidate or train_candidate(
            config, output / "training", train_per_class=args.train_per_class,
            validation_per_class=args.validation_per_class, epochs=args.epochs, patience=args.patience,
            seed=args.seed, out_degree=args.out_degree, batch_size=args.batch_size,
            protocol_version=args.protocol,
        )
        print(f"[CANDIDATE] {candidate.resolve()}", flush=True)
        if args.train_only:
            (output / "status.json").write_text(json.dumps({"status": "trained_awaiting_comparison", "candidate": str(candidate.resolve())}) + "\n")
            print("Trained candidate only. Actual baseline still required; no superiority or adoption claim.")
            return 0
        report = compare_candidates(args.baseline, candidate, config, output / "comparison", samples_per_class=args.test_per_class)
        for name, passed in report["decision"]["checks"].items():
            print(f"[GATE] {name}: {'PASS' if passed else 'FAIL'}")
        if args.adopt_if_superior and report["decision"]["superior"]:
            adoption = adopt_candidate(args.baseline, candidate, report)
            (output / "adoption.json").write_text(json.dumps(adoption, indent=2) + "\n")
            print(f"[ADOPTED] {adoption['active']}\n[BACKUP] {adoption['backup']}\nRestart the application to load the new model.")
        else:
            print("[UNCHANGED] Active checkpoint retained." + (" Candidate failed at least one gate." if not report["decision"]["superior"] else " Adoption was not requested."))
        (output / "status.json").write_text(json.dumps({"status": "complete", "superior": report["decision"]["superior"]}) + "\n")
        print(f"[REPORT] {(output / 'comparison/comparison.md').resolve()}")
        return 0
    except KeyboardInterrupt:
        (output / "status.json").write_text(json.dumps({"status": "interrupted"}) + "\n")
        print("Interrupted; incomplete results cannot authorize adoption.")
        return 130
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        if output.is_dir():
            (output / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
