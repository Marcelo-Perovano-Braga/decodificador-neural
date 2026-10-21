"""Evaluate frozen CNN robustness, withheld conditions and timing baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from inference import default_cpu_threads
from network import NetworkConfig
from research_protocol import make_protocol
from research_study import load_frozen_checkpoint, run_study


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights" / "classifier.pt")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "neural_patterns.sqlite")
    parser.add_argument("--baseline-source", choices=("database", "synthetic"), default="database")
    parser.add_argument("--synthetic-fit-per-class", type=int, default=500)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--out-degree", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=default_cpu_threads())
    parser.add_argument("--burst-isi-ms", type=float, default=5.0)
    parser.add_argument("--conditions", nargs="+", help="Explicit named subset; omitted means the complete protocol")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1 or not 1 <= args.samples_per_class <= 10000 or not 1 <= args.synthetic_fit_per_class <= 10000:
        parser.error("Threads/batch size must be positive; samples per class must be 1–10,000.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use distinct nonnegative evaluation seeds.")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint missing. Supply your existing trained --checkpoint; this tool never trains or substitutes the CNN.")
    torch.set_num_threads(args.threads)
    try:
        _, _, config = load_frozen_checkpoint(args.checkpoint)
        conditions = make_protocol(config)
        if args.list_conditions:
            for condition in conditions:
                print(f"{condition.name}: {condition.axis}" + (f" · HELD OUT: {condition.held_out_reason}" if condition.held_out_reason else ""))
            return 0
        if args.conditions:
            by_name = {condition.name: condition for condition in conditions}
            if len(set(args.conditions)) != len(args.conditions) or any(name not in by_name for name in args.conditions):
                parser.error("Unknown or repeated condition. Use --list-conditions.")
            conditions = [by_name[name] for name in args.conditions]
        network_config = NetworkConfig(out_degree=args.out_degree)
        if any(condition.network_weight is not None for condition in conditions):
            network_config.degree_for(3 * args.samples_per_class)
        database = args.database if args.baseline_source == "database" else None
        if database is not None and not database.is_file():
            parser.error("Original training database missing. Supply --database, or explicitly choose --baseline-source synthetic for a new exploratory baseline cohort.")
        output = args.output or ROOT / "artifacts" / "research" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        report = run_study(
            checkpoint_path=args.checkpoint, database=database, output=output,
            conditions=conditions, seeds=args.seeds, samples_per_class=args.samples_per_class,
            synthetic_samples=args.synthetic_fit_per_class, out_degree=args.out_degree,
            batch_size=args.batch_size, burst_isi_ms=args.burst_isi_ms,
        )
    except KeyboardInterrupt:
        print("Study interrupted; partial output is marked in status.json.")
        return 130
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Completed {len(report['results'])} condition/seed runs. Checkpoint unchanged.")
    print(f"Open: {(output / 'report.html').resolve()}")
    print(f"Exact metrics: {(output / 'study.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
