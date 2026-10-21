"""Measure the application's complete population processor on the local CPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from config import CLASS_CODES, SimulationConfig
from inference import default_cpu_threads
from models import NeuronClassifier1D
from population import MAX_NEURONS_PER_CLASS, PopulationRequest
from population_worker import process_population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, nargs="+", default=[3000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=default_cpu_threads())
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "weights" / "classifier.pt")
    parser.add_argument("--simulation-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("--threads and --repeats must be positive.")
    if any(not 1 <= n <= MAX_NEURONS_PER_CLASS * len(CLASS_CODES) for n in args.neurons):
        parser.error(f"--neurons must be between 1 and {MAX_NEURONS_PER_CLASS * len(CLASS_CODES)}.")

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    checkpoint = model = None
    config = SimulationConfig()
    if not args.simulation_only:
        if not args.checkpoint.is_file():
            parser.error("Checkpoint missing. Supply --checkpoint or use --simulation-only.")
        try:
            try:
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(args.checkpoint, map_location=device)
            if (
                len(checkpoint["class_codes"]) != len(CLASS_CODES)
                or set(checkpoint["class_codes"]) != set(CLASS_CODES)
                or len(checkpoint["class_names"]) != len(CLASS_CODES)
            ):
                raise ValueError("Checkpoint must contain the RS, CH and FS classes.")
            model = NeuronClassifier1D(num_classes=len(CLASS_CODES)).to(device).eval()
            model.load_state_dict(checkpoint["model_state"])
            model = model.for_inference()
            config = SimulationConfig(**{
                name: value for name, value in checkpoint["simulation"].items()
                if name in SimulationConfig.__dataclass_fields__
            })
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            parser.error(f"Cannot load checkpoint: {exc}")

    def run(total: int):
        counts = (total // 3, total // 3, total - 2 * (total // 3))
        return process_population(
            PopulationRequest(counts, current=config.current_max, seed=42),
            lambda: False, config=config, model=model,
            checkpoint=checkpoint, device=device,
        )

    # Warm the simulator and CPU convolution backend outside measured runs.
    run(64)
    measurements = []
    for total in args.neurons:
        timings = []
        for _ in range(args.repeats):
            started = perf_counter()
            result = run(total)
            timings.append(perf_counter() - started)
            classified = result.statistics.classified if model is not None else None
            trace_bytes = result.population.traces.nbytes
            del result
        measurements.append({
            "neurons": total,
            "seconds": timings,
            "median_seconds": median(timings),
            "trace_bytes": trace_bytes,
            "classified_last_run": classified,
        })
    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_version": torch.__version__,
        "inference_threads": args.threads,
        "mode": "simulation_only" if model is None else "checkpoint_inference",
        "simulation": config.to_dict(),
        "scope": (
            "Simulation, observation noise, inference when enabled, statistics and raster preparation; "
            "excludes GUI rendering, checkpoint loading, warm-up and retaining an earlier result. "
            "This measures processing time, not scientific accuracy or a real-time guarantee."
        ),
        "measurements": measurements,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
