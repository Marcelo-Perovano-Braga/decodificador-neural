"""Reproducible computational-cost comparison for LIF and Izhikevich."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import CLASS_CODES, IZHIKEVICH_PARAMETERS, SimulationConfig
from simulation import IzhikevichBatch, count_spikes, simulate_izhikevich_batch, simulate_lif_batch


PROJECT_ROOT = Path(__file__).resolve().parent


def _median_runtime(operation, repeats: int) -> float:
    operation()  # Warm-up outside the measured samples.
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def benchmark_models(
    *,
    samples: int = 512,
    repeats: int = 5,
    seed: int = 42,
    config: SimulationConfig | None = None,
) -> dict[str, object]:
    if samples < 3:
        raise ValueError("samples must be at least 3.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    config = config or SimulationConfig()
    rng = np.random.default_rng(seed)
    codes = [CLASS_CODES[index % len(CLASS_CODES)] for index in range(samples)]
    params = [IZHIKEVICH_PARAMETERS[code] for code in codes]
    izh_batch = IzhikevichBatch(
        a=np.asarray([item["a"] for item in params], dtype=np.float32),
        b=np.asarray([item["b"] for item in params], dtype=np.float32),
        c=np.asarray([item["c"] for item in params], dtype=np.float32),
        d=np.asarray([item["d"] for item in params], dtype=np.float32),
        current=rng.uniform(config.current_min, config.current_max, size=samples).astype(np.float32),
    )
    lif_drives = rng.uniform(20.5, 40.0, size=samples).astype(np.float32)

    def run_izhikevich() -> np.ndarray:
        return simulate_izhikevich_batch(
            izh_batch,
            time_ms=config.time_ms,
            dt_ms=config.dt_ms,
        )

    def run_lif() -> np.ndarray:
        return simulate_lif_batch(
            lif_drives,
            time_ms=config.time_ms,
            dt_ms=config.dt_ms,
        )

    izh_seconds = _median_runtime(run_izhikevich, repeats)
    lif_seconds = _median_runtime(run_lif, repeats)
    izh_traces = run_izhikevich()
    lif_traces = run_lif()
    lif_spikes = np.count_nonzero(lif_traces >= -50.0, axis=1)

    return {
        "scope_note": (
            "This benchmark compares simulation cost only. The models use different "
            "state equations and input scales, so spike totals are descriptive rather "
            "than a biological equivalence test."
        ),
        "samples": samples,
        "time_steps": config.window_size,
        "repeats": repeats,
        "lif": {
            "median_seconds": lif_seconds,
            "microseconds_per_trace": lif_seconds / samples * 1_000_000,
            "mean_spikes_per_trace": float(np.mean(lif_spikes)),
            "trace_memory_bytes": int(lif_traces.nbytes),
        },
        "izhikevich": {
            "median_seconds": izh_seconds,
            "microseconds_per_trace": izh_seconds / samples * 1_000_000,
            "mean_spikes_per_trace": float(np.mean(count_spikes(izh_traces))),
            "trace_memory_bytes": int(izh_traces.nbytes),
        },
        "izhikevich_to_lif_runtime_ratio": izh_seconds / lif_seconds if lif_seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "model_benchmark.json",
    )
    args = parser.parse_args()

    report = benchmark_models(samples=args.samples, repeats=args.repeats, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
