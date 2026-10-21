"""Validate the SQLite dataset and emit an auditable summary."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from data_pipeline import read_metadata


PROJECT_ROOT = Path(__file__).resolve().parent


def validate_dataset(path: Path) -> dict[str, object]:
    metadata = read_metadata(path)
    expected_trace_bytes = int(metadata["simulation"]["window_size"]) * 4
    expected_per_class = int(metadata["samples_per_class"])

    with sqlite3.connect(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        total = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        splits = dict(connection.execute(
            "SELECT split, COUNT(*) FROM samples GROUP BY split ORDER BY split"
        ).fetchall())
        classes = dict(connection.execute(
            "SELECT class_code, COUNT(*) FROM samples GROUP BY class_code ORDER BY label"
        ).fetchall())
        min_trace_bytes, max_trace_bytes = connection.execute(
            "SELECT MIN(length(trace)), MAX(length(trace)) FROM samples"
        ).fetchone()
        min_spikes, max_spikes = connection.execute(
            "SELECT MIN(spike_count), MAX(spike_count) FROM samples"
        ).fetchone()
        min_current, max_current = connection.execute(
            "SELECT MIN(input_current), MAX(input_current) FROM samples"
        ).fetchone()

    checks = {
        "sqlite_integrity": integrity == "ok",
        "expected_total": total == expected_per_class * len(metadata["class_codes"]),
        "balanced_classes": all(count == expected_per_class for count in classes.values()),
        "trace_size": min_trace_bytes == max_trace_bytes == expected_trace_bytes,
        "minimum_spikes": min_spikes >= int(metadata["simulation"]["minimum_spikes"]),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "total_samples": total,
        "splits": splits,
        "classes": classes,
        "trace_bytes": {"minimum": min_trace_bytes, "maximum": max_trace_bytes},
        "spike_count": {"minimum": min_spikes, "maximum": max_spikes},
        "input_current": {"minimum": min_current, "maximum": max_current},
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "neural_patterns.sqlite",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "dataset_summary.json",
    )
    args = parser.parse_args()
    report = validate_dataset(args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

