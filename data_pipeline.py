"""Generate and persist a reproducible SQLite dataset of neural traces."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from config import CLASS_CODES, CLASS_NAMES, IZHIKEVICH_PARAMETERS, SimulationConfig
from simulation import IzhikevichBatch, add_observation_noise, count_spikes, simulate_izhikevich_batch


SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


def _split_counts(samples_per_class: int) -> dict[str, int]:
    if samples_per_class < 30:
        raise ValueError("samples_per_class must be at least 30 for reliable splits.")
    train = int(round(samples_per_class * 0.70))
    validation = int(round(samples_per_class * 0.15))
    test = samples_per_class - train - validation
    return {"train": train, "validation": validation, "test": test}


def _jittered_parameters(
    code: str,
    size: int,
    jitter: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = IZHIKEVICH_PARAMETERS[code]

    def sample(name: str) -> np.ndarray:
        center = base[name]
        if jitter == 0:
            return np.full(size, center, dtype=np.float32)
        multiplier = rng.normal(1.0, jitter, size=size)
        return (center * multiplier).astype(np.float32)

    return sample("a"), sample("b"), sample("c"), sample("d")


def _candidate_batches(
    code: str,
    remaining: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    chunk_size: int,
) -> Iterator[tuple[IzhikevichBatch, np.ndarray]]:
    """Yield classifiable candidates until the caller has enough samples."""

    attempts_without_progress = 0
    while remaining > 0:
        candidate_count = min(max(remaining * 2, 32), chunk_size)
        a, b, c, d = _jittered_parameters(
            code,
            candidate_count,
            config.parameter_jitter,
            rng,
        )
        current = rng.uniform(
            config.current_min,
            config.current_max,
            size=candidate_count,
        ).astype(np.float32)
        batch = IzhikevichBatch(a=a, b=b, c=c, d=d, current=current)
        traces = simulate_izhikevich_batch(
            batch,
            time_ms=config.time_ms,
            dt_ms=config.dt_ms,
        )
        eligible = count_spikes(traces) >= config.minimum_spikes
        if not np.any(eligible):
            attempts_without_progress += 1
            if attempts_without_progress >= 20:
                raise RuntimeError(
                    f"Could not generate active traces for class {code}; "
                    "adjust the current range or minimum_spikes."
                )
            continue

        attempts_without_progress = 0
        indices = np.flatnonzero(eligible)[:remaining]
        selected_batch = IzhikevichBatch(
            a=batch.a[indices],
            b=batch.b[indices],
            c=batch.c[indices],
            d=batch.d[indices],
            current=batch.current[indices],
        )
        selected_traces = traces[indices]
        yield selected_batch, selected_traces
        remaining -= indices.size


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL CHECK(split IN ('train', 'validation', 'test')),
            label INTEGER NOT NULL,
            class_code TEXT NOT NULL,
            class_name TEXT NOT NULL,
            input_current REAL NOT NULL,
            a REAL NOT NULL,
            b REAL NOT NULL,
            c REAL NOT NULL,
            d REAL NOT NULL,
            spike_count INTEGER NOT NULL,
            trace BLOB NOT NULL
        );

        CREATE INDEX idx_samples_split ON samples(split);
        CREATE INDEX idx_samples_split_label ON samples(split, label);
        """
    )


def generate_dataset(
    database_path: str | os.PathLike[str],
    *,
    samples_per_class: int = 1000,
    seed: int = 42,
    config: SimulationConfig | None = None,
    chunk_size: int = 512,
    overwrite: bool = False,
) -> Path:
    """Create an atomic SQLite dataset with stratified train/validation/test rows."""

    config = config or SimulationConfig()
    destination = Path(database_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    rng = np.random.default_rng(seed)
    counts = _split_counts(samples_per_class)

    try:
        connection = sqlite3.connect(temporary)
        try:
            _initialize_database(connection)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "samples_per_class": samples_per_class,
                "class_codes": CLASS_CODES,
                "class_names": CLASS_NAMES,
                "split_counts_per_class": counts,
                "simulation": config.to_dict(),
                "trace_dtype": "float32",
                "description": (
                    "Independent single-neuron Izhikevich traces. Rows do not "
                    "represent a connected network or biological measurements."
                ),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
            )

            insert_sql = """
                INSERT INTO samples(
                    split, label, class_code, class_name, input_current,
                    a, b, c, d, spike_count, trace
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            for label, (code, name) in enumerate(zip(CLASS_CODES, CLASS_NAMES, strict=True)):
                for split in SPLIT_NAMES:
                    needed = counts[split]
                    for batch, clean_traces in _candidate_batches(
                        code,
                        needed,
                        config,
                        rng,
                        chunk_size,
                    ):
                        noisy_traces = add_observation_noise(
                            clean_traces,
                            config.observation_noise_std,
                            rng,
                        )
                        spikes = count_spikes(clean_traces)
                        rows = []
                        for index, trace in enumerate(noisy_traces):
                            rows.append((
                                split,
                                label,
                                code,
                                name,
                                float(batch.current[index]),
                                float(batch.a[index]),
                                float(batch.b[index]),
                                float(batch.c[index]),
                                float(batch.d[index]),
                                int(spikes[index]),
                                sqlite3.Binary(np.asarray(trace, dtype=np.float32).tobytes()),
                            ))
                        connection.executemany(insert_sql, rows)
                        connection.commit()
        finally:
            connection.close()

        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return destination


def read_metadata(database_path: str | os.PathLike[str]) -> dict[str, object]:
    connection = sqlite3.connect(Path(database_path))
    try:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    finally:
        connection.close()
    return {key: json.loads(value) for key, value in rows}


def load_split(
    database_path: str | os.PathLike[str],
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load one split into memory for efficient mini-batch training."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    metadata = read_metadata(database_path)
    window_size = int(metadata["simulation"]["window_size"])

    connection = sqlite3.connect(Path(database_path))
    try:
        rows = connection.execute(
            """
            SELECT label, input_current, a, b, c, d, spike_count, trace
            FROM samples
            WHERE split = ?
            ORDER BY id
            """,
            (split,),
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise RuntimeError(f"Dataset split is empty: {split}")

    traces = np.stack([
        np.frombuffer(row[7], dtype=np.float32, count=window_size).copy()
        for row in rows
    ])
    labels = np.asarray([row[0] for row in rows], dtype=np.int64)
    sample_metadata = {
        "input_current": np.asarray([row[1] for row in rows], dtype=np.float32),
        "a": np.asarray([row[2] for row in rows], dtype=np.float32),
        "b": np.asarray([row[3] for row in rows], dtype=np.float32),
        "c": np.asarray([row[4] for row in rows], dtype=np.float32),
        "d": np.asarray([row[5] for row in rows], dtype=np.float32),
        "spike_count": np.asarray([row[6] for row in rows], dtype=np.int32),
    }
    return traces, labels, sample_metadata
