"""Aggregate per-sample evaluation metrics by run."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence


REQUIRED_METRICS_COLUMNS = {
    "sample_id",
    "run_id",
    "concept_id",
    "checkpoint_step",
    "rank",
    "data_size",
    "dino_subject_similarity",
    "clip_prompt_similarity",
    "valid",
    "invalid_reason",
}


AGGREGATE_COLUMNS = [
    "run_id",
    "concept_id",
    "checkpoint_step",
    "rank",
    "data_size",
    "expected_sample_count",
    "observed_sample_count",
    "missing_sample_count",
    "invalid_sample_count",
    "dino_mean",
    "dino_sd",
    "dino_sample_count",
    "dino_missing_count",
    "clip_mean",
    "clip_sd",
    "clip_sample_count",
    "clip_missing_count",
]


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _is_valid(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes"}


def _score(row: dict[str, object], column: str) -> float | None:
    raw = _clean(row.get(column))

    if raw == "":
        return None

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column} for sample {row.get('sample_id', '')}: {raw!r}"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {column} for sample {row.get('sample_id', '')}"
        )

    return value


def _one_value(
    rows: Sequence[dict[str, object]],
    column: str,
) -> str:
    values = {
        _clean(row.get(column))
        for row in rows
        if _clean(row.get(column))
    }

    if len(values) > 1:
        raise ValueError(
            f"Inconsistent {column} values within run: {sorted(values)}"
        )

    return next(iter(values), "")


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: Sequence[float]) -> float | None:
    # Descriptive population SD over the fixed prompt/seed evaluation grid.
    return statistics.pstdev(values) if values else None


def aggregate_rows(
    rows: Sequence[dict[str, object]],
    *,
    expected_sample_count: int = 32,
    run_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Aggregate DINO and CLIP scores separately for each run."""

    if expected_sample_count <= 0:
        raise ValueError("expected_sample_count must be positive")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        run_id = _clean(row.get("run_id"))

        if not run_id:
            raise ValueError("Per-sample row is missing run_id")

        grouped[run_id].append(row)

    if run_ids is None:
        selected_run_ids = sorted(grouped)
    else:
        selected_run_ids = list(dict.fromkeys(run_ids))

        missing_runs = [
            run_id for run_id in selected_run_ids
            if run_id not in grouped
        ]

        if missing_runs:
            raise ValueError(
                "Requested run IDs are missing from per-sample metrics: "
                + ", ".join(missing_runs)
            )

    output: list[dict[str, object]] = []

    for run_id in selected_run_ids:
        run_rows = grouped[run_id]

        if len(run_rows) > expected_sample_count:
            raise ValueError(
                f"{run_id} has {len(run_rows)} rows, "
                f"expected at most {expected_sample_count}"
            )

        sample_ids = [_clean(row.get("sample_id")) for row in run_rows]

        if any(not sample_id for sample_id in sample_ids):
            raise ValueError(f"{run_id} contains a row without sample_id")

        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{run_id} contains duplicate sample_id values")

        dino_scores = [
            value
            for row in run_rows
            if (value := _score(row, "dino_subject_similarity")) is not None
        ]

        clip_scores = [
            value
            for row in run_rows
            if (value := _score(row, "clip_prompt_similarity")) is not None
        ]

        observed_count = len(run_rows)

        output.append(
            {
                "run_id": run_id,
                "concept_id": _one_value(run_rows, "concept_id"),
                "checkpoint_step": _one_value(run_rows, "checkpoint_step"),
                "rank": _one_value(run_rows, "rank"),
                "data_size": _one_value(run_rows, "data_size"),
                "expected_sample_count": expected_sample_count,
                "observed_sample_count": observed_count,
                "missing_sample_count": expected_sample_count - observed_count,
                "invalid_sample_count": sum(
                    not _is_valid(row.get("valid")) for row in run_rows
                ),
                "dino_mean": _mean(dino_scores),
                "dino_sd": _sd(dino_scores),
                "dino_sample_count": len(dino_scores),
                "dino_missing_count": expected_sample_count - len(dino_scores),
                "clip_mean": _mean(clip_scores),
                "clip_sd": _sd(clip_scores),
                "clip_sample_count": len(clip_scores),
                "clip_missing_count": expected_sample_count - len(clip_scores),
            }
        )

    return output


def load_metrics_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Per-sample metrics CSV does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("Per-sample metrics CSV has no header")

        missing_columns = REQUIRED_METRICS_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Per-sample metrics CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        return list(reader)


def write_aggregate_csv(
    rows: Sequence[dict[str, object]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics_file(
    input_path: str | Path = "results/metrics_per_sample.csv",
    output_path: str | Path = "results/metrics_aggregate.csv",
    *,
    expected_sample_count: int = 32,
    run_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Build metrics_aggregate.csv from metrics_per_sample.csv."""

    rows = load_metrics_csv(input_path)
    aggregate = aggregate_rows(
        rows,
        expected_sample_count=expected_sample_count,
        run_ids=run_ids,
    )
    write_aggregate_csv(aggregate, output_path)

    return aggregate
