"""Aggregate per-sample evaluation metrics by run and generation mode."""

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
    "generation_mode",
    "concept_id",
    "prompt_id",
    "generation_seed",
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
    "generation_mode",
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
    "dino_invalid_count",
    "clip_mean",
    "clip_sd",
    "clip_sample_count",
    "clip_missing_count",
    "clip_invalid_count",
]

EXPECTED_PROMPT_IDS = tuple(f"p{index:02d}" for index in range(1, 9))
EXPECTED_GENERATION_SEEDS = (11, 22, 33, 44)
PROTOCOL_SAMPLE_IDENTITIES = frozenset(
    (prompt_id, seed)
    for prompt_id in EXPECTED_PROMPT_IDS
    for seed in EXPECTED_GENERATION_SEEDS
)


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


def _seed(row: dict[str, object]) -> int:
    raw = _clean(row.get("generation_seed"))
    try:
        seed = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid generation_seed for sample {row.get('sample_id', '')}: {raw!r}"
        ) from exc
    return seed


def _metric_result(
    row: dict[str, object],
    *,
    column: str,
    metric_prefix: str,
) -> tuple[float | None, bool]:
    """Return (usable score, metric-specific invalid flag).

    The scorer CSV has a combined ``valid`` flag. A finite score may still be
    used when only the *other* metric failed, as identified in invalid_reason.
    Unattributed failures invalidate both metrics rather than trusting stale
    values.
    """
    score = _score(row, column)
    valid = _is_valid(row.get("valid"))
    reasons = [
        reason.strip()
        for reason in _clean(row.get("invalid_reason")).split(";")
        if reason.strip()
    ]
    reason_prefixes = [reason.split(":", 1)[0].strip().upper() for reason in reasons]
    known_metric_reasons = bool(reason_prefixes) and all(
        prefix in {"DINO", "CLIP"} for prefix in reason_prefixes
    )

    if valid:
        return score, False

    if metric_prefix in reason_prefixes:
        return None, True

    # A failed other metric does not invalidate this score. A finite score is
    # required; an empty value is a missing score, not a valid zero.
    if known_metric_reasons:
        if score is not None:
            return score, False
        return None, False

    # The combined flag is false but the cause cannot be attributed safely.
    return None, True


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
    """Aggregate DINO and CLIP separately for every run and generation mode.

    With the protocol's 32-sample evaluation grid, validate the exact eight
    prompt IDs crossed with seeds 11, 22, 33, and 44. For smaller custom test
    grids, validate observed count and uniqueness without assuming that matrix.
    """

    if expected_sample_count <= 0:
        raise ValueError("expected_sample_count must be positive")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        run_id = _clean(row.get("run_id"))
        generation_mode = _clean(row.get("generation_mode"))

        if not run_id:
            raise ValueError("Per-sample row is missing run_id")
        if not generation_mode:
            raise ValueError(
                f"{run_id} contains a row without generation_mode; "
                "base/adapter identity cannot be inferred"
            )

        grouped[(run_id, generation_mode)].append(row)

    if run_ids is None:
        selected_run_ids = sorted({run_id for run_id, _ in grouped})
    else:
        selected_run_ids = list(dict.fromkeys(run_ids))

        available_run_ids = {group_run_id for group_run_id, _ in grouped}
        missing_runs = [run_id for run_id in selected_run_ids if run_id not in available_run_ids]

        if missing_runs:
            raise ValueError(
                "Requested run IDs are missing from per-sample metrics: "
                + ", ".join(missing_runs)
            )

    output: list[dict[str, object]] = []

    selected_groups = sorted(
        group_key
        for group_key in grouped
        if group_key[0] in selected_run_ids
    )

    for run_id, generation_mode in selected_groups:
        run_rows = grouped[(run_id, generation_mode)]

        if len(run_rows) > expected_sample_count:
            raise ValueError(
                f"{run_id}/{generation_mode} has {len(run_rows)} rows, "
                f"expected at most {expected_sample_count}"
            )

        sample_ids = [_clean(row.get("sample_id")) for row in run_rows]

        if any(not sample_id for sample_id in sample_ids):
            raise ValueError(f"{run_id} contains a row without sample_id")

        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{run_id} contains duplicate sample_id values")

        sample_identities: list[tuple[str, int]] = []
        for row in run_rows:
            prompt_id = _clean(row.get("prompt_id"))
            if not prompt_id:
                raise ValueError(f"{run_id}/{generation_mode} contains a row without prompt_id")
            sample_identities.append((prompt_id, _seed(row)))

        if len(sample_identities) != len(set(sample_identities)):
            raise ValueError(
                f"{run_id}/{generation_mode} contains duplicate prompt_id/generation_seed identities"
            )

        observed_identities = set(sample_identities)
        if expected_sample_count == len(PROTOCOL_SAMPLE_IDENTITIES):
            unexpected = observed_identities - PROTOCOL_SAMPLE_IDENTITIES
            if unexpected:
                raise ValueError(
                    f"{run_id}/{generation_mode} contains unexpected prompt/seed identities: "
                    f"{sorted(unexpected)}"
                )
            missing_sample_count = len(PROTOCOL_SAMPLE_IDENTITIES - observed_identities)
        else:
            missing_sample_count = expected_sample_count - len(run_rows)

        dino_results = [
            _metric_result(
                row,
                column="dino_subject_similarity",
                metric_prefix="DINO",
            )
            for row in run_rows
        ]
        clip_results = [
            _metric_result(
                row,
                column="clip_prompt_similarity",
                metric_prefix="CLIP",
            )
            for row in run_rows
        ]
        dino_scores = [score for score, invalid in dino_results if score is not None and not invalid]
        clip_scores = [score for score, invalid in clip_results if score is not None and not invalid]

        observed_count = len(run_rows)

        output.append(
            {
                "run_id": run_id,
                "generation_mode": generation_mode,
                "concept_id": _one_value(run_rows, "concept_id"),
                "checkpoint_step": _one_value(run_rows, "checkpoint_step"),
                "rank": _one_value(run_rows, "rank"),
                "data_size": _one_value(run_rows, "data_size"),
                "expected_sample_count": expected_sample_count,
                "observed_sample_count": observed_count,
                "missing_sample_count": missing_sample_count,
                "invalid_sample_count": sum(
                    not _is_valid(row.get("valid")) for row in run_rows
                ),
                "dino_mean": _mean(dino_scores),
                "dino_sd": _sd(dino_scores),
                "dino_sample_count": len(dino_scores),
                "dino_missing_count": expected_sample_count - len(dino_scores),
                "dino_invalid_count": sum(invalid for _, invalid in dino_results),
                "clip_mean": _mean(clip_scores),
                "clip_sd": _sd(clip_scores),
                "clip_sample_count": len(clip_scores),
                "clip_missing_count": expected_sample_count - len(clip_scores),
                "clip_invalid_count": sum(invalid for _, invalid in clip_results),
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
