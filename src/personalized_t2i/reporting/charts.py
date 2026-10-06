"""Generate RQ1/RQ2 charts from aggregated evaluation metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "run_id",
    "concept_id",
    "rank",
    "data_size",
    "dino_mean",
    "dino_sd",
    "clip_mean",
    "clip_sd",
}


def load_aggregate_metrics(
    path: str | Path = "results/metrics_aggregate.csv",
) -> list[dict[str, str]]:
    """Load and validate aggregated evaluation metrics."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Aggregate metrics CSV does not exist: {path}"
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("Aggregate metrics CSV has no header")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Aggregate metrics CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        return list(reader)


def _float(row: dict[str, str], column: str) -> float | None:
    """Convert a numeric field, treating empty values as missing."""

    value = row.get(column, "").strip()

    if value == "":
        return None

    return float(value)


def _source_text(rows: list[dict[str, str]]) -> str:
    """Build a compact source/run identifier for a figure."""

    run_ids = sorted(
        {
            row["run_id"].strip()
            for row in rows
            if row.get("run_id", "").strip()
        }
    )

    if not run_ids:
        return "Source: results/metrics_aggregate.csv"

    return (
        "Source: results/metrics_aggregate.csv | "
        f"Runs: {', '.join(run_ids)}"
    )


def _plot_metric(
    rows: list[dict[str, str]],
    *,
    x_column: str,
    x_label: str,
    metric_column: str,
    metric_label: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Plot one metric against one experiment variable."""

    grouped: dict[str, list[tuple[float, float]]] = {}

    for row in rows:
        concept = row["concept_id"].strip()

        x_value = _float(row, x_column)
        y_value = _float(row, metric_column)

        if x_value is None or y_value is None:
            continue

        grouped.setdefault(concept, []).append((x_value, y_value))

    if not grouped:
        raise ValueError(
            f"No valid data available for {metric_column}"
        )

    figure, axis = plt.subplots(figsize=(8, 5))

    for concept in sorted(grouped):
        points = sorted(grouped[concept])

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        axis.plot(
            x_values,
            y_values,
            marker="o",
            label=concept,
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel(metric_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Concept")

    figure.text(
        0.5,
        0.01,
        _source_text(rows),
        ha="center",
        va="bottom",
        fontsize=7,
    )

    figure.tight_layout(rect=(0, 0.03, 1, 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_rq1(
    rows: list[dict[str, str]],
    *,
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """Generate RQ1 charts: effect of training data size."""

    rq1_rows = [
        row
        for row in rows
        if _float(row, "rank") == 16
    ]

    if not rq1_rows:
        raise ValueError("No RQ1 rows found for rank=16.")

    output_dir = Path(output_dir)

    dino_path = output_dir / "rq1_dino.png"
    clip_path = output_dir / "rq1_clip.png"

    _plot_metric(
        rq1_rows,
        x_column="data_size",
        x_label="Training data size",
        metric_column="dino_mean",
        metric_label="DINO subject similarity",
        title="RQ1: Effect of Training Data Size on DINO Similarity",
        output_path=dino_path,
    )

    _plot_metric(
        rq1_rows,
        x_column="data_size",
        x_label="Training data size",
        metric_column="clip_mean",
        metric_label="CLIP prompt similarity",
        title="RQ1: Effect of Training Data Size on CLIP Similarity",
        output_path=clip_path,
    )

    return [dino_path, clip_path]


def plot_rq2(
    rows: list[dict[str, str]],
    *,
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """Generate RQ2 charts: effect of LoRA rank."""

    rq2_rows = [
        row
        for row in rows
        if _float(row, "data_size") == 5
    ]

    if not rq2_rows:
        raise ValueError("No RQ2 rows found for data_size=5.")

    output_dir = Path(output_dir)

    dino_path = output_dir / "rq2_dino.png"
    clip_path = output_dir / "rq2_clip.png"

    _plot_metric(
        rq2_rows,
        x_column="rank",
        x_label="LoRA rank",
        metric_column="dino_mean",
        metric_label="DINO subject similarity",
        title="RQ2: Effect of LoRA Rank on DINO Similarity",
        output_path=dino_path,
    )

    _plot_metric(
        rq2_rows,
        x_column="rank",
        x_label="LoRA rank",
        metric_column="clip_mean",
        metric_label="CLIP prompt similarity",
        title="RQ2: Effect of LoRA Rank on CLIP Similarity",
        output_path=clip_path,
    )

    return [dino_path, clip_path]


def generate_all_charts(
    metrics_path: str | Path = "results/metrics_aggregate.csv",
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """Generate all RQ1 and RQ2 charts."""

    rows = load_aggregate_metrics(metrics_path)

    return [
        *plot_rq1(rows, output_dir=output_dir),
        *plot_rq2(rows, output_dir=output_dir),
    ]
