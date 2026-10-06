"""Generate RQ1/RQ2 qualitative comparison grids."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


RQ1_DATA_SIZES = {1, 3, 5, 10}
RQ2_RANKS = {4, 16, 32}


def load_generation_metadata(
    artifacts_dir: str | Path = "artifacts",
) -> list[dict]:
    """Load generation metadata from all run directories."""

    artifacts_dir = Path(artifacts_dir)

    if not artifacts_dir.exists():
        return []

    records = []

    for metadata_path in sorted(
        artifacts_dir.glob("*/generated/metadata.jsonl")
    ):
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                record["_metadata_path"] = str(metadata_path)
                records.append(record)

    return records


def _image_path(record: dict) -> Path:
    """Resolve an image path from a generation metadata record."""

    raw_path = str(record["image_path"])
    path = Path(raw_path)

    if path.exists():
        return path

    normalized = Path(raw_path.replace("\\", "/"))

    if normalized.exists():
        return normalized

    repo_root_candidate = Path.cwd() / normalized

    if repo_root_candidate.exists():
        return repo_root_candidate

    metadata_path = Path(record["_metadata_path"])
    artifact_root = metadata_path.parent.parent.parent
    artifact_candidate = artifact_root / normalized

    if artifact_candidate.exists():
        return artifact_candidate

    raise FileNotFoundError(
        f"Could not resolve image path: {record['image_path']}"
    )


def _index_records(records: list[dict]) -> dict[tuple, dict]:
    """Index records by run, prompt, and seed."""

    index = {}

    for record in records:
        key = (
            record["run_id"],
            record["prompt_id"],
            int(record["seed"]),
        )

        if key in index:
            raise ValueError(
                f"Duplicate generation record for {key}"
            )

        index[key] = record

    return index


def _run_metadata(records: list[dict]) -> dict[str, dict]:
    """Extract concept metadata for each run."""

    metadata = {}

    for record in records:
        run_id = record["run_id"]

        if run_id in metadata:
            existing_concept = metadata[run_id]["concept_id"]
            current_concept = record.get("concept_id")

            if existing_concept != current_concept:
                raise ValueError(
                    f"Run {run_id} has inconsistent concept_id values."
                )

            continue

        metadata[run_id] = {
            "run_id": run_id,
            "concept_id": record.get("concept_id"),
        }

    return metadata


def _run_data_size(run_id: str) -> int | None:
    """Extract n from a run ID such as toy01_n5_r16_ts42."""

    marker = "_n"

    if marker not in run_id:
        return None

    value = run_id.split(marker, 1)[1].split("_", 1)[0]

    try:
        return int(value)
    except ValueError:
        return None


def _run_rank(run_id: str) -> int | None:
    """Extract rank from a run ID such as toy01_n5_r16_ts42."""

    marker = "_r"

    if marker not in run_id:
        return None

    value = run_id.split(marker, 1)[1].split("_", 1)[0]

    try:
        return int(value)
    except ValueError:
        return None


def _complete_prompt_seed_pairs(
    index: dict[tuple, dict],
    run_ids: list[str],
    concept_id: str,
) -> list[tuple[str, int]]:
    """Return only prompt/seed pairs shared by every selected run."""

    available = set()

    for (
        run_id,
        prompt_id,
        seed,
    ) in index:
        if run_id in run_ids:
            record = index[(run_id, prompt_id, seed)]

            if record.get("concept_id") == concept_id:
                available.add((prompt_id, seed))

    return sorted(
        pair
        for pair in available
        if all(
            (run_id, pair[0], pair[1]) in index
            for run_id in run_ids
        )
    )


def _plot_grid(
    records: list[dict],
    run_ids: list[str],
    row_pairs: list[tuple[str, int]],
    column_labels: list[str],
    title: str,
    output_path: str | Path,
) -> Path:
    """Create a same-prompt/same-seed qualitative comparison grid."""

    index = _index_records(records)

    if not row_pairs:
        raise ValueError("No complete prompt/seed pairs available.")

    if not run_ids:
        raise ValueError("No runs available for grid.")

    rows = len(row_pairs)
    cols = len(run_ids)

    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 3.2 * rows),
        squeeze=False,
    )

    for col, label in enumerate(column_labels):
        axes[0][col].set_title(label, fontsize=11)

    for row, (prompt_id, seed) in enumerate(row_pairs):
        for col, run_id in enumerate(run_ids):
            axis = axes[row][col]
            key = (run_id, prompt_id, seed)

            record = index[key]

            try:
                image = Image.open(
                    _image_path(record)
                ).convert("RGB")

                axis.imshow(image)

            except (FileNotFoundError, OSError):
                axis.text(
                    0.5,
                    0.5,
                    "Image unavailable",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            axis.axis("off")

        axes[row][0].set_ylabel(
            f"{prompt_id}\nseed={seed}",
            fontsize=9,
            rotation=0,
            labelpad=45,
            va="center",
        )

    figure.suptitle(title, fontsize=14)

    source_runs = ", ".join(run_ids)

    figure.text(
        0.5,
        0.01,
        f"Source: artifacts/<run_id>/generated/metadata.jsonl | Runs: {source_runs}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    figure.tight_layout(rect=(0, 0.03, 1, 0.98))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def generate_rq1_grids(
    records: list[dict],
    output_dir: str | Path = "results/qualitative_grids",
) -> list[Path]:
    """
    Generate RQ1 qualitative grids.

    Requires all four configurations:
    n=1, 3, 5, 10 with rank=16.
    """

    output_dir = Path(output_dir)
    index = _index_records(records)
    run_metadata = _run_metadata(records)

    generated = []

    concepts = sorted(
        {
            metadata["concept_id"]
            for metadata in run_metadata.values()
            if metadata["concept_id"]
        }
    )

    for concept_id in concepts:
        selected = []

        for run_id, metadata in run_metadata.items():
            if metadata["concept_id"] != concept_id:
                continue

            if _run_rank(run_id) != 16:
                continue

            data_size = _run_data_size(run_id)

            if data_size in RQ1_DATA_SIZES:
                selected.append((data_size, run_id))

        if {size for size, _ in selected} != RQ1_DATA_SIZES:
            continue

        selected.sort()
        run_ids = [run_id for _, run_id in selected]

        row_pairs = _complete_prompt_seed_pairs(
            index=index,
            run_ids=run_ids,
            concept_id=concept_id,
        )

        if not row_pairs:
            continue

        labels = [
            f"n={data_size}"
            for data_size, _ in selected
        ]

        output_path = (
            output_dir
            / "rq1"
            / concept_id
            / "rq1_qualitative_grid.png"
        )

        generated.append(
            _plot_grid(
                records=records,
                run_ids=run_ids,
                row_pairs=row_pairs,
                column_labels=labels,
                title=f"RQ1 Qualitative Comparison — {concept_id}",
                output_path=output_path,
            )
        )

    return generated


def generate_rq2_grids(
    records: list[dict],
    output_dir: str | Path = "results/qualitative_grids",
) -> list[Path]:
    """
    Generate RQ2 qualitative grids.

    Requires all three configurations:
    rank=4, 16, 32 with data size n=5.
    """

    output_dir = Path(output_dir)
    index = _index_records(records)
    run_metadata = _run_metadata(records)

    generated = []

    concepts = sorted(
        {
            metadata["concept_id"]
            for metadata in run_metadata.values()
            if metadata["concept_id"]
        }
    )

    for concept_id in concepts:
        selected = []

        for run_id, metadata in run_metadata.items():
            if metadata["concept_id"] != concept_id:
                continue

            if _run_data_size(run_id) != 5:
                continue

            rank = _run_rank(run_id)

            if rank in RQ2_RANKS:
                selected.append((rank, run_id))

        if {rank for rank, _ in selected} != RQ2_RANKS:
            continue

        selected.sort()
        run_ids = [run_id for _, run_id in selected]

        row_pairs = _complete_prompt_seed_pairs(
            index=index,
            run_ids=run_ids,
            concept_id=concept_id,
        )

        if not row_pairs:
            continue

        labels = [
            f"rank={rank}"
            for rank, _ in selected
        ]

        output_path = (
            output_dir
            / "rq2"
            / concept_id
            / "rq2_qualitative_grid.png"
        )

        generated.append(
            _plot_grid(
                records=records,
                run_ids=run_ids,
                row_pairs=row_pairs,
                column_labels=labels,
                title=f"RQ2 Qualitative Comparison — {concept_id}",
                output_path=output_path,
            )
        )

    return generated


def generate_all_grids(
    artifacts_dir: str | Path = "artifacts",
    output_dir: str | Path = "results/qualitative_grids",
) -> list[Path]:
    """Generate all available RQ1 and RQ2 qualitative grids."""

    records = load_generation_metadata(artifacts_dir)

    return [
        *generate_rq1_grids(records, output_dir=output_dir),
        *generate_rq2_grids(records, output_dir=output_dir),
    ]
