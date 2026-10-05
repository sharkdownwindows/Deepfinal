"""DINOv2 subject-fidelity scoring."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model


DINO_MODEL_ID = "facebook/dinov2-base"
DINO_MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"

METRICS_COLUMNS = [
    "sample_id",
    "run_id",
    "concept_id",
    "prompt_id",
    "generation_seed",
    "checkpoint_step",
    "rank",
    "data_size",
    "dino_subject_similarity",
    "clip_prompt_similarity",
    "lpips_diversity_optional",
    "valid",
    "invalid_reason",
]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def make_sample_id(
    run_id: str,
    prompt_id: str,
    generation_seed: int,
) -> str:
    return f"{run_id}__{prompt_id}__gs{generation_seed}"


def list_reference_images(reference_dir: str | Path) -> list[Path]:
    reference_dir = Path(reference_dir)

    if not reference_dir.is_dir():
        raise FileNotFoundError(
            f"Held-out reference directory does not exist: {reference_dir}"
        )

    image_paths = sorted(
        path
        for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )

    if len(image_paths) != 3:
        raise ValueError(
            f"Expected exactly 3 held-out reference images in "
            f"{reference_dir}, found {len(image_paths)}"
        )

    return image_paths


def load_metadata_records(metadata_path: str | Path) -> list[dict]:
    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    records = []

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {metadata_path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Metadata line {line_number} must contain a JSON object"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No metadata records found in {metadata_path}")

    return records


def load_resolved_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.is_file():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return config if isinstance(config, dict) else {}


def _data_size_from_run_id(run_id: str) -> int | None:
    match = re.search(r"_n(\d+)_r\d+", run_id)
    return int(match.group(1)) if match else None


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class Dinov2FidelityScorer:
    """Score generated images against a held-out DINOv2 reference centroid."""

    def __init__(
        self,
        model_id: str = DINO_MODEL_ID,
        revision: str = DINO_MODEL_REVISION,
        device: str | torch.device | None = None,
        processor=None,
        model=None,
    ) -> None:
        if not revision or revision in {"main", "master", "latest"}:
            raise ValueError(
                "DINOv2 revision must be an immutable pinned revision"
            )

        self.model_id = model_id
        self.revision = revision
        self.device = (
            torch.device(device)
            if device is not None
            else _default_device()
        )

        self.processor = processor or AutoImageProcessor.from_pretrained(
            model_id,
            revision=revision,
        )
        self.model = model or Dinov2Model.from_pretrained(
            model_id,
            revision=revision,
        )

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode_images(
        self,
        image_paths: Sequence[str | Path],
    ) -> torch.Tensor:
        paths = [Path(path) for path in image_paths]

        if not paths:
            raise ValueError("At least one image is required")

        images = []

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image does not exist: {path}")

            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        embeddings = outputs.pooler_output

        if embeddings.ndim != 2 or embeddings.shape[0] != len(paths):
            raise RuntimeError(
                "Unexpected DINOv2 embedding shape: "
                f"{tuple(embeddings.shape)}"
            )

        if not torch.isfinite(embeddings).all():
            raise ValueError("DINOv2 produced a non-finite embedding")

        return embeddings.detach().float().cpu()

    def build_reference_centroid(
        self,
        reference_dir: str | Path,
    ) -> torch.Tensor:
        reference_paths = list_reference_images(reference_dir)
        reference_embeddings = self.encode_images(reference_paths)
        centroid = reference_embeddings.mean(dim=0, keepdim=True)

        if not torch.isfinite(centroid).all():
            raise ValueError("Held-out reference centroid is non-finite")

        if torch.linalg.vector_norm(centroid).item() == 0.0:
            raise ValueError("Held-out reference centroid has zero norm")

        return centroid

    def score_image(
        self,
        image_path: str | Path,
        reference_centroid: torch.Tensor,
    ) -> float:
        image_embedding = self.encode_images([image_path])
        centroid = reference_centroid.detach().float().cpu().reshape(1, -1)

        if centroid.shape[1] != image_embedding.shape[1]:
            raise ValueError(
                "Generated-image embedding and reference centroid "
                "have different dimensions"
            )

        score = F.cosine_similarity(
            image_embedding,
            centroid,
            dim=1,
        ).item()

        if not math.isfinite(score):
            raise ValueError(
                "DINOv2 subject-fidelity score is non-finite"
            )

        return float(score)

    def score_image_against_references(
        self,
        image_path: str | Path,
        reference_dir: str | Path,
    ) -> float:
        centroid = self.build_reference_centroid(reference_dir)
        return self.score_image(image_path, centroid)


def score_run_records(
    records: Sequence[dict],
    eval_refs_root: str | Path,
    scorer: Dinov2FidelityScorer,
    resolved_config: dict | None = None,
    expected_run_id: str | None = None,
) -> list[dict]:
    eval_refs_root = Path(eval_refs_root)
    resolved_config = resolved_config or {}

    config_data = resolved_config.get("data", {})
    config_training = resolved_config.get("training", {})

    rows = []
    centroid_cache: dict[str, torch.Tensor] = {}

    for record_index, record in enumerate(records, start=1):
        run_id = str(record.get("run_id") or "")
        concept_id = str(record.get("concept_id") or "")
        prompt_id = str(record.get("prompt_id") or "")
        seed = record.get("seed")
        image_path = record.get("image_path")

        rank = config_training.get("rank", record.get("rank"))
        data_size = config_data.get(
            "subset_size",
            _data_size_from_run_id(run_id),
        )

        row = {
            "sample_id": "",
            "run_id": run_id,
            "concept_id": concept_id,
            "prompt_id": prompt_id,
            "generation_seed": seed,
            "checkpoint_step": record.get("checkpoint_step"),
            "rank": rank,
            "data_size": data_size,
            "dino_subject_similarity": None,
            "clip_prompt_similarity": None,
            "lpips_diversity_optional": None,
            "valid": False,
            "invalid_reason": "",
        }

        try:
            if not run_id:
                raise ValueError("missing run_id")
            if not concept_id:
                raise ValueError("missing concept_id")
            if not prompt_id:
                raise ValueError("missing prompt_id")
            if type(seed) is not int:
                raise ValueError("missing or invalid generation seed")
            if not image_path:
                raise ValueError("missing image_path")

            row["sample_id"] = make_sample_id(
                run_id,
                prompt_id,
                seed,
            )

            image_path = Path(image_path)

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"generated image does not exist: {image_path}"
                )

            if concept_id not in centroid_cache:
                centroid_cache[concept_id] = scorer.build_reference_centroid(
                    eval_refs_root / concept_id
                )

            score = scorer.score_image(
                image_path,
                centroid_cache[concept_id],
            )

            if not math.isfinite(score):
                raise ValueError(
                    "DINOv2 subject-fidelity score is non-finite"
                )

            row["dino_subject_similarity"] = score
            row["valid"] = True

        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            if not row["sample_id"]:
                if run_id and prompt_id and type(seed) is int:
                    row["sample_id"] = make_sample_id(
                        run_id,
                        prompt_id,
                        seed,
                    )
                else:
                    fallback_run_id = expected_run_id or run_id or "unknown"
                    row["sample_id"] = (
                        f"invalid__{fallback_run_id}__row{record_index:04d}"
                    )

            row["invalid_reason"] = str(exc)

        rows.append(row)

    return rows


def upsert_metrics_csv(
    rows: Sequence[dict],
    output_path: str | Path = "results/metrics_per_sample.csv",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}

    if output_path.is_file():
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                sample_id = row.get("sample_id", "")
                if sample_id:
                    existing[sample_id] = row

    for new_row in rows:
        sample_id = new_row.get("sample_id", "")

        if not sample_id:
            continue

        merged = {
            column: existing.get(sample_id, {}).get(column, "")
            for column in METRICS_COLUMNS
        }

        for key, value in new_row.items():
            if key not in METRICS_COLUMNS:
                continue

            if key == "dino_subject_similarity":
                merged[key] = "" if value is None else value
            elif value is not None:
                merged[key] = value

        existing[sample_id] = merged

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METRICS_COLUMNS,
        )
        writer.writeheader()

        for sample_id in sorted(existing):
            writer.writerow(existing[sample_id])

    return output_path


def evaluate_run_dino(
    run_id: str,
    artifacts_root: str | Path = "artifacts",
    eval_refs_root: str | Path = "data/eval_refs",
    output_path: str | Path = "results/metrics_per_sample.csv",
    scorer: Dinov2FidelityScorer | None = None,
) -> list[dict]:
    run_dir = Path(artifacts_root) / run_id

    records = load_metadata_records(run_dir / "metadata.jsonl")
    resolved_config = load_resolved_config(
        run_dir / "config.resolved.yaml"
    )

    scorer = scorer or Dinov2FidelityScorer()

    rows = score_run_records(
        records=records,
        eval_refs_root=eval_refs_root,
        scorer=scorer,
        resolved_config=resolved_config,
        expected_run_id=run_id,
    )

    upsert_metrics_csv(rows, output_path)

    return rows
