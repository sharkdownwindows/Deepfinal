from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from personalized_t2i.evaluation.fidelity import (
    METRICS_COLUMNS,
    load_metadata_records,
    load_resolved_config,
    make_sample_id,
)


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

DEFAULT_CONCEPT_REGISTRY = "data/manifests/concepts.csv"


def load_concept_registry(
    registry_path: str | Path = DEFAULT_CONCEPT_REGISTRY,
) -> dict[str, dict[str, str]]:
    """Load concept ID, class noun, and unique token mappings."""
    registry_path = Path(registry_path)

    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Concept registry does not exist: {registry_path}"
        )

    registry: dict[str, dict[str, str]] = {}

    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {
            "concept_id",
            "class_noun",
            "unique_token",
        }

        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Concept registry must contain concept_id, "
                "class_noun, and unique_token"
            )

        for row in reader:
            concept_id = (row.get("concept_id") or "").strip()
            class_noun = (row.get("class_noun") or "").strip()
            unique_token = (row.get("unique_token") or "").strip()

            if not concept_id or not class_noun or not unique_token:
                raise ValueError(
                    "Concept registry contains an incomplete concept row"
                )

            if concept_id in registry:
                raise ValueError(
                    f"Duplicate concept_id in registry: {concept_id}"
                )

            registry[concept_id] = {
                "class_noun": class_noun,
                "unique_token": unique_token,
            }

    if not registry:
        raise ValueError("Concept registry is empty")

    return registry


def normalize_prompt_for_clip(
    prompt: str,
    concept_id: str,
    registry: dict[str, dict[str, str]],
) -> str:
    """Replace the concept unique token with its class noun for CLIP text."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if concept_id not in registry:
        raise ValueError(
            f"concept_id not found in concept registry: {concept_id}"
        )

    unique_token = registry[concept_id]["unique_token"]
    class_noun = registry[concept_id]["class_noun"]

    if unique_token not in prompt:
        raise ValueError(
            f"unique token {unique_token!r} not found in prompt"
        )

    instance_phrase = f"{unique_token} {class_noun}"
    normalized = prompt.replace(instance_phrase, class_noun)
    normalized = normalized.replace(unique_token, class_noun)

    if unique_token in normalized:
        raise ValueError(
            f"unique token {unique_token!r} remains after normalization"
        )

    return normalized


def _data_size_from_run_id(run_id: str) -> int | None:
    import re

    match = re.search(r"_n(\d+)_r\d+", run_id)
    return int(match.group(1)) if match else None


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class ClipPromptScorer:
    """Score image-text prompt alignment with frozen CLIP."""

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        revision: str = CLIP_MODEL_REVISION,
        device: str | torch.device | None = None,
        processor=None,
        model=None,
    ) -> None:
        if not revision or revision in {"main", "master", "latest"}:
            raise ValueError(
                "CLIP revision must be an immutable pinned revision"
            )

        self.model_id = model_id
        self.revision = revision
        self.device = (
            torch.device(device)
            if device is not None
            else _default_device()
        )

        self.processor = processor or CLIPProcessor.from_pretrained(
            model_id,
            revision=revision,
        )
        self.model = model or CLIPModel.from_pretrained(
            model_id,
            revision=revision,
        )

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def score_image_text(
        self,
        image_path: str | Path,
        text: str,
    ) -> float:
        """Return cosine similarity between one image and one text prompt."""
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"generated image does not exist: {image_path}"
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError("CLIP text must be a non-empty string")

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        inputs = self.processor(
            text=[text],
            images=[rgb_image],
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        image_embedding = outputs.image_embeds
        text_embedding = outputs.text_embeds

        if image_embedding.ndim != 2 or image_embedding.shape[0] != 1:
            raise RuntimeError(
                "Unexpected CLIP image embedding shape: "
                f"{tuple(image_embedding.shape)}"
            )

        if text_embedding.ndim != 2 or text_embedding.shape[0] != 1:
            raise RuntimeError(
                "Unexpected CLIP text embedding shape: "
                f"{tuple(text_embedding.shape)}"
            )

        if image_embedding.shape[1] != text_embedding.shape[1]:
            raise RuntimeError(
                "CLIP image and text embeddings have different dimensions"
            )

        if not torch.isfinite(image_embedding).all():
            raise ValueError("CLIP produced a non-finite image embedding")

        if not torch.isfinite(text_embedding).all():
            raise ValueError("CLIP produced a non-finite text embedding")

        score = F.cosine_similarity(
            image_embedding.float(),
            text_embedding.float(),
            dim=1,
        ).item()

        if not math.isfinite(score):
            raise ValueError(
                "CLIP prompt-alignment score is non-finite"
            )

        return float(score)


def score_clip_records(
    records: Sequence[dict],
    scorer: ClipPromptScorer,
    registry: dict[str, dict[str, str]],
    resolved_config: dict | None = None,
    expected_run_id: str | None = None,
) -> list[dict]:
    """Score generated-sample metadata records with CLIP."""
    resolved_config = resolved_config or {}

    config_data = resolved_config.get("data", {})
    config_training = resolved_config.get("training", {})

    rows = []

    for record_index, record in enumerate(records, start=1):
        run_id = str(record.get("run_id") or "")
        concept_id = str(record.get("concept_id") or "")
        prompt_id = str(record.get("prompt_id") or "")
        prompt = record.get("prompt")
        seed = record.get("seed")
        generation_mode = str(record.get("generation_mode") or "")
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
            "generation_mode": generation_mode,
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
            if expected_run_id is not None and run_id != expected_run_id:
                raise ValueError(
                    f"run_id mismatch: expected {expected_run_id}, got {run_id}"
                )
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
                generation_mode or None,
            )

            normalized_prompt = normalize_prompt_for_clip(
                prompt=prompt,
                concept_id=concept_id,
                registry=registry,
            )

            score = scorer.score_image_text(
                image_path=image_path,
                text=normalized_prompt,
            )

            if not math.isfinite(score):
                raise ValueError(
                    "CLIP prompt-alignment score is non-finite"
                )

            row["clip_prompt_similarity"] = score
            row["valid"] = True

        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            if not row["sample_id"]:
                if run_id and prompt_id and type(seed) is int:
                    row["sample_id"] = make_sample_id(
                        run_id,
                        prompt_id,
                        seed,
                        generation_mode or None,
                    )
                else:
                    fallback_run_id = expected_run_id or run_id or "unknown"
                    row["sample_id"] = (
                        f"invalid__{fallback_run_id}__row{record_index:04d}"
                    )

            row["invalid_reason"] = f"CLIP: {exc}"

        rows.append(row)

    return rows


def upsert_clip_metrics_csv(
    rows: Sequence[dict],
    output_path: str | Path = "results/metrics_per_sample.csv",
) -> Path:
    """Update only CLIP-owned fields while preserving existing DINO values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    fieldnames = METRICS_COLUMNS

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
            for column in fieldnames
        }

        for key in (
            "sample_id",
            "run_id",
            "concept_id",
            "prompt_id",
            "generation_seed",
            "generation_mode",
            "checkpoint_step",
            "rank",
            "data_size",
        ):
            value = new_row.get(key)
            if value is not None:
                merged[key] = value

        clip_score = new_row.get("clip_prompt_similarity")
        merged["clip_prompt_similarity"] = (
            "" if clip_score is None else clip_score
        )

        previous = existing.get(sample_id, {})
        previous_reason = str(
            previous.get("invalid_reason", "") or ""
        ).strip()
        current_reason = str(
            new_row.get("invalid_reason", "") or ""
        ).strip()

        retained_reasons = [
            reason.strip()
            for reason in previous_reason.split(";")
            if reason.strip() and not reason.strip().startswith("CLIP:")
        ]

        reasons = retained_reasons.copy()
        if current_reason and current_reason not in reasons:
            reasons.append(current_reason)

        current_valid = bool(new_row.get("valid", False))
        merged["valid"] = current_valid and not reasons
        merged["invalid_reason"] = "; ".join(reasons)

        existing[sample_id] = merged

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for sample_id in sorted(existing):
            writer.writerow(existing[sample_id])

    return output_path


def evaluate_run_clip(
    run_id: str,
    artifacts_root: str | Path = "artifacts",
    concept_registry_path: str | Path = DEFAULT_CONCEPT_REGISTRY,
    output_path: str | Path = "results/metrics_per_sample.csv",
    scorer: ClipPromptScorer | None = None,
) -> list[dict]:
    """Evaluate one generated run with CLIP prompt alignment."""
    run_dir = Path(artifacts_root) / run_id

    records = load_metadata_records(run_dir / "metadata.jsonl")
    resolved_config = load_resolved_config(
        run_dir / "config.resolved.yaml"
    )
    registry = load_concept_registry(concept_registry_path)
    scorer = scorer or ClipPromptScorer()

    rows = score_clip_records(
        records=records,
        scorer=scorer,
        registry=registry,
        resolved_config=resolved_config,
        expected_run_id=run_id,
    )

    upsert_clip_metrics_csv(rows, output_path)

    return rows
