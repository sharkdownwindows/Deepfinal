import csv
import math
from pathlib import Path

import torch
from PIL import Image

from personalized_t2i.evaluation.alignment import (
    ClipPromptScorer,
    load_concept_registry,
    normalize_prompt_for_clip,
    score_clip_records,
    upsert_clip_metrics_csv,
)


def _write_registry(path: Path) -> None:
    path.write_text(
        "concept_id,class_noun,unique_token,dataset_version,owner,consent_or_license\n"
        "cat_mug,mug,zzobj01,v1,project_team,self-captured\n"
        "dog_plush,plush toy,zzobj02,v1,project_team,self-captured\n"
        "blue_white_vase,vase,zzobj03,v1,project_team,self-captured\n",
        encoding="utf-8",
    )


def _make_image(path: Path) -> None:
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)


class FakeProcessor:
    def __init__(self):
        self.last_text = None

    def __call__(self, text, images, return_tensors, padding):
        self.last_text = text
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.ones((1, 3, 2, 2)),
        }


class FakeOutput:
    def __init__(self):
        self.image_embeds = torch.tensor([[1.0, 0.0, 0.0]])
        self.text_embeds = torch.tensor([[1.0, 0.0, 0.0]])


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def __call__(self, **kwargs):
        return FakeOutput()


class RecordingScorer:
    def __init__(self, score=0.75):
        self.score = score
        self.seen_text = []

    def score_image_text(self, image_path, text):
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"generated image does not exist: {path}")
        self.seen_text.append(text)
        return self.score


def test_unique_token_is_normalized_to_class_noun(tmp_path):
    registry_path = tmp_path / "concepts.csv"
    _write_registry(registry_path)

    registry = load_concept_registry(registry_path)

    assert normalize_prompt_for_clip(
        "a photo of zzobj01 mug on a table",
        "cat_mug",
        registry,
    ) == "a photo of mug on a table"

    assert normalize_prompt_for_clip(
        "a photo of zzobj02 plush toy in a room",
        "dog_plush",
        registry,
    ) == "a photo of plush toy in a room"

    assert normalize_prompt_for_clip(
        "a watercolor painting of zzobj03 vase",
        "blue_white_vase",
        registry,
    ) == "a watercolor painting of vase"


def test_clip_score_is_finite(tmp_path):
    image_path = tmp_path / "sample.png"
    _make_image(image_path)

    processor = FakeProcessor()
    scorer = ClipPromptScorer(
        revision="deadbeef",
        device="cpu",
        processor=processor,
        model=FakeModel(),
    )

    score = scorer.score_image_text(
        image_path,
        "a photo of mug",
    )

    assert isinstance(score, float)
    assert math.isfinite(score)
    assert -1.0 <= score <= 1.0
    assert score == 1.0
    assert processor.last_text == ["a photo of mug"]


def test_score_records_exports_clip_score_and_reports_invalid(tmp_path):
    valid_image = tmp_path / "valid.png"
    _make_image(valid_image)

    registry = {
        "cat_mug": {
            "class_noun": "mug",
            "unique_token": "zzobj01",
        }
    }

    records = [
        {
            "run_id": "cat_mug_n5_r16_ts42",
            "concept_id": "cat_mug",
            "checkpoint_step": 500,
            "rank": 16,
            "prompt_id": "p01",
            "prompt": "a photo of zzobj01 mug",
            "seed": 11,
            "image_path": str(valid_image),
        },
        {
            "run_id": "cat_mug_n5_r16_ts42",
            "concept_id": "cat_mug",
            "checkpoint_step": 500,
            "rank": 16,
            "prompt_id": "p02",
            "prompt": "a close-up photo of zzobj01 mug",
            "seed": 22,
            "image_path": str(tmp_path / "missing.png"),
        },
    ]

    scorer = RecordingScorer(score=0.75)

    rows = score_clip_records(
        records,
        scorer=scorer,
        registry=registry,
        resolved_config={
            "data": {"subset_size": 5},
            "training": {"rank": 16},
        },
        expected_run_id="cat_mug_n5_r16_ts42",
    )

    assert len(rows) == 2

    assert rows[0]["valid"] is True
    assert rows[0]["clip_prompt_similarity"] == 0.75
    assert rows[0]["data_size"] == 5
    assert rows[0]["rank"] == 16
    assert rows[0]["generation_seed"] == 11
    assert scorer.seen_text == ["a photo of mug"]

    assert rows[1]["valid"] is False
    assert rows[1]["clip_prompt_similarity"] is None
    assert "does not exist" in rows[1]["invalid_reason"]


def test_clip_upsert_preserves_existing_dino_score(tmp_path):
    output_path = tmp_path / "metrics_per_sample.csv"
    sample_id = "cat_mug_n5_r16_ts42__p01__gs11"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "dino_subject_similarity": 0.91,
                "clip_prompt_similarity": "",
                "lpips_diversity_optional": "",
                "valid": True,
                "invalid_reason": "",
            }
        )

    upsert_clip_metrics_csv(
        [
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "clip_prompt_similarity": 0.63,
                "valid": True,
                "invalid_reason": "",
            }
        ],
        output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert float(row["dino_subject_similarity"]) == 0.91
    assert float(row["clip_prompt_similarity"]) == 0.63
    assert row["valid"] == "True"


def test_invalid_rescore_clears_stale_clip_score_and_keeps_dino(tmp_path):
    output_path = tmp_path / "metrics_per_sample.csv"
    sample_id = "cat_mug_n5_r16_ts42__p01__gs11"

    upsert_clip_metrics_csv(
        [
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "clip_prompt_similarity": 0.63,
                "valid": True,
                "invalid_reason": "",
            }
        ],
        output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows[0]["dino_subject_similarity"] = "0.91"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    upsert_clip_metrics_csv(
        [
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "clip_prompt_similarity": None,
                "valid": False,
                "invalid_reason": "generated image does not exist",
            }
        ],
        output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["dino_subject_similarity"] == "0.91"
    assert row["clip_prompt_similarity"] == ""
    assert row["valid"] == "False"
    assert row["invalid_reason"] == "generated image does not exist"

def test_clip_valid_does_not_hide_existing_dino_invalid(tmp_path):
    output_path = tmp_path / "metrics_per_sample.csv"
    sample_id = "cat_mug_n5_r16_ts42__p01__gs11"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "dino_subject_similarity": "",
                "clip_prompt_similarity": "",
                "lpips_diversity_optional": "",
                "valid": False,
                "invalid_reason": "DINO reference image missing",
            }
        )

    upsert_clip_metrics_csv(
        [
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "clip_prompt_similarity": 0.72,
                "valid": True,
                "invalid_reason": "",
            }
        ],
        output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["clip_prompt_similarity"] == "0.72"
    assert row["valid"] == "False"
    assert row["invalid_reason"] == "DINO reference image missing"

def test_clip_rerun_can_recover_from_previous_clip_invalid(tmp_path):
    output_path = tmp_path / "metrics_per_sample.csv"
    sample_id = "cat_mug_n5_r16_ts42__p01__gs11"

    upsert_clip_metrics_csv(
        [
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "clip_prompt_similarity": None,
                "valid": False,
                "invalid_reason": "CLIP: generated image does not exist",
            }
        ],
        output_path,
    )

    upsert_clip_metrics_csv(
        [
            {
                "sample_id": sample_id,
                "run_id": "cat_mug_n5_r16_ts42",
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "generation_seed": 11,
                "checkpoint_step": 500,
                "rank": 16,
                "data_size": 5,
                "clip_prompt_similarity": 0.71,
                "valid": True,
                "invalid_reason": "",
            }
        ],
        output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["clip_prompt_similarity"] == "0.71"
    assert row["valid"] == "True"
    assert row["invalid_reason"] == ""


def test_clip_expected_run_id_match_is_valid():
    from personalized_t2i.evaluation.alignment import score_clip_records

    class FakeScorer:
        def score_image_text(self, image_path, text):
            return 0.42

    run_id = "cat_mug_n5_r16_ts42"

    rows = score_clip_records(
        records=[
            {
                "run_id": run_id,
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "prompt": "a photo of zzobj01 mug",
                "seed": 11,
                "image_path": "generated.png",
            }
        ],
        scorer=FakeScorer(),
        registry={
            "cat_mug": {
                "class_noun": "mug",
                "unique_token": "zzobj01",
            }
        },
        expected_run_id=run_id,
    )

    assert rows[0]["valid"] is True
    assert rows[0]["clip_prompt_similarity"] == 0.42
    assert rows[0]["invalid_reason"] == ""


def test_clip_expected_run_id_mismatch_is_invalid():
    from personalized_t2i.evaluation.alignment import score_clip_records

    class FakeScorer:
        def score_image_text(self, image_path, text):
            raise AssertionError(
                "CLIP scorer should not run for mismatched run_id"
            )

    expected_run_id = "cat_mug_n5_r16_ts42"

    rows = score_clip_records(
        records=[
            {
                "run_id": "dog_plush_n5_r16_ts42",
                "concept_id": "dog_plush",
                "prompt_id": "p01",
                "prompt": "a photo of zzobj02 plush toy",
                "seed": 11,
                "image_path": "not_used.png",
            }
        ],
        scorer=FakeScorer(),
        registry={
            "dog_plush": {
                "class_noun": "plush toy",
                "unique_token": "zzobj02",
            }
        },
        expected_run_id=expected_run_id,
    )

    assert rows[0]["valid"] is False
    assert rows[0]["clip_prompt_similarity"] is None
    assert rows[0]["sample_id"] == (
        "dog_plush_n5_r16_ts42__p01__gs11"
    )
    assert "CLIP: run_id mismatch" in rows[0]["invalid_reason"]
