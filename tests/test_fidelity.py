from types import SimpleNamespace

import torch
from PIL import Image

from personalized_t2i.evaluation.fidelity import Dinov2FidelityScorer


class FakeProcessor:
    def __call__(self, images, return_tensors):
        values = []

        for image in images:
            red, green, blue = image.getpixel((0, 0))
            values.append(
                [
                    red / 255.0,
                    green / 255.0,
                    blue / 255.0,
                    1.0,
                ]
            )

        return {
            "pixel_values": torch.tensor(
                values,
                dtype=torch.float32,
            )
        }


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, requires_grad):
        return self

    def __call__(self, pixel_values):
        return SimpleNamespace(pooler_output=pixel_values)


def test_dino_fidelity_smoke_score_is_finite(tmp_path):
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()

    reference_colors = [
        (220, 30, 30),
        (210, 40, 30),
        (230, 20, 40),
    ]

    for index, color in enumerate(reference_colors, start=1):
        Image.new("RGB", (8, 8), color).save(
            reference_dir / f"ref_{index}.png"
        )

    generated_path = tmp_path / "generated.png"
    Image.new("RGB", (8, 8), (225, 25, 35)).save(generated_path)

    scorer = Dinov2FidelityScorer(
        device="cpu",
        processor=FakeProcessor(),
        model=FakeModel(),
    )

    score = scorer.score_image_against_references(
        generated_path,
        reference_dir,
    )

    assert isinstance(score, float)
    assert torch.isfinite(torch.tensor(score))
    assert -1.0 <= score <= 1.0


def test_score_records_and_write_metrics_csv(tmp_path):
    import csv

    from personalized_t2i.evaluation.fidelity import (
        score_run_records,
        upsert_metrics_csv,
    )

    class FakeRunScorer:
        def build_reference_centroid(self, reference_dir):
            return torch.tensor([[1.0, 0.0]])

        def score_image(self, image_path, reference_centroid):
            return 0.75

    generated_path = tmp_path / "22.png"
    Image.new("RGB", (8, 8), (100, 120, 140)).save(generated_path)

    records = [
        {
            "run_id": "dog_plush_n5_r16_ts42",
            "concept_id": "dog_plush",
            "prompt_id": "p01",
            "seed": 22,
            "checkpoint_step": 500,
            "rank": 16,
            "image_path": str(generated_path),
        },
        {
            "run_id": "dog_plush_n5_r16_ts42",
            "concept_id": "dog_plush",
            "prompt_id": "p02",
            "seed": 22,
            "checkpoint_step": 500,
            "rank": 16,
            "image_path": str(tmp_path / "missing.png"),
        },
    ]

    rows = score_run_records(
        records=records,
        eval_refs_root=tmp_path / "eval_refs",
        scorer=FakeRunScorer(),
    )

    assert len(rows) == 2

    assert rows[0]["sample_id"] == "dog_plush_n5_r16_ts42__p01__gs22"
    assert rows[0]["dino_subject_similarity"] == 0.75
    assert rows[0]["valid"] is True
    assert rows[0]["invalid_reason"] == ""
    assert rows[0]["data_size"] == 5

    assert rows[1]["valid"] is False
    assert rows[1]["dino_subject_similarity"] is None
    assert "does not exist" in rows[1]["invalid_reason"]

    output_path = tmp_path / "metrics_per_sample.csv"
    upsert_metrics_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))

    assert len(written) == 2
    assert written[0]["dino_subject_similarity"] == "0.75"
    assert written[0]["valid"] == "True"
    assert written[1]["dino_subject_similarity"] == ""
    assert written[1]["valid"] == "False"


def test_upsert_clears_stale_dino_score_for_invalid_sample(tmp_path):
    import csv

    from personalized_t2i.evaluation.fidelity import upsert_metrics_csv

    output_path = tmp_path / "metrics_per_sample.csv"

    valid_row = {
        "sample_id": "dog_plush_n5_r16_ts42__p01__gs22",
        "run_id": "dog_plush_n5_r16_ts42",
        "concept_id": "dog_plush",
        "prompt_id": "p01",
        "generation_seed": 22,
        "checkpoint_step": 500,
        "rank": 16,
        "data_size": 5,
        "dino_subject_similarity": 0.81,
        "clip_prompt_similarity": None,
        "lpips_diversity_optional": None,
        "valid": True,
        "invalid_reason": "",
    }

    invalid_row = {
        **valid_row,
        "dino_subject_similarity": None,
        "valid": False,
        "invalid_reason": "DINO: generated image does not exist",
    }

    upsert_metrics_csv([valid_row], output_path)
    upsert_metrics_csv([invalid_row], output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["dino_subject_similarity"] == ""
    assert row["valid"] == "False"
    assert row["invalid_reason"] == "DINO: generated image does not exist"


def test_invalid_metadata_record_is_retained(tmp_path):
    import csv

    from personalized_t2i.evaluation.fidelity import (
        score_run_records,
        upsert_metrics_csv,
    )

    class FakeRunScorer:
        def build_reference_centroid(self, reference_dir):
            return torch.tensor([[1.0, 0.0]])

        def score_image(self, image_path, reference_centroid):
            return 0.5

    records = [
        {
            "concept_id": "dog_plush",
            "prompt_id": "p01",
            "seed": 22,
            "image_path": str(tmp_path / "missing.png"),
        }
    ]

    rows = score_run_records(
        records=records,
        eval_refs_root=tmp_path / "eval_refs",
        scorer=FakeRunScorer(),
        expected_run_id="dog_plush_n5_r16_ts42",
    )

    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert rows[0]["sample_id"] == (
        "invalid__dog_plush_n5_r16_ts42__row0001"
    )
    assert rows[0]["dino_subject_similarity"] is None
    assert "missing run_id" in rows[0]["invalid_reason"]

    output_path = tmp_path / "metrics_per_sample.csv"
    upsert_metrics_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))

    assert len(written) == 1
    assert written[0]["sample_id"] == (
        "invalid__dog_plush_n5_r16_ts42__row0001"
    )
    assert written[0]["valid"] == "False"
    assert written[0]["dino_subject_similarity"] == ""


def test_reference_set_requires_exactly_three_images(tmp_path):
    import pytest

    from personalized_t2i.evaluation.fidelity import list_reference_images

    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()

    for index in range(2):
        Image.new("RGB", (8, 8), (100, 100, 100)).save(
            reference_dir / f"ref_{index}.png"
        )

    with pytest.raises(
        ValueError,
        match="Expected exactly 3 held-out reference images",
    ):
        list_reference_images(reference_dir)

def test_reference_hash_change_is_rejected(tmp_path):
    import csv
    import hashlib
    import pytest

    from personalized_t2i.evaluation.fidelity import list_reference_images

    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    manifest_path = tmp_path / "manifest.csv"

    rows = []

    for index, color in enumerate(
        [(100, 10, 10), (10, 100, 10), (10, 10, 100)],
        start=1,
    ):
        path = reference_dir / f"ref_{index}.png"
        Image.new("RGB", (8, 8), color).save(path)

        rows.append(
            {
                "file_path": f"data/eval_refs/cat_mug/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "concept_id": "cat_mug",
                "split": "heldout",
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_path", "sha256", "concept_id", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    verified = list_reference_images(
        reference_dir,
        manifest_path=manifest_path,
        concept_id="cat_mug",
    )
    assert len(verified) == 3

    Image.new("RGB", (8, 8), (255, 255, 255)).save(
        reference_dir / "ref_2.png"
    )

    with pytest.raises(
        ValueError,
        match="Held-out reference hash mismatch",
    ):
        list_reference_images(
            reference_dir,
            manifest_path=manifest_path,
            concept_id="cat_mug",
        )


def test_expected_run_id_match_is_valid(tmp_path):
    from personalized_t2i.evaluation.fidelity import score_run_records

    class FakeRunScorer:
        def build_reference_centroid(self, reference_dir):
            return torch.tensor([[1.0, 0.0]])

        def score_image(self, image_path, reference_centroid):
            return 0.8

    generated_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), (100, 100, 100)).save(generated_path)

    run_id = "cat_mug_n5_r16_ts42"

    rows = score_run_records(
        records=[
            {
                "run_id": run_id,
                "concept_id": "cat_mug",
                "prompt_id": "p01",
                "seed": 11,
                "image_path": str(generated_path),
            }
        ],
        eval_refs_root=tmp_path / "eval_refs",
        scorer=FakeRunScorer(),
        expected_run_id=run_id,
    )

    assert rows[0]["valid"] is True
    assert rows[0]["dino_subject_similarity"] == 0.8


def test_expected_run_id_mismatch_is_invalid(tmp_path):
    from personalized_t2i.evaluation.fidelity import score_run_records

    class FakeRunScorer:
        def build_reference_centroid(self, reference_dir):
            raise AssertionError("scorer should not run for mismatched run_id")

        def score_image(self, image_path, reference_centroid):
            raise AssertionError("scorer should not run for mismatched run_id")

    expected_run_id = "cat_mug_n5_r16_ts42"

    rows = score_run_records(
        records=[
            {
                "run_id": "dog_plush_n5_r16_ts42",
                "concept_id": "dog_plush",
                "prompt_id": "p01",
                "seed": 11,
                "image_path": str(tmp_path / "not_used.png"),
            }
        ],
        eval_refs_root=tmp_path / "eval_refs",
        scorer=FakeRunScorer(),
        expected_run_id=expected_run_id,
    )

    assert rows[0]["valid"] is False
    assert rows[0]["dino_subject_similarity"] is None
    assert "run_id mismatch" in rows[0]["invalid_reason"]


def test_dino_valid_does_not_hide_existing_clip_invalid(tmp_path):
    import csv

    from personalized_t2i.evaluation.fidelity import upsert_metrics_csv

    output_path = tmp_path / "metrics.csv"
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
                "invalid_reason": "CLIP: generated image does not exist",
            }
        )

    upsert_metrics_csv(
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
                "dino_subject_similarity": 0.88,
                "valid": True,
                "invalid_reason": "",
            }
        ],
        output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["dino_subject_similarity"] == "0.88"
    assert row["valid"] == "False"
    assert row["invalid_reason"] == "CLIP: generated image does not exist"


def test_dino_rerun_can_recover_from_previous_dino_invalid(tmp_path):
    import csv

    from personalized_t2i.evaluation.fidelity import upsert_metrics_csv

    output_path = tmp_path / "metrics.csv"
    sample_id = "cat_mug_n5_r16_ts42__p01__gs11"

    base_row = {
        "sample_id": sample_id,
        "run_id": "cat_mug_n5_r16_ts42",
        "concept_id": "cat_mug",
        "prompt_id": "p01",
        "generation_seed": 11,
        "checkpoint_step": 500,
        "rank": 16,
        "data_size": 5,
    }

    upsert_metrics_csv(
        [
            {
                **base_row,
                "dino_subject_similarity": None,
                "valid": False,
                "invalid_reason": "DINO: generated image does not exist",
            }
        ],
        output_path,
    )

    upsert_metrics_csv(
        [
            {
                **base_row,
                "dino_subject_similarity": 0.9,
                "valid": True,
                "invalid_reason": "",
            }
        ],
        output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["dino_subject_similarity"] == "0.9"
    assert row["valid"] == "True"
    assert row["invalid_reason"] == ""
