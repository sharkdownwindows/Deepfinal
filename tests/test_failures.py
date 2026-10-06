from personalized_t2i.evaluation.failures import (
    FAILURE_CASE_COLUMNS,
    load_failure_cases,
    validate_failure_rows,
)


def valid_case(**updates):
    row = {
        "case_id": "FC-001",
        "run_id": "toy01_n5_r16_ts42",
        "concept_id": "toy01",
        "prompt_id": "p03",
        "generation_seed": "11",
        "failure_tag": "identity_drift",
        "attribution": "lora",
        "severity": "medium",
        "lora_image_path": "artifacts/toy01_n5_r16_ts42/generated/p03/11.png",
        "base_image_path": "artifacts/base_toy01/generated/p03/11.png",
        "base_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "base_model_revision": "abc123",
        "nearest_training_image": "",
        "automated_evidence": "dino_subject_similarity=0.42",
        "review_status": "confirmed",
        "reviewer_id": "reviewer-a",
        "notes": "Distinctive blue marking is absent in the LoRA output.",
    }
    row.update(updates)
    return row


def test_accepts_traceable_paired_case():
    assert validate_failure_rows([valid_case()]) == []


def test_requires_run_prompt_seed_and_integer_seed():
    errors = validate_failure_rows(
        [valid_case(run_id="", prompt_id="", generation_seed="seed-11")]
    )
    assert any("run_id is required" in error for error in errors)
    assert any("prompt_id is required" in error for error in errors)
    assert any("must be an integer" in error for error in errors)


def test_requires_paired_base_evidence_for_attribution():
    errors = validate_failure_rows([valid_case(base_image_path="", base_model_revision="")])
    assert any("compare base and LoRA outputs" in error for error in errors)


def test_memorization_requires_nearest_training_image():
    errors = validate_failure_rows([valid_case(failure_tag="memorization")])
    assert any("nearest_training_image is required" in error for error in errors)


def test_loader_rejects_incomplete_schema(tmp_path):
    source = tmp_path / "failures.csv"
    source.write_text("case_id,run_id\n", encoding="utf-8")

    rows, errors = load_failure_cases(source)

    assert rows == []
    assert errors == [
        "CSV is missing required columns: "
        + ", ".join(column for column in FAILURE_CASE_COLUMNS if column not in {"case_id", "run_id"})
    ]


def test_duplicate_case_id_is_rejected():
    errors = validate_failure_rows([valid_case(), valid_case()])
    assert any("duplicate case_id" in error for error in errors)
