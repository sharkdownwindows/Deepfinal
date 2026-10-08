import csv
import statistics

import pytest

from personalized_t2i.evaluation.aggregate import (
    AGGREGATE_COLUMNS,
    aggregate_metrics_file,
    aggregate_rows,
)


def make_row(
    seed,
    *,
    generation_mode="adapter",
    prompt_id="p01",
    dino="0.8",
    clip="0.7",
    valid="True",
    invalid_reason="",
    concept_id="dog_plush",
):
    run_id = "dog_plush_n5_r16"

    return {
        "sample_id": f"{run_id}__{generation_mode}__{prompt_id}__gs{seed}",
        "run_id": run_id,
        "generation_mode": generation_mode,
        "concept_id": concept_id,
        "prompt_id": prompt_id,
        "generation_seed": str(seed),
        "checkpoint_step": "500",
        "rank": "16",
        "data_size": "5",
        "dino_subject_similarity": dino,
        "clip_prompt_similarity": clip,
        "lpips_diversity_optional": "",
        "valid": valid,
        "invalid_reason": invalid_reason,
    }


def test_mean_sd_and_counts():
    rows = [
        make_row(11, dino="0.1", clip="0.4"),
        make_row(22, dino="0.2", clip="0.5"),
        make_row(33, dino="0.3", clip="0.6"),
        make_row(44, dino="0.4", clip="0.7"),
    ]

    result = aggregate_rows(rows, expected_sample_count=4)[0]

    assert result["observed_sample_count"] == 4
    assert result["missing_sample_count"] == 0

    assert result["dino_mean"] == pytest.approx(0.25)
    assert result["dino_sd"] == pytest.approx(
        statistics.pstdev([0.1, 0.2, 0.3, 0.4])
    )
    assert result["dino_sample_count"] == 4
    assert result["dino_missing_count"] == 0

    assert result["clip_mean"] == pytest.approx(0.55)
    assert result["clip_sd"] == pytest.approx(
        statistics.pstdev([0.4, 0.5, 0.6, 0.7])
    )
    assert result["clip_sample_count"] == 4
    assert result["clip_missing_count"] == 0


def test_missing_scores_are_not_zero():
    rows = [
        make_row(11, dino="0.8", clip="0.6"),
        make_row(
            22,
            dino="",
            clip="0.7",
            valid="False",
            invalid_reason="DINO: reference embedding failed",
        ),
        make_row(33, dino="0.6", clip="0.8"),
    ]

    result = aggregate_rows(rows, expected_sample_count=4)[0]

    assert result["observed_sample_count"] == 3
    assert result["missing_sample_count"] == 1
    assert result["invalid_sample_count"] == 1

    assert result["dino_sample_count"] == 2
    assert result["dino_missing_count"] == 2
    assert result["dino_mean"] == pytest.approx(0.7)

    assert result["clip_sample_count"] == 3
    assert result["clip_missing_count"] == 1
    assert result["clip_mean"] == pytest.approx(0.7)


def test_metric_specific_failure_does_not_drop_other_metric():
    row = make_row(
        11,
        dino="",
        clip="0.73",
        valid="False",
        invalid_reason="DINO: reference embedding failed",
    )

    result = aggregate_rows([row], expected_sample_count=1)[0]

    assert result["dino_sample_count"] == 0
    assert result["dino_invalid_count"] == 1
    assert result["dino_mean"] is None
    assert result["clip_sample_count"] == 1
    assert result["clip_invalid_count"] == 0
    assert result["clip_mean"] == pytest.approx(0.73)


def test_base_and_adapter_are_aggregated_separately_for_same_run():
    rows = []
    for mode, score in (("base", "0.2"), ("adapter", "0.8")):
        for prompt_id in (f"p{index:02d}" for index in range(1, 9)):
            for seed in (11, 22, 33, 44):
                rows.append(
                    make_row(
                        seed,
                        generation_mode=mode,
                        prompt_id=prompt_id,
                        dino=score,
                        clip=score,
                    )
                )

    results = aggregate_rows(rows)

    assert [(row["run_id"], row["generation_mode"]) for row in results] == [
        ("dog_plush_n5_r16", "adapter"),
        ("dog_plush_n5_r16", "base"),
    ]
    adapter, base = results
    assert adapter["observed_sample_count"] == 32
    assert adapter["missing_sample_count"] == 0
    assert adapter["dino_mean"] == pytest.approx(0.8)
    assert adapter["clip_mean"] == pytest.approx(0.8)
    assert base["observed_sample_count"] == 32
    assert base["missing_sample_count"] == 0
    assert base["dino_mean"] == pytest.approx(0.2)
    assert base["clip_mean"] == pytest.approx(0.2)


def test_protocol_missing_count_uses_prompt_seed_identity():
    rows = [
        make_row(seed, prompt_id=prompt_id)
        for prompt_id in (f"p{index:02d}" for index in range(1, 9))
        for seed in (11, 22, 33, 44)
        if (prompt_id, seed) != ("p03", 22)
    ]

    result = aggregate_rows(rows)[0]

    assert result["observed_sample_count"] == 31
    assert result["missing_sample_count"] == 1


def test_unexpected_protocol_prompt_seed_is_rejected():
    rows = [
        make_row(seed, prompt_id=prompt_id)
        for prompt_id in (f"p{index:02d}" for index in range(1, 9))
        for seed in (11, 22, 33, 44)
    ]
    rows[0]["prompt_id"] = "p09"

    with pytest.raises(ValueError, match="unexpected prompt/seed identities"):
        aggregate_rows(rows)


def test_duplicate_prompt_seed_identity_is_rejected_even_with_distinct_sample_ids():
    first = make_row(11)
    duplicate = dict(first, sample_id="different-sample-id")

    with pytest.raises(ValueError, match="duplicate prompt_id/generation_seed"):
        aggregate_rows([first, duplicate], expected_sample_count=4)


def test_generation_mode_is_required():
    row = make_row(11)
    row["generation_mode"] = ""

    with pytest.raises(ValueError, match="without generation_mode"):
        aggregate_rows([row], expected_sample_count=1)


def test_duplicate_sample_is_rejected():
    row = make_row(11)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        aggregate_rows(
            [row, dict(row)],
            expected_sample_count=4,
        )


def test_non_finite_score_is_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        aggregate_rows(
            [make_row(11, dino="nan")],
            expected_sample_count=4,
        )


def test_aggregate_csv_is_written(tmp_path):
    input_path = tmp_path / "metrics_per_sample.csv"
    output_path = tmp_path / "metrics_aggregate.csv"

    rows = [
        make_row(11, dino="0.8", clip="0.6"),
        make_row(22, dino="0.6", clip="0.8"),
    ]

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    result = aggregate_metrics_file(
        input_path,
        output_path,
        expected_sample_count=2,
    )

    assert len(result) == 1
    assert output_path.is_file()

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)

    assert reader.fieldnames == AGGREGATE_COLUMNS
    assert len(written) == 1
    assert written[0]["dino_sample_count"] == "2"
    assert written[0]["clip_sample_count"] == "2"
    assert written[0]["generation_mode"] == "adapter"
    assert "dino_invalid_count" in reader.fieldnames


def test_requested_run_filter():
    dog = make_row(11)

    cat = dict(dog)
    cat["run_id"] = "cat_mug_n5_r16_ts42"
    cat["sample_id"] = "cat_mug_n5_r16_ts42__p01__gs11"
    cat["concept_id"] = "cat_mug"

    result = aggregate_rows(
        [dog, cat],
        expected_sample_count=1,
        run_ids=["cat_mug_n5_r16_ts42"],
    )

    assert len(result) == 1
    assert result[0]["run_id"] == "cat_mug_n5_r16_ts42"


def test_requested_missing_run_is_rejected():
    with pytest.raises(
        ValueError,
        match="missing from per-sample metrics",
    ):
        aggregate_rows(
            [make_row(11)],
            expected_sample_count=1,
            run_ids=["blue_white_vase_n5_r16_ts42"],
        )


def test_missing_required_csv_column_is_rejected(tmp_path):
    input_path = tmp_path / "metrics_per_sample.csv"

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "run_id",
                "concept_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-1",
                "run_id": "dog_plush_n5_r16_ts42",
                "concept_id": "dog_plush",
            }
        )

    with pytest.raises(ValueError, match="missing columns"):
        aggregate_metrics_file(
            input_path,
            tmp_path / "metrics_aggregate.csv",
        )
