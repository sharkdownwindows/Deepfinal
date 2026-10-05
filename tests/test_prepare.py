import csv
from pathlib import Path

import pytest
from PIL import Image

from personalized_t2i.data.prepare import (
    SELECTION_ORDER,
    TRANSFORM_VERSION,
    assign_nested_subsets,
    process_manifest,
    provenance_path,
    sha256_file,
    verify_nested_subsets,
)


COLUMNS = [
    "image_id",
    "file_path",
    "sha256",
    "concept_id",
    "split",
    "subset_membership",
    "caption",
    "source",
    "consent_or_license",
    "notes",
]


def make_image(
    path: Path,
    value: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.new(
        "RGB",
        (32, 32),
        (value, value, value),
    ).save(path)


def write_manifest(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=COLUMNS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def read_rows(
    path: Path,
) -> list[dict]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def make_repo(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    list[dict],
]:
    repo = tmp_path / "repo"

    manifests_dir = (
        repo
        / "data/manifests"
    )

    manifests_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_path = (
        manifests_dir
        / "concepts.csv"
    )

    with registry_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concept_id",
                "class_noun",
                "unique_token",
                "dataset_version",
                "owner",
                "consent_or_license",
            ],
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerow(
            {
                "concept_id": "cat_mug",
                "class_noun": "mug",
                "unique_token": "zzobj01",
                "dataset_version": "v1",
                "owner": "project_team",
                "consent_or_license": (
                    "self-captured"
                ),
            }
        )

    rows = []

    for index, image_id in enumerate(
        SELECTION_ORDER["cat_mug"],
        start=1,
    ):
        path = (
            repo
            / "data/raw/cat_mug"
            / f"{image_id}.png"
        )

        make_image(
            path,
            index * 10,
        )

        rows.append(
            {
                "image_id": image_id,
                "file_path": (
                    "data/raw/cat_mug/"
                    f"{image_id}.png"
                ),
                "sha256": (
                    sha256_file(path)
                ),
                "concept_id": "cat_mug",
                "split": "train_pool",
                "subset_membership": "",
                "caption": (
                    "a photo of zzobj01 mug"
                ),
                "source": "self-captured",
                "consent_or_license": (
                    "self-captured"
                ),
                "notes": "",
            }
        )

    for index in range(3):
        image_id = (
            f"heldout_{index + 1}"
        )

        path = (
            repo
            / "data/eval_refs/cat_mug"
            / f"{image_id}.png"
        )

        make_image(
            path,
            150 + index,
        )

        rows.append(
            {
                "image_id": image_id,
                "file_path": (
                    "data/eval_refs/"
                    "cat_mug/"
                    f"{image_id}.png"
                ),
                "sha256": (
                    sha256_file(path)
                ),
                "concept_id": "cat_mug",
                "split": "heldout",
                "subset_membership": "",
                "caption": "",
                "source": "self-captured",
                "consent_or_license": (
                    "self-captured"
                ),
                "notes": "",
            }
        )

    manifest = (
        manifests_dir
        / "cat_mug_v1.csv"
    )

    write_manifest(
        manifest,
        rows,
    )

    return (
        repo,
        manifest,
        rows,
    )


def version_hashes(
    version_dir: Path,
) -> dict[str, str]:
    return {
        path.relative_to(
            version_dir
        ).as_posix(): sha256_file(
            path
        )
        for path
        in version_dir.rglob("*")
        if path.is_file()
    }


def test_locked_prefix_assignment_and_verify(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    verify_nested_subsets(
        manifest
    )


def test_different_nested_selection_fails(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    rows = read_rows(
        manifest
    )

    first = rows[0][
        "subset_membership"
    ]

    second = rows[1][
        "subset_membership"
    ]

    rows[0][
        "subset_membership"
    ] = second

    rows[1][
        "subset_membership"
    ] = first

    write_manifest(
        manifest,
        rows,
    )

    with pytest.raises(
        ValueError
    ):
        verify_nested_subsets(
            manifest
        )


def test_duplicate_image_id_fails(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    rows = read_rows(
        manifest
    )

    rows[1]["image_id"] = (
        rows[0]["image_id"]
    )

    write_manifest(
        manifest,
        rows,
    )

    with pytest.raises(
        ValueError
    ):
        verify_nested_subsets(
            manifest
        )


def test_invalid_membership_fails(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    rows = read_rows(
        manifest
    )

    rows[0][
        "subset_membership"
    ] = "1,2,3,5,10"

    write_manifest(
        manifest,
        rows,
    )

    with pytest.raises(
        ValueError
    ):
        verify_nested_subsets(
            manifest
        )


def test_heldout_membership_fails(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    rows = read_rows(
        manifest
    )

    rows[-1][
        "subset_membership"
    ] = "10"

    write_manifest(
        manifest,
        rows,
    )

    with pytest.raises(
        ValueError
    ):
        verify_nested_subsets(
            manifest
        )


def test_raw_hash_mismatch_keeps_existing_output(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    process_manifest(
        manifest,
        repo,
    )

    version_dir = (
        repo
        / "data/processed/"
        "cat_mug/v1"
    )

    before = version_hashes(
        version_dir
    )

    mapping = provenance_path(
        repo,
        "cat_mug",
        "v1",
    )

    mapping_before = (
        sha256_file(mapping)
    )

    raw_path = (
        repo
        / "data/raw/cat_mug/"
        "img_7898.png"
    )

    make_image(
        raw_path,
        250,
    )

    with pytest.raises(
        ValueError
    ):
        process_manifest(
            manifest,
            repo,
        )

    after = version_hashes(
        version_dir
    )

    assert after == before

    assert (
        sha256_file(mapping)
        == mapping_before
    )


def test_existing_version_with_different_bytes_fails_without_rewrite(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    process_manifest(
        manifest,
        repo,
    )

    version_dir = (
        repo
        / "data/processed/"
        "cat_mug/v1"
    )

    processed_path = (
        version_dir
        / "train_pool/"
        "img_7898.jpeg"
    )

    make_image(
        processed_path,
        240,
    )

    tampered_hash = sha256_file(
        processed_path
    )

    with pytest.raises(
        ValueError
    ):
        process_manifest(
            manifest,
            repo,
        )

    assert (
        sha256_file(
            processed_path
        )
        == tampered_hash
    )


def test_rerun_is_noop(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    process_manifest(
        manifest,
        repo,
    )

    version_dir = (
        repo
        / "data/processed/"
        "cat_mug/v1"
    )

    before = version_hashes(
        version_dir
    )

    mapping = provenance_path(
        repo,
        "cat_mug",
        "v1",
    )

    mapping_before = (
        sha256_file(mapping)
    )

    process_manifest(
        manifest,
        repo,
    )

    after = version_hashes(
        version_dir
    )

    assert after == before

    assert (
        sha256_file(mapping)
        == mapping_before
    )


def test_provenance_contains_hashes_and_transform_version(
    tmp_path,
):
    repo, manifest, _ = make_repo(
        tmp_path
    )

    assign_nested_subsets(
        manifest,
        repo,
    )

    process_manifest(
        manifest,
        repo,
    )

    mapping = provenance_path(
        repo,
        "cat_mug",
        "v1",
    )

    assert mapping.exists()

    with mapping.open(
        newline="",
        encoding="utf-8",
    ) as f:
        records = list(
            csv.DictReader(f)
        )

    assert len(records) == 10

    for record in records:
        assert record[
            "dataset_version"
        ] == "v1"

        assert record[
            "raw_sha256"
        ]

        assert record[
            "processed_sha256"
        ]

        assert record[
            "transform_version"
        ] == TRANSFORM_VERSION