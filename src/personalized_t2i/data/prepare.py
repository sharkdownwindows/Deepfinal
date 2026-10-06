import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from personalized_t2i.data.validate import validate_manifest


SELECTION_ORDER = {
    "cat_mug": [
        "img_7898",
        "img_7901",
        "img_7904",
        "img_7899",
        "img_7902",
        "img_7900",
        "img_7903",
        "img_7905",
        "img_7916",
        "img_7917",
    ],
    "dog_plush": [
        "img_7907",
        "img_7911",
        "img_7912",
        "img_7909",
        "img_7910",
        "img_7908",
        "img_7913",
        "img_7914",
        "img_7915",
        "img_7918",
    ],
    "blue_white_vase": [
        "img_7888",
        "img_7891",
        "img_7893",
        "img_7895",
        "img_7897",
        "img_7889",
        "img_7890",
        "img_7892",
        "img_7894",
        "img_7896",
    ],
}


SUBSET_SIZES = (1, 3, 5, 10)

TRANSFORM_VERSION = (
    "exif-transpose_rgb_square-required_"
    "resize-512x512-lanczos_jpeg-q95_v1"
)

PROVENANCE_FIELDS = [
    "image_id",
    "concept_id",
    "dataset_version",
    "raw_path",
    "raw_sha256",
    "processed_path",
    "processed_sha256",
    "transform_version",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def subset_membership(position: int) -> str:
    memberships = []

    for size in SUBSET_SIZES:
        if position <= size:
            memberships.append(str(size))

    return ",".join(memberships)


def read_manifest(
    manifest_path: Path,
) -> tuple[list[str], list[dict]]:
    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"{manifest_path}: missing CSV header"
            )

        return list(reader.fieldnames), list(reader)


def get_dataset_version(
    repo_root: Path,
    concept_id: str,
) -> str:
    registry_path = (
        repo_root
        / "data/manifests/concepts.csv"
    )

    with registry_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    matches = [
        row
        for row in rows
        if row.get("concept_id") == concept_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"{concept_id}: expected exactly "
            "one registry entry"
        )

    version = (
        matches[0]
        .get("dataset_version", "")
        .strip()
    )

    if not version:
        raise ValueError(
            f"{concept_id}: missing dataset_version "
            "in registry"
        )

    return version


def get_manifest_concept(
    rows: list[dict],
) -> str:
    train_rows = [
        row
        for row in rows
        if row.get("split") == "train_pool"
    ]

    if not train_rows:
        raise ValueError(
            "manifest contains no train_pool rows"
        )

    concept_id = (
        train_rows[0]
        .get("concept_id", "")
        .strip()
    )

    if concept_id not in SELECTION_ORDER:
        raise ValueError(
            f"{concept_id}: no locked selection order"
        )

    return concept_id


def expected_memberships(
    concept_id: str,
) -> dict[str, str]:
    order = SELECTION_ORDER[concept_id]

    if len(order) != 10:
        raise ValueError(
            f"{concept_id}: selection order "
            "must contain 10 IDs"
        )

    if len(set(order)) != 10:
        raise ValueError(
            f"{concept_id}: selection order "
            "contains duplicate IDs"
        )

    return {
        image_id: subset_membership(position)
        for position, image_id in enumerate(
            order,
            start=1,
        )
    }


def require_valid_manifest(
    manifest_path: Path,
    repo_root: Path,
) -> None:
    if not validate_manifest(
        manifest_path,
        repo_root,
    ):
        raise ValueError(
            f"manifest validation failed: "
            f"{manifest_path}"
        )


def assign_nested_subsets(
    manifest_path: Path,
    repo_root: Path,
) -> None:
    manifest_path = manifest_path.resolve()
    repo_root = repo_root.resolve()

    require_valid_manifest(
        manifest_path,
        repo_root,
    )

    fields, rows = read_manifest(
        manifest_path
    )

    if "subset_membership" not in fields:
        raise ValueError(
            f"{manifest_path}: missing "
            "subset_membership column"
        )

    concept_id = get_manifest_concept(
        rows
    )

    expected = expected_memberships(
        concept_id
    )

    train_rows = [
        row
        for row in rows
        if row.get("split") == "train_pool"
    ]

    train_ids = [
        row.get("image_id", "")
        for row in train_rows
    ]

    if len(train_ids) != len(set(train_ids)):
        raise ValueError(
            f"{concept_id}: duplicate image_id "
            "in train_pool"
        )

    if set(train_ids) != set(
        SELECTION_ORDER[concept_id]
    ):
        raise ValueError(
            f"{concept_id}: train_pool does not "
            "match locked selection order"
        )

    changed = False

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        image_id = row.get("image_id", "")
        split = row.get("split", "")
        membership = (
            row.get(
                "subset_membership",
                "",
            )
            .strip()
        )

        if split == "train_pool":
            expected_value = expected[
                image_id
            ]

            if membership == "":
                row["subset_membership"] = (
                    expected_value
                )
                changed = True

            elif membership != expected_value:
                raise ValueError(
                    f"row {row_number} "
                    "field subset_membership: "
                    f"{image_id} has "
                    f"'{membership}', expected "
                    f"'{expected_value}'"
                )

        elif split == "heldout":
            if membership:
                raise ValueError(
                    f"row {row_number} "
                    "field subset_membership: "
                    "heldout image must not belong "
                    "to training subsets"
                )

    if not changed:
        return

    temp_path = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    try:
        with temp_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(rows)

        temp_path.replace(
            manifest_path
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def verify_nested_subsets(
    manifest_path: Path,
) -> None:
    _, rows = read_manifest(
        manifest_path
    )

    if not rows:
        raise ValueError(
            f"{manifest_path}: manifest is empty"
        )

    concept_id = get_manifest_concept(
        rows
    )

    order = SELECTION_ORDER[
        concept_id
    ]

    expected = expected_memberships(
        concept_id
    )

    all_ids = [
        row.get("image_id", "")
        for row in rows
    ]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            f"{concept_id}: duplicate image_id "
            "in manifest"
        )

    train_rows = [
        row
        for row in rows
        if row.get("split") == "train_pool"
    ]

    train_ids = {
        row.get("image_id", "")
        for row in train_rows
    }

    if train_ids != set(order):
        raise ValueError(
            f"{concept_id}: train_pool does not "
            "match locked selection order"
        )

    allowed_memberships = {
        "1",
        "3",
        "5",
        "10",
    }

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        image_id = row.get("image_id", "")
        split = row.get("split", "")
        membership = (
            row.get(
                "subset_membership",
                "",
            )
            .strip()
        )

        if split == "heldout":
            if membership:
                raise ValueError(
                    f"row {row_number} "
                    "field subset_membership: "
                    "heldout image must not belong "
                    "to training subsets"
                )

            continue

        if split != "train_pool":
            continue

        tokens = [
            token
            for token in membership.split(",")
            if token
        ]

        if len(tokens) != len(set(tokens)):
            raise ValueError(
                f"row {row_number} "
                "field subset_membership: "
                "duplicate membership values"
            )

        if any(
            token not in allowed_memberships
            for token in tokens
        ):
            raise ValueError(
                f"row {row_number} "
                "field subset_membership: "
                f"invalid membership "
                f"'{membership}'"
            )

        expected_value = expected[
            image_id
        ]

        if membership != expected_value:
            raise ValueError(
                f"row {row_number} "
                "field subset_membership: "
                f"{image_id} has "
                f"'{membership}', expected "
                f"'{expected_value}'"
            )

    for size in SUBSET_SIZES:
        actual = {
            row["image_id"]
            for row in train_rows
            if str(size)
            in row[
                "subset_membership"
            ].split(",")
        }

        expected_prefix = set(
            order[:size]
        )

        if actual != expected_prefix:
            raise ValueError(
                f"{concept_id}: D{size} "
                "does not match locked "
                "selection prefix"
            )

    print(
        f"{concept_id}: "
        "D1/D3/D5/D10 match locked "
        "selection order PASS"
    )


def preprocess_image(
    input_path: Path,
    output_path: Path,
) -> None:
    with Image.open(
        input_path
    ) as image:
        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert(
            "RGB"
        )

        if image.width != image.height:
            raise ValueError(
                "source image is not square: "
                f"{input_path} {image.size}"
            )

        image = image.resize(
            (512, 512),
            Image.Resampling.LANCZOS,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95,
        )


def build_processed_candidate(
    manifest_path: Path,
    repo_root: Path,
    staging_root: Path,
) -> tuple[str, str, list[dict]]:
    _, rows = read_manifest(
        manifest_path
    )

    concept_id = get_manifest_concept(
        rows
    )

    dataset_version = get_dataset_version(
        repo_root,
        concept_id,
    )

    train_rows = [
        row
        for row in rows
        if row.get("split") == "train_pool"
    ]

    if len(train_rows) != 10:
        raise ValueError(
            f"{concept_id}: expected "
            "10 train_pool rows, found "
            f"{len(train_rows)}"
        )

    records = []

    for row_number, row in enumerate(
        train_rows,
        start=2,
    ):
        relative_path = Path(
            row["file_path"]
        )

        if relative_path.is_absolute():
            raise ValueError(
                f"row {row_number} "
                "field file_path: absolute "
                "paths are not allowed"
            )

        input_path = (
            repo_root
            / relative_path
        ).resolve()

        try:
            input_path.relative_to(
                repo_root
            )

        except ValueError as exc:
            raise ValueError(
                f"row {row_number} "
                "field file_path: path "
                "escapes repository root"
            ) from exc

        if not input_path.exists():
            raise ValueError(
                f"row {row_number} "
                "field file_path: "
                f"missing {input_path}"
            )

        expected_raw_sha = (
            row["sha256"].strip()
        )

        actual_raw_sha = sha256_file(
            input_path
        )

        if (
            actual_raw_sha
            != expected_raw_sha
        ):
            raise ValueError(
                f"row {row_number} "
                "field sha256: raw hash "
                "mismatch for "
                f"{row['image_id']}"
            )

        output_name = (
            relative_path
            .with_suffix(".jpeg")
            .name
        )

        staged_path = (
            staging_root
            / "train_pool"
            / output_name
        )

        preprocess_image(
            input_path,
            staged_path,
        )

        processed_sha = sha256_file(
            staged_path
        )

        records.append(
            {
                "image_id": row[
                    "image_id"
                ],
                "concept_id": concept_id,
                "dataset_version": (
                    dataset_version
                ),
                "raw_path": (
                    relative_path
                    .as_posix()
                ),
                "raw_sha256": (
                    actual_raw_sha
                ),
                "processed_path": (
                    "train_pool/"
                    f"{output_name}"
                ),
                "processed_sha256": (
                    processed_sha
                ),
                "transform_version": (
                    TRANSFORM_VERSION
                ),
            }
        )

    return (
        concept_id,
        dataset_version,
        records,
    )


def provenance_path(
    repo_root: Path,
    concept_id: str,
    dataset_version: str,
) -> Path:
    return (
        repo_root
        / "artifacts"
        / "data03_validation"
        / (
            f"{concept_id}_"
            f"{dataset_version}_"
            "preprocessing_manifest.csv"
        )
    )


def serialize_provenance(
    records: list[dict],
) -> str:
    from io import StringIO

    buffer = StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=PROVENANCE_FIELDS,
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(records)

    return buffer.getvalue()


def write_or_verify_provenance(
    path: Path,
    records: list[dict],
) -> None:
    content = serialize_provenance(
        records
    )

    if path.exists():
        existing = path.read_text(
            encoding="utf-8"
        )

        if existing != content:
            raise ValueError(
                "existing provenance differs: "
                f"{path}"
            )

        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        temp_path.write_text(
            content,
            encoding="utf-8",
        )

        temp_path.replace(
            path
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def version_matches_candidate(
    version_dir: Path,
    candidate_dir: Path,
    records: list[dict],
) -> bool:
    existing_train = (
        version_dir
        / "train_pool"
    )

    candidate_train = (
        candidate_dir
        / "train_pool"
    )

    if not existing_train.exists():
        return False

    existing_files = sorted(
        path.name
        for path
        in existing_train.glob(
            "*.jpeg"
        )
    )

    candidate_files = sorted(
        path.name
        for path
        in candidate_train.glob(
            "*.jpeg"
        )
    )

    if existing_files != candidate_files:
        return False

    for record in records:
        relative_processed = Path(
            record["processed_path"]
        )

        existing_path = (
            version_dir
            / relative_processed
        )

        candidate_path = (
            candidate_dir
            / relative_processed
        )

        if not existing_path.exists():
            return False

        if (
            sha256_file(existing_path)
            != sha256_file(candidate_path)
        ):
            return False

    return True


def process_manifest(
    manifest_path: Path,
    repo_root: Path,
) -> int:
    manifest_path = manifest_path.resolve()
    repo_root = repo_root.resolve()

    require_valid_manifest(
        manifest_path,
        repo_root,
    )

    verify_nested_subsets(
        manifest_path
    )

    _, rows = read_manifest(
        manifest_path
    )

    concept_id = get_manifest_concept(
        rows
    )

    dataset_version = get_dataset_version(
        repo_root,
        concept_id,
    )

    version_dir = (
        repo_root
        / "data"
        / "processed"
        / concept_id
        / dataset_version
    )

    staging_dir = (
        repo_root
        / "data"
        / "processed"
        / concept_id
        / (
            f".{dataset_version}"
            "_staging"
        )
    )

    if staging_dir.exists():
        shutil.rmtree(
            staging_dir
        )

    staging_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        (
            _,
            _,
            records,
        ) = build_processed_candidate(
            manifest_path,
            repo_root,
            staging_dir,
        )

        if version_dir.exists():
            if not version_matches_candidate(
                version_dir,
                staging_dir,
                records,
            ):
                raise ValueError(
                    f"{concept_id} "
                    f"{dataset_version}: "
                    "processed dataset already "
                    "exists with different bytes; "
                    "create a new dataset version"
                )

            write_or_verify_provenance(
                provenance_path(
                    repo_root,
                    concept_id,
                    dataset_version,
                ),
                records,
            )

            print(
                f"{concept_id} "
                f"{dataset_version}: "
                "existing processed dataset "
                "verified, no-op"
            )

            return len(records)

        staging_dir.replace(
            version_dir
        )

        write_or_verify_provenance(
            provenance_path(
                repo_root,
                concept_id,
                dataset_version,
            ),
            records,
        )

        print(
            f"{concept_id} "
            f"{dataset_version}: "
            f"processed {len(records)} images"
        )

        return len(records)

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )


def verify_processed(
    repo_root: Path,
) -> None:
    repo_root = repo_root.resolve()
    total = 0

    for concept_id in SELECTION_ORDER:
        dataset_version = (
            get_dataset_version(
                repo_root,
                concept_id,
            )
        )

        version_dir = (
            repo_root
            / "data"
            / "processed"
            / concept_id
            / dataset_version
        )

        folder = (
            version_dir
            / "train_pool"
        )

        mapping_path = provenance_path(
            repo_root,
            concept_id,
            dataset_version,
        )

        files = sorted(
            folder.glob("*.jpeg")
        )

        if len(files) != 10:
            raise ValueError(
                f"{concept_id}: expected "
                "10 processed images, found "
                f"{len(files)}"
            )

        if not mapping_path.exists():
            raise ValueError(
                f"{concept_id}: missing "
                "preprocessing provenance "
                f"{mapping_path}"
            )

        with mapping_path.open(
            newline="",
            encoding="utf-8",
        ) as f:
            records = list(
                csv.DictReader(f)
            )

        if len(records) != 10:
            raise ValueError(
                f"{concept_id}: expected "
                "10 provenance records, found "
                f"{len(records)}"
            )

        expected_ids = set(
            SELECTION_ORDER[
                concept_id
            ]
        )

        record_ids = {
            record["image_id"]
            for record in records
        }

        if record_ids != expected_ids:
            raise ValueError(
                f"{concept_id}: provenance "
                "image IDs do not match "
                "locked selection"
            )

        for record in records:
            if (
                record["dataset_version"]
                != dataset_version
            ):
                raise ValueError(
                    f"{concept_id}: invalid "
                    "dataset version in provenance"
                )

            if (
                record["transform_version"]
                != TRANSFORM_VERSION
            ):
                raise ValueError(
                    f"{concept_id}: invalid "
                    "transform version "
                    "in provenance"
                )

            raw_path = (
                repo_root
                / record["raw_path"]
            )

            if not raw_path.exists():
                raise ValueError(
                    f"missing raw image: "
                    f"{raw_path}"
                )

            if (
                sha256_file(raw_path)
                != record["raw_sha256"]
            ):
                raise ValueError(
                    f"raw hash mismatch: "
                    f"{raw_path}"
                )

            processed_path = (
                version_dir
                / record[
                    "processed_path"
                ]
            )

            if not processed_path.exists():
                raise ValueError(
                    "missing processed image: "
                    f"{processed_path}"
                )

            if (
                sha256_file(
                    processed_path
                )
                != record[
                    "processed_sha256"
                ]
            ):
                raise ValueError(
                    "processed hash mismatch: "
                    f"{processed_path}"
                )

            with Image.open(
                processed_path
            ) as image:
                if image.size != (
                    512,
                    512,
                ):
                    raise ValueError(
                        "invalid size: "
                        f"{processed_path} "
                        f"{image.size}"
                    )

                if image.mode != "RGB":
                    raise ValueError(
                        "invalid mode: "
                        f"{processed_path} "
                        f"{image.mode}"
                    )

        total += len(files)

        print(
            f"{concept_id}: "
            "10/10 raw hashes, processed "
            "hashes and transforms PASS"
        )

    print(
        f"processed total: {total}/30"
    )

    print(
        "512x512 RGB: PASS"
    )


def main() -> None:
    repo_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    manifests = [
        repo_root
        / "data/manifests/"
        "cat_mug_v1.csv",
        repo_root
        / "data/manifests/"
        "dog_plush_v1.csv",
        repo_root
        / "data/manifests/"
        "blue_white_vase_v1.csv",
    ]

    for manifest in manifests:
        require_valid_manifest(
            manifest,
            repo_root,
        )

    for manifest in manifests:
        assign_nested_subsets(
            manifest,
            repo_root,
        )

        verify_nested_subsets(
            manifest
        )

    total = 0

    for manifest in manifests:
        total += process_manifest(
            manifest,
            repo_root,
        )

    print(
        f"preprocessed: {total} images"
    )

    verify_processed(
        repo_root
    )


if __name__ == "__main__":
    main()