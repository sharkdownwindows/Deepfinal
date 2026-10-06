import csv
import hashlib
from pathlib import Path

from PIL import Image


REQUIRED_COLUMNS = {
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
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_concepts(registry_path: Path) -> dict[str, dict[str, str]]:
    with registry_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["concept_id"]: row for row in reader}


def validate_manifest(
    manifest_path: Path,
    repo_root: Path,
    registry_path: Path | None = None,
) -> bool:
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()

    if registry_path is None:
        registry_path = repo_root / "data/manifests/concepts.csv"
    else:
        registry_path = registry_path.resolve()

    print(f"\nchecking: {manifest_path}")

    try:
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])

            missing_columns = REQUIRED_COLUMNS - fieldnames
            if missing_columns:
                print(
                    "FAIL: missing required columns:",
                    ", ".join(sorted(missing_columns)),
                )
                return False

            rows = list(reader)
    except (OSError, csv.Error) as exc:
        print(f"FAIL: cannot read manifest: {exc}")
        return False

    try:
        concepts = load_concepts(registry_path)
    except (OSError, csv.Error, KeyError) as exc:
        print(f"FAIL: cannot read concept registry: {exc}")
        return False

    if not rows:
        print("FAIL: manifest is empty")
        return False

    manifest_concepts = {row["concept_id"] for row in rows}

    if len(manifest_concepts) != 1:
        print(
            "FAIL: manifest contains multiple concept_id values:",
            ", ".join(sorted(manifest_concepts)),
        )
        return False

    concept_id = next(iter(manifest_concepts))

    if concept_id not in concepts:
        print(f"FAIL: unknown concept_id: {concept_id}")
        return False

    concept = concepts[concept_id]
    expected_caption = (
        f'a photo of {concept["unique_token"]} {concept["class_noun"]}'
    )

    train = [row for row in rows if row["split"] == "train_pool"]
    heldout = [row for row in rows if row["split"] == "heldout"]

    print("train:", len(train))
    print("heldout:", len(heldout))

    if len(train) != 10:
        print("FAIL: expected 10 train-pool images")
        return False

    if len(heldout) != 3:
        print("FAIL: expected 3 held-out images")
        return False

    seen_ids = set()
    train_hashes = set()
    heldout_hashes = set()

    for row_number, row in enumerate(rows, start=2):
        image_id = row["image_id"]

        if not image_id:
            print(f"FAIL: row {row_number} field image_id: missing value")
            return False

        if image_id in seen_ids:
            print(
                f"FAIL: row {row_number} field image_id: "
                f"duplicate value {image_id}"
            )
            return False
        seen_ids.add(image_id)

        if row["concept_id"] != concept_id:
            print(
                f"FAIL: row {row_number} field concept_id: "
                f"expected {concept_id}, got {row['concept_id']}"
            )
            return False

        if row["split"] not in {"train_pool", "heldout"}:
            print(
                f"FAIL: row {row_number} field split: "
                f"invalid value {row['split']}"
            )
            return False

        if not row["sha256"]:
            print(f"FAIL: row {row_number} field sha256: missing value")
            return False

        if not row["source"]:
            print(f"FAIL: row {row_number} field source: missing value")
            return False

        if not row["consent_or_license"]:
            print(
                f"FAIL: row {row_number} field consent_or_license: "
                "missing value"
            )
            return False

        if row["split"] == "train_pool" and row["caption"] != expected_caption:
            print(
                f"FAIL: row {row_number} field caption: "
                f"expected {expected_caption!r}, got {row['caption']!r}"
            )
            return False

        file_path = Path(row["file_path"])

        if file_path.is_absolute():
            print(
                f"FAIL: row {row_number} field file_path: "
                "absolute paths are not allowed"
            )
            return False

        resolved_path = (repo_root / file_path).resolve()

        try:
            resolved_path.relative_to(repo_root)
        except ValueError:
            print(
                f"FAIL: row {row_number} field file_path: "
                f"path escapes repository root: {file_path}"
            )
            return False

        if not resolved_path.exists():
            print(
                f"FAIL: row {row_number} field file_path: "
                f"missing file {file_path}"
            )
            return False

        try:
            with Image.open(resolved_path) as image:
                image.verify()
        except Exception as exc:
            print(
                f"FAIL: row {row_number} field file_path: "
                f"unreadable image {file_path}: {exc}"
            )
            return False

        actual_hash = sha256_file(resolved_path)

        if actual_hash != row["sha256"]:
            print(
                f"FAIL: row {row_number} field sha256: "
                f"hash mismatch for {file_path}"
            )
            return False

        if row["split"] == "train_pool":
            if actual_hash in train_hashes:
                print(
                    f"FAIL: row {row_number}: duplicate image bytes "
                    f"in train pool: {file_path}"
                )
                return False
            train_hashes.add(actual_hash)
        else:
            if actual_hash in heldout_hashes:
                print(
                    f"FAIL: row {row_number}: duplicate image bytes "
                    f"in held-out set: {file_path}"
                )
                return False
            heldout_hashes.add(actual_hash)

    if train_hashes & heldout_hashes:
        print("FAIL: held-out image also appears in train pool")
        return False

    print("schema: PASS")
    print("image_id uniqueness: PASS")
    print("concept identity: PASS")
    print("image readability: PASS")
    print("within-split duplicates: PASS")
    print("hash: PASS")
    print("split: PASS")
    print("source/license: PASS")
    print("caption: PASS")
    print("path safety: PASS")
    print("train/heldout overlap: PASS")
    print("PASS")
    return True
