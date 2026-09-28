import csv
import hashlib
from pathlib import Path


MANIFESTS = [
    Path("data/manifests/cat_mug_v1.csv"),
    Path("data/manifests/dog_plush_v1.csv"),
    Path("data/manifests/blue_white_vase_v1.csv"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_manifest(manifest_path: Path) -> bool:
    print(f"\nchecking: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

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

    train_hashes = set()
    heldout_hashes = set()

    for row in rows:
        if row["split"] not in {"train_pool", "heldout"}:
            print("FAIL: invalid split:", row["image_id"])
            return False

        if not row["sha256"]:
            print("FAIL: missing hash:", row["image_id"])
            return False

        if not row["source"] or not row["consent_or_license"]:
            print("FAIL: missing source/license:", row["image_id"])
            return False

        if row["split"] == "train_pool" and not row["caption"]:
            print("FAIL: missing caption:", row["image_id"])
            return False

        path = Path(row["file_path"])

        if not path.exists():
            print("FAIL: missing file:", path)
            return False

        actual_hash = sha256_file(path)

        if actual_hash != row["sha256"]:
            print("FAIL: hash mismatch:", path)
            return False

        if row["split"] == "train_pool":
            train_hashes.add(actual_hash)
        else:
            heldout_hashes.add(actual_hash)

    if train_hashes & heldout_hashes:
        print("FAIL: held-out image also appears in train pool")
        return False

    print("hash: PASS")
    print("split: PASS")
    print("source/license: PASS")
    print("caption: PASS")
    print("train/heldout overlap: PASS")
    print("PASS")
    return True


def main() -> None:
    results = [validate_manifest(path) for path in MANIFESTS]

    if not all(results):
        raise SystemExit(1)

    print("\nall DATA-02 checks passed")


if __name__ == "__main__":
    main()
