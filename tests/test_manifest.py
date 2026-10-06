import csv
import hashlib
import subprocess
import sys
from pathlib import Path

from PIL import Image

from personalized_t2i.data.validate import validate_manifest


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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_image(path: Path, value: int) -> None:
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def make_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    images = repo / "data" / "raw" / "cat_mug"
    manifests = repo / "data" / "manifests"

    images.mkdir(parents=True)
    manifests.mkdir(parents=True)

    registry = manifests / "concepts.csv"
    registry.write_text(
        "concept_id,class_noun,unique_token,dataset_version,owner,"
        "consent_or_license\n"
        "cat_mug,mug,zzobj01,v1,project_team,self-captured\n",
        encoding="utf-8",
    )

    manifest = manifests / "cat_mug_v1.csv"
    return repo, images, manifest


def valid_rows(repo: Path, images: Path) -> list[dict[str, str]]:
    rows = []

    for index in range(13):
        path = images / f"image_{index}.png"
        make_image(path, index)

        split = "train_pool" if index < 10 else "heldout"

        rows.append(
            {
                "image_id": f"cat_{index}",
                "file_path": str(path.relative_to(repo)),
                "sha256": file_hash(path),
                "concept_id": "cat_mug",
                "split": split,
                "subset_membership": "",
                "caption": (
                    "a photo of zzobj01 mug"
                    if split == "train_pool"
                    else ""
                ),
                "source": "self-captured",
                "consent_or_license": "self-captured",
                "notes": "",
            }
        )

    return rows


def write_manifest(
    manifest: Path,
    rows: list[dict[str, str]],
    columns: list[str] | None = None,
) -> None:
    fieldnames = columns or COLUMNS

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_valid_manifest(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    write_manifest(manifest, valid_rows(repo, images))

    assert validate_manifest(manifest, repo) is True


def test_duplicate_image_id_fails(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    rows = valid_rows(repo, images)
    rows[1]["image_id"] = rows[0]["image_id"]
    write_manifest(manifest, rows)

    assert validate_manifest(manifest, repo) is False


def test_mixed_concept_fails(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    rows = valid_rows(repo, images)
    rows[1]["concept_id"] = "wrong_concept"
    write_manifest(manifest, rows)

    assert validate_manifest(manifest, repo) is False


def test_wrong_caption_fails(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    rows = valid_rows(repo, images)
    rows[0]["caption"] = "a completely unrelated caption"
    write_manifest(manifest, rows)

    assert validate_manifest(manifest, repo) is False


def test_missing_required_column_fails(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    rows = valid_rows(repo, images)
    columns = [column for column in COLUMNS if column != "image_id"]
    write_manifest(manifest, rows, columns)

    assert validate_manifest(manifest, repo) is False


def test_absolute_file_path_fails(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    rows = valid_rows(repo, images)
    rows[0]["file_path"] = str((images / "image_0.png").resolve())
    write_manifest(manifest, rows)

    assert validate_manifest(manifest, repo) is False


def test_path_outside_repo_fails(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    rows = valid_rows(repo, images)

    outside = tmp_path / "outside.png"
    make_image(outside, 100)

    rows[0]["file_path"] = "../../outside.png"
    rows[0]["sha256"] = file_hash(outside)
    write_manifest(manifest, rows)

    assert validate_manifest(manifest, repo) is False


def test_validation_does_not_depend_on_cwd(tmp_path, monkeypatch):
    repo, images, manifest = make_repo(tmp_path)
    write_manifest(manifest, valid_rows(repo, images))

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    assert validate_manifest(manifest, repo) is True


def test_cli_returns_nonzero_for_invalid_manifest(tmp_path):
    repo, images, manifest = make_repo(tmp_path)
    rows = valid_rows(repo, images)
    rows[0]["caption"] = "wrong caption"
    write_manifest(manifest, rows)

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "validate_data.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--repo-root",
            str(repo),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
