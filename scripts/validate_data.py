import argparse
import hashlib
from pathlib import Path

from personalized_t2i.data.validate import validate_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset manifest.")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the dataset manifest.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    manifest = manifest.resolve()

    if not manifest.exists():
        print(f"FAIL: manifest not found: {manifest}")
        raise SystemExit(1)

    print("command: scripts/validate_data.py")
    print("manifest:", manifest)
    print("manifest sha256:", sha256_file(manifest))

    valid = validate_manifest(
        manifest_path=manifest,
        repo_root=repo_root,
    )

    if not valid:
        raise SystemExit(1)

    print("\nvalidation completed successfully")


if __name__ == "__main__":
    main()
