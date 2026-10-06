"""Preflight checks for dataset and model provenance."""

import csv
import hashlib
import re
from pathlib import Path

from personalized_t2i.config import MODEL_ID


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
MODEL_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def preflight_dataset(
    concept_id: str,
    dataset_version: str,
    manifest: str,
    manifest_sha256: str,
    concepts_registry: str | Path = "data/manifests/concepts.csv",
) -> None:
    """Verify dataset manifest existence, version, and SHA-256."""
    manifest_path = Path(manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest does not exist: {manifest}"
        )

    if not SHA256_PATTERN.fullmatch(manifest_sha256):
        raise ValueError(
            "data.manifest_sha256 must be a 64-character SHA-256 hex digest"
        )

    registry_path = Path(concepts_registry)

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Concept registry does not exist: {concepts_registry}"
        )

    with registry_path.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    matches = [
        row for row in rows
        if row.get("concept_id") == concept_id
    ]

    if not matches:
        raise ValueError(
            f"Concept is not registered: {concept_id}"
        )

    registered_version = matches[0].get("dataset_version")

    if registered_version != dataset_version:
        raise ValueError(
            "dataset_version does not match the concept registry"
        )

    actual_hash = _sha256_file(manifest_path)

    if actual_hash.lower() != manifest_sha256.lower():
        raise ValueError(
            "Dataset manifest SHA-256 does not match "
            "data.manifest_sha256"
        )


def preflight_model(
    model_id: str,
    revision: str,
) -> None:
    """Verify the model identity and immutable revision format."""
    if model_id != MODEL_ID:
        raise ValueError(
            f"model.id must be {MODEL_ID}"
        )

    if not MODEL_REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            "model.revision must be an immutable 40-character "
            "commit SHA"
        )


def preflight_config(
    config: dict,
    *,
    concepts_registry: str | Path = "data/manifests/concepts.csv",
) -> None:
    """Run dataset and model provenance checks for a real experiment."""
    data = config["data"]
    model = config["model"]

    manifest_sha256 = data.get("manifest_sha256")

    if not isinstance(manifest_sha256, str):
        raise ValueError(
            "data.manifest_sha256 is required for experiment preflight"
        )

    preflight_dataset(
        concept_id=data["concept_id"],
        dataset_version=data["dataset_version"],
        manifest=data["manifest"],
        manifest_sha256=manifest_sha256,
        concepts_registry=concepts_registry,
    )

    preflight_model(
        model_id=model["id"],
        revision=model["revision"],
    )
