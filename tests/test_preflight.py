import hashlib

import pytest

from personalized_t2i.preflight import (
    preflight_dataset,
    preflight_model,
)


def test_preflight_dataset_accepts_matching_manifest(tmp_path):
    manifest = tmp_path / "toy01_v1.csv"
    manifest.write_text(
        "image_id,file_path,sha256,concept_id,split\n"
        "img01,data/img01.png,abc,toy01,train\n",
        encoding="utf-8",
    )

    manifest_hash = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    registry = tmp_path / "concepts.csv"
    registry.write_text(
        "concept_id,class_noun,unique_token,dataset_version,owner,consent_or_license\n"
        "toy01,toy,toktoy,v1,test,test\n",
        encoding="utf-8",
    )

    preflight_dataset(
        concept_id="toy01",
        dataset_version="v1",
        manifest=str(manifest),
        manifest_sha256=manifest_hash,
        concepts_registry=registry,
    )


def test_preflight_dataset_rejects_missing_manifest(tmp_path):
    registry = tmp_path / "concepts.csv"
    registry.write_text(
        "concept_id,class_noun,unique_token,dataset_version,owner,consent_or_license\n"
        "toy01,toy,toktoy,v1,test,test\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        preflight_dataset(
            concept_id="toy01",
            dataset_version="v1",
            manifest=str(tmp_path / "missing.csv"),
            manifest_sha256="0" * 64,
            concepts_registry=registry,
        )


def test_preflight_dataset_rejects_wrong_hash(tmp_path):
    manifest = tmp_path / "toy01_v1.csv"
    manifest.write_text("test\n", encoding="utf-8")

    registry = tmp_path / "concepts.csv"
    registry.write_text(
        "concept_id,class_noun,unique_token,dataset_version,owner,consent_or_license\n"
        "toy01,toy,toktoy,v1,test,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        preflight_dataset(
            concept_id="toy01",
            dataset_version="v1",
            manifest=str(manifest),
            manifest_sha256="0" * 64,
            concepts_registry=registry,
        )


def test_preflight_dataset_rejects_wrong_version(tmp_path):
    manifest = tmp_path / "toy01_v1.csv"
    manifest.write_text("test\n", encoding="utf-8")

    manifest_hash = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    registry = tmp_path / "concepts.csv"
    registry.write_text(
        "concept_id,class_noun,unique_token,dataset_version,owner,consent_or_license\n"
        "toy01,toy,toktoy,v1,test,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset_version"):
        preflight_dataset(
            concept_id="toy01",
            dataset_version="v2",
            manifest=str(manifest),
            manifest_sha256=manifest_hash,
            concepts_registry=registry,
        )


def test_preflight_model_accepts_immutable_sha():
    preflight_model(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "0123456789abcdef0123456789abcdef01234567",
    )


def test_preflight_model_rejects_floating_revision():
    with pytest.raises(ValueError):
        preflight_model(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "main",
        )
