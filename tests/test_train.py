import hashlib

import pytest
import yaml

from personalized_t2i.training.train import prepare_run
from tests.test_config import valid_config


def write_test_config(tmp_path):
    config = valid_config()

    manifest = tmp_path / "toy01_v1.csv"
    manifest.write_text(
        "image_id,file_path,sha256,concept_id,split\n"
        "img01,data/img01.png,abc,toy01,train\n",
        encoding="utf-8",
    )

    config["data"]["manifest"] = str(manifest)
    config["data"]["manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    config["model"]["revision"] = (
        "0123456789abcdef0123456789abcdef01234567"
    )

    registry = tmp_path / "concepts.csv"
    registry.write_text(
        "concept_id,class_noun,unique_token,dataset_version,owner,consent_or_license\n"
        "toy01,toy,toktoy,v1,test,test\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    return config_path, registry


def test_prepare_run_creates_artifacts(tmp_path):
    config_path, registry = write_test_config(tmp_path)

    run_dir = prepare_run(
        config_path=config_path,
        environment={"test": True},
        artifacts_root=tmp_path / "artifacts",
        concepts_registry=registry,
    )

    assert run_dir.exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "status.json").exists()


def test_prepare_run_rejects_duplicate_run(tmp_path):
    config_path, registry = write_test_config(tmp_path)
    artifacts_root = tmp_path / "artifacts"

    prepare_run(
        config_path=config_path,
        environment={"test": True},
        artifacts_root=artifacts_root,
        concepts_registry=registry,
    )

    with pytest.raises(FileExistsError):
        prepare_run(
            config_path=config_path,
            environment={"test": True},
            artifacts_root=artifacts_root,
            concepts_registry=registry,
        )
