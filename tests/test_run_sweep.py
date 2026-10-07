import json
import hashlib
from pathlib import Path

import yaml

from scripts.run_sweep import classify_existing_artifact, expand_sweep_matrix


def test_ml04_matrix_expands_to_valid_12_cells():
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "configs/ml04_sweep.yaml").open(encoding="utf-8") as stream:
        matrix = yaml.safe_load(stream)

    configs = expand_sweep_matrix(matrix)

    assert len(configs) == 12
    assert {config["data"]["concept_id"] for config in configs} == {
        "cat_mug",
        "dog_plush",
        "blue_white_vase",
    }
    assert {config["data"]["subset_size"] for config in configs} == {1, 3, 5, 10}
    assert all(config["training"]["rank"] == 16 for config in configs)
    assert all(config["training"]["alpha"] == 16 for config in configs)
    assert all(config["training"]["max_train_steps"] == 500 for config in configs)


def test_pilot_output_namespace_is_separate_from_core_sweep():
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "configs/pilot/dog_plush_n10_r16_ts42.yaml").open(encoding="utf-8") as stream:
        pilot = yaml.safe_load(stream)
    with (repo_root / "configs/ml04_sweep.yaml").open(encoding="utf-8") as stream:
        matrix = yaml.safe_load(stream)
    core = next(
        config for config in expand_sweep_matrix(matrix)
        if config["run"]["id"] == pilot["run"]["id"]
    )

    assert pilot["training"]["max_train_steps"] == 1
    assert pilot["run"]["artifact_namespace"] == "pilot"
    assert Path(pilot["output"]["output_dir"]) != Path(core["output"]["output_dir"])


def test_existing_directory_requires_completed_matching_evidence(tmp_path):
    config = {
        "run": {"id": "dog_plush_n10_r16_ts42"},
        "model": {"id": "base", "revision": "rev"},
        "data": {
            "concept_id": "dog_plush", "dataset_version": "v1", "subset_size": 10,
            "manifest": str(tmp_path / "manifest.csv"),
        },
        "training": {"rank": 16, "alpha": 16, "seed": 42, "max_train_steps": 500},
    }
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    assert classify_existing_artifact(artifact, config, tmp_path)[0] == "incomplete_existing_artifact"

    (tmp_path / "manifest.csv").write_text("manifest", encoding="utf-8")
    config["run"]["artifact_namespace"] = "core"
    resolved = dict(config)
    dataset = {
        "concept_id": "dog_plush", "dataset_version": "v1", "subset_size": 10,
        "manifest": str(tmp_path / "manifest.csv"),
        "manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
    }
    provenance = {
        "run": {"id": config["run"]["id"], "artifact_namespace": "core"},
        "model": config["model"], "dataset": dataset, "training": config["training"],
    }
    (artifact / "status.json").write_text(
        json.dumps({"status": "completed", "run_id": config["run"]["id"]}), encoding="utf-8"
    )
    (artifact / "config.resolved.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    (artifact / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    assert classify_existing_artifact(artifact, config, tmp_path)[0] == "completed_existing"

    resolved["training"]["max_train_steps"] = 1
    (artifact / "config.resolved.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    assert classify_existing_artifact(artifact, config, tmp_path)[0] == "incompatible_artifact"
