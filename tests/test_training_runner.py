import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from personalized_t2i.training.runner import (
    build_trainer_command,
    materialize_training_subset,
    resolve_training_subset,
    validate_training_config,
    write_pilot_metadata,
    _git_revision,
    _gpu_snapshot,
)


def valid_training_config(train_dir: Path, output_dir: Path, manifest: Path) -> dict:
    return {
        "run": {"id": "dog_plush_n1_r16_ts42"},
        "model": {
            "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        },
        "data": {
            "concept_id": "dog_plush",
            "subset_size": 1,
            "train_data_dir": str(train_dir),
            "manifest": str(manifest),
            "instance_prompt": "a photo of zzobj02 plush toy",
        },
        "training": {
            "rank": 16,
            "alpha": 16,
            "resolution": 512,
            "learning_rate": 1e-4,
            "max_train_steps": 1,
            "checkpointing_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "scheduler": "constant",
            "warmup_steps": 0,
            "mixed_precision": "fp16",
            "gradient_checkpointing": True,
            "seed": 42,
        },
        "inference": {
            "prompt": "a photo of zzobj02 plush toy",
            "seed": 42,
            "num_inference_steps": 4,
        },
        "output": {"output_dir": str(output_dir)},
    }


@pytest.fixture
def pilot_config(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    image_path = train_dir / "one.jpg"
    image_path.write_bytes(b"image")
    manifest = tmp_path / "manifest.csv"
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest.write_text(
        "image_id,file_path,sha256,concept_id,split,subset_membership\n"
        f"img1,train/one.jpg,{digest},dog_plush,train_pool,1\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts" / "dog_plush_n1_r16_ts42"
    return valid_training_config(train_dir, output_dir, manifest), train_dir, output_dir


def test_training_config_accepts_core_values(pilot_config, tmp_path):
    config, _train_dir, output_dir = pilot_config

    run_id, resolved_output = validate_training_config(config, tmp_path)

    assert run_id == "dog_plush_n1_r16_ts42"
    assert resolved_output == output_dir


def test_training_subset_uses_manifest_membership_and_hashes(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    rows = []
    for index, membership in enumerate(("1,3", "3", "3"), start=1):
        image = train_dir / f"{index}.jpg"
        image.write_bytes(f"image-{index}".encode())
        rows.append(
            f"img{index},train/{index}.jpg,{hashlib.sha256(image.read_bytes()).hexdigest()},"
            f"dog_plush,train_pool,{membership}"
        )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_id,file_path,sha256,concept_id,split,subset_membership\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    config = valid_training_config(
        train_dir,
        tmp_path / "artifacts" / "dog_plush_n1_r16_ts42",
        manifest,
    )

    selected = resolve_training_subset(config, tmp_path)
    run_dir = tmp_path / "artifacts" / "dog_plush_n1_r16_ts42"
    run_dir.mkdir(parents=True)
    materialized = materialize_training_subset(config, tmp_path, run_dir)

    assert [path.name for path in selected] == ["1.jpg"]
    assert len(list(materialized.iterdir())) == 1
    provenance = json.loads((run_dir / "training_subset.json").read_text())
    assert len(provenance) == 1
    assert provenance[0]["sha256"] == hashlib.sha256(b"image-1").hexdigest()


def test_pilot_metadata_uses_evaluation_contract(pilot_config, tmp_path):
    config, _train_dir, _output_dir = pilot_config
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    image = run_dir / "pilot.png"
    image.write_bytes(b"image")

    metadata_path = write_pilot_metadata(run_dir, config, image, (512, 512))
    record = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert record["run_id"] == config["run"]["id"]
    assert record["generation_mode"] == "adapter"
    assert record["generation_seed"] == 42
    assert record["image_path"] == str(image.resolve())


def test_pilot_metadata_does_not_overwrite_existing_evidence(pilot_config, tmp_path):
    config, _train_dir, _output_dir = pilot_config
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = run_dir / "pilot_metadata.jsonl"
    metadata.write_text('{"existing": true}\n', encoding="utf-8")
    image = run_dir / "pilot.png"

    with pytest.raises(FileExistsError):
        write_pilot_metadata(run_dir, config, image, (512, 512))
    assert metadata.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_gpu_snapshot_reads_nvidia_smi_and_returns_unknown_on_failure(monkeypatch):
    monkeypatch.setattr("personalized_t2i.training.runner.shutil.which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(
        "personalized_t2i.training.runner.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "123, 4096\n", ""),
    )
    assert _gpu_snapshot() == (123, 4096)

    monkeypatch.setattr(
        "personalized_t2i.training.runner.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "unavailable"),
    )
    assert _gpu_snapshot() == (None, None)


def test_git_revision_does_not_run_nvidia_smi(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "abc123\n", "")

    monkeypatch.setattr("personalized_t2i.training.runner.subprocess.run", fake_run)
    assert _git_revision(tmp_path) == "abc123"
    assert commands == [["git", "rev-parse", "HEAD"]]


def test_alpha_must_equal_rank(pilot_config, tmp_path):
    config, _train_dir, _output_dir = pilot_config
    config["training"]["alpha"] = 4

    with pytest.raises(ValueError, match="alpha must equal"):
        validate_training_config(config, tmp_path)


def test_run_id_must_match_training_config(pilot_config, tmp_path):
    config, _train_dir, _output_dir = pilot_config
    config["run"]["id"] = "dog_plush_n1_r16_ts41"

    with pytest.raises(ValueError, match="run.id must match"):
        validate_training_config(config, tmp_path)


def test_image_count_must_match_subset_size(pilot_config, tmp_path):
    config, train_dir, _output_dir = pilot_config
    (train_dir / "two.png").write_bytes(b"image")

    with pytest.raises(ValueError, match="contains 2 images"):
        validate_training_config(config, tmp_path)


def test_existing_output_is_not_overwritten(pilot_config, tmp_path):
    config, _train_dir, output_dir = pilot_config
    output_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        validate_training_config(config, tmp_path)


def test_trainer_command_maps_config_to_official_cli(pilot_config, tmp_path):
    config, train_dir, output_dir = pilot_config
    trainer_script = tmp_path / "diffusers" / "examples" / "dreambooth" / "train_dreambooth_lora.py"
    trainer_output = output_dir / "trainer_output"

    command = build_trainer_command(
        "accelerate",
        trainer_script,
        config,
        trainer_output,
        train_dir,
    )

    assert command[:4] == ["accelerate", "launch", "--mixed_precision", "fp16"]
    assert str(trainer_script) in command
    assert "--rank" in command
    assert command[command.index("--rank") + 1] == "16"
    assert command[command.index("--instance_data_dir") + 1] == str(train_dir)
