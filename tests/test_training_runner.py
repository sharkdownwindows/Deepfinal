from pathlib import Path

import pytest

from personalized_t2i.training.runner import (
    build_trainer_command,
    validate_training_config,
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
    (train_dir / "one.jpg").write_bytes(b"image")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,file_path,sha256,concept_id,split\n", encoding="utf-8")
    output_dir = tmp_path / "artifacts" / "dog_plush_n1_r16_ts42"
    return valid_training_config(train_dir, output_dir, manifest), train_dir, output_dir


def test_training_config_accepts_core_values(pilot_config, tmp_path):
    config, _train_dir, output_dir = pilot_config

    run_id, resolved_output = validate_training_config(config, tmp_path)

    assert run_id == "dog_plush_n1_r16_ts42"
    assert resolved_output == output_dir


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
