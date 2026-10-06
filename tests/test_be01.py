from pathlib import Path

import pytest

from personalized_t2i.config import validate_config
from personalized_t2i.registry import build_run_id, ensure_run_available


def valid_config():
    return {
        "run": {
            "id": "toy01_n5_r16_ts42",
            "protocol_version": "v1",
        },
        "model": {
            "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "revision": "test-revision",
        },
        "data": {
            "concept_id": "toy01",
            "dataset_version": "v1",
            "manifest": "data/manifests/concepts.csv",
            "subset_size": 5,
            "instance_prompt": "a photo of toktoy toy",
        },
        "training": {
            "rank": 16,
            "alpha": 16,
            "resolution": 512,
            "learning_rate": 1e-4,
            "max_train_steps": 500,
            "checkpointing_steps": 100,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "scheduler": "constant",
            "warmup_steps": 0,
            "mixed_precision": "fp16",
            "train_text_encoder": False,
            "prior_preservation": False,
            "lora_dropout": 0.0,
            "lora_target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "center_crop": True,
            "random_flip": False,
            "seed": 42,
        },
        "inference": {
            "prompt_bank_version": "v1",
            "prompt_count": 8,
            "seeds": [11, 22, 33, 44],
            "resolution": 512,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "scheduler": "fixed",
            "negative_prompt": "",
            "lora_scale": 1.0,
        },
    }


def test_build_run_id():
    run_id = build_run_id(valid_config())

    assert run_id == "toy01_n5_r16_ts42"


def test_duplicate_run_id_is_rejected(tmp_path: Path):
    run_id = "toy01_n5_r16_ts42"
    (tmp_path / run_id).mkdir()

    with pytest.raises(FileExistsError):
        ensure_run_available(run_id, tmp_path)


def test_alpha_must_equal_rank():
    config = valid_config()
    config["training"]["alpha"] = 32

    with pytest.raises(ValueError):
        validate_config(config)


def test_missing_model_revision_is_rejected():
    config = valid_config()
    config["model"]["revision"] = ""

    with pytest.raises(ValueError):
        validate_config(config)


def test_missing_dataset_is_rejected():
    config = valid_config()
    config["data"]["dataset_version"] = ""

    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_steps_are_rejected():
    config = valid_config()
    config["training"]["max_train_steps"] = 600

    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_seed_is_rejected():
    config = valid_config()
    config["training"]["seed"] = "42"

    with pytest.raises(ValueError):
        validate_config(config)