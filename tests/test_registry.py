import json
from pathlib import Path

import pytest

from personalized_t2i.registry import (
    create_run,
    update_run_status,
)


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


def environment():
    return {
        "git": {"commit": "test-commit"},
        "python": {"version": "3.11.9"},
        "packages": {"torch": "test"},
        "cuda": {"available": False, "version": None},
        "driver": {"version": "test"},
        "gpu": {"name": "test", "count": 0},
        "runtime": {"platform": "test", "device": "cpu"},
    }


def read_status(run_dir: Path):
    with (run_dir / "status.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def test_run_starts_queued(tmp_path):
    run_dir = create_run(
        "toy01_n5_r16_ts42",
        valid_config(),
        environment(),
        tmp_path,
    )

    status = read_status(run_dir)

    assert status["status"] == "queued"
    assert status["error"] is None


def test_run_can_complete(tmp_path):
    run_id = "toy01_n5_r16_ts42"

    create_run(run_id, valid_config(), environment(), tmp_path)

    update_run_status(run_id, "running", tmp_path)
    update_run_status(run_id, "completed", tmp_path)

    status = read_status(tmp_path / run_id)

    assert status["status"] == "completed"
    assert status["started_at"] is not None
    assert status["completed_at"] is not None
    assert status["error"] is None


def test_failed_run_stores_error(tmp_path):
    run_id = "toy01_n5_r16_ts42"

    create_run(run_id, valid_config(), environment(), tmp_path)

    update_run_status(
        run_id,
        "running",
        tmp_path,
    )
    update_run_status(
        run_id,
        "failed",
        tmp_path,
        error="training failed",
    )

    status = read_status(tmp_path / run_id)

    assert status["status"] == "failed"
    assert status["error"] == "training failed"


def test_invalid_status_transition_is_rejected(tmp_path):
    run_id = "toy01_n5_r16_ts42"

    create_run(run_id, valid_config(), environment(), tmp_path)

    with pytest.raises(ValueError):
        update_run_status(run_id, "completed", tmp_path)


def test_completed_run_cannot_be_overwritten(tmp_path):
    run_id = "toy01_n5_r16_ts42"

    run_dir = create_run(
        run_id,
        valid_config(),
        environment(),
        tmp_path,
    )

    update_run_status(run_id, "running", tmp_path)
    update_run_status(run_id, "completed", tmp_path)

    with pytest.raises(FileExistsError):
        create_run(
            run_id,
            valid_config(),
            environment(),
            tmp_path,
        )

    assert read_status(run_dir)["status"] == "completed"


def test_provenance_files_are_created(tmp_path):
    run_id = "toy01_n5_r16_ts42"

    run_dir = create_run(
        run_id,
        valid_config(),
        environment(),
        tmp_path,
    )

    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "logs").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "adapter").is_dir()
    assert (run_dir / "generations").is_dir()