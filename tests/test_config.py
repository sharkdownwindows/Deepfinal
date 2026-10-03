from personalized_t2i.config import validate_config


def valid_config():
    return {
        "run": {
            "id": "toy01_n5_r16_ts42",
            "protocol_version": "v1",
        },
        "model": {
            "id": "test-model",
            "revision": "test-revision",
        },
        "data": {
            "concept_id": "toy01",
            "dataset_version": "v1",
            "manifest": "data/manifests/toy01_v1.csv",
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


def test_valid_config():
    validate_config(valid_config())


def test_invalid_rank():
    config = valid_config()
    config["training"]["rank"] = 8

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Invalid rank was accepted")


def test_missing_model_revision():
    config = valid_config()
    config["model"]["revision"] = ""

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Missing model revision was accepted")
