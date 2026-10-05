from personalized_t2i.config import validate_config


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
            "lora_target_modules": [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
            ],
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


def test_reject_n1_r4():
    config = valid_config()

    config["data"]["subset_size"] = 1
    config["training"]["rank"] = 4
    config["training"]["alpha"] = 4
    config["run"]["id"] = "toy01_n1_r4_ts42"

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Invalid core cell n1-r4 was accepted")


def test_reject_n3_r32():
    config = valid_config()

    config["data"]["subset_size"] = 3
    config["training"]["rank"] = 32
    config["training"]["alpha"] = 32
    config["run"]["id"] = "toy01_n3_r32_ts42"

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Invalid core cell n3-r32 was accepted")


def test_accept_six_core_cells():
    core_cells = [
        (1, 16),
        (3, 16),
        (5, 16),
        (10, 16),
        (5, 4),
        (5, 32),
    ]

    for subset_size, rank in core_cells:
        config = valid_config()

        config["data"]["subset_size"] = subset_size
        config["training"]["rank"] = rank
        config["training"]["alpha"] = rank
        config["run"]["id"] = (
            f"toy01_n{subset_size}_r{rank}_ts42"
        )

        validate_config(config)


def test_missing_instance_prompt():
    config = valid_config()
    del config["data"]["instance_prompt"]

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Missing data.instance_prompt was accepted")


def test_missing_prompt_bank_version():
    config = valid_config()
    del config["inference"]["prompt_bank_version"]

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Missing inference.prompt_bank_version was accepted")


def test_invalid_inference_scheduler():
    config = valid_config()
    config["inference"]["scheduler"] = "anything"

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Invalid inference.scheduler was accepted")


def test_seed_true_is_rejected():
    config = valid_config()
    config["training"]["seed"] = True

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Boolean seed was accepted as an integer")


def test_training_section_must_be_mapping():
    config = valid_config()
    config["training"] = []

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("List training section was accepted")


def test_null_data_section_is_rejected():
    config = valid_config()
    config["data"] = None

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Null data section was accepted")


def test_null_inference_section_is_rejected():
    config = valid_config()
    config["inference"] = None

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Null inference section was accepted")


def test_prompt_count_must_be_integer():
    config = valid_config()
    config["inference"]["prompt_count"] = "8"

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("String prompt_count was accepted")


def test_inference_seeds_must_be_list():
    config = valid_config()
    config["inference"]["seeds"] = 11

    try:
        validate_config(config)
    except ValueError:
        return

    raise AssertionError("Non-list inference seeds were accepted")