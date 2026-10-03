"""Configuration loading and validation for experiment runs."""

from pathlib import Path
import yaml


ALLOWED_SUBSET_SIZES = {1, 3, 5, 10}
ALLOWED_RANKS = {4, 16, 32}
EXPECTED_RESOLUTION = 512
EXPECTED_LEARNING_RATE = 1e-4
EXPECTED_MAX_TRAIN_STEPS = 500
EXPECTED_CHECKPOINTING_STEPS = 100
EXPECTED_BATCH_SIZE = 1
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 1
EXPECTED_SCHEDULER = "constant"
EXPECTED_WARMUP_STEPS = 0
EXPECTED_MIXED_PRECISION = "fp16"
EXPECTED_LORA_DROPOUT = 0.0
EXPECTED_LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    """Validate the core experiment configuration contract."""
    required_sections = {"run", "model", "data", "training", "inference"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing sections: {sorted(missing)}")

    run = config["run"]
    model = config["model"]
    data = config["data"]
    training = config["training"]
    inference = config["inference"]

    if not run.get("id"):
        raise ValueError("run.id is required")

    if not run.get("protocol_version"):
        raise ValueError("run.protocol_version is required")

    if not model.get("id"):
        raise ValueError("model.id is required")

    if not model.get("revision"):
        raise ValueError("model.revision is required")

    if data.get("subset_size") not in ALLOWED_SUBSET_SIZES:
        raise ValueError(
            f"data.subset_size must be one of {sorted(ALLOWED_SUBSET_SIZES)}"
        )

    if training.get("rank") not in ALLOWED_RANKS:
        raise ValueError(
            f"training.rank must be one of {sorted(ALLOWED_RANKS)}"
        )

    if training.get("alpha") != training.get("rank"):
        raise ValueError("training.alpha must equal training.rank")

    if training.get("resolution") != EXPECTED_RESOLUTION:
        raise ValueError("training.resolution must be 512")

    if training.get("learning_rate") != EXPECTED_LEARNING_RATE:
        raise ValueError("training.learning_rate must be 1e-4")

    if training.get("max_train_steps") != EXPECTED_MAX_TRAIN_STEPS:
        raise ValueError("training.max_train_steps must be 500")

    if training.get("checkpointing_steps") != EXPECTED_CHECKPOINTING_STEPS:
        raise ValueError("training.checkpointing_steps must be 100")

    if training.get("batch_size") != EXPECTED_BATCH_SIZE:
        raise ValueError("training.batch_size must be 1")

    if training.get("gradient_accumulation_steps") != EXPECTED_GRADIENT_ACCUMULATION_STEPS:
        raise ValueError("training.gradient_accumulation_steps must be 1")

    if training.get("scheduler") != EXPECTED_SCHEDULER:
        raise ValueError("training.scheduler must be constant")

    if training.get("warmup_steps") != EXPECTED_WARMUP_STEPS:
        raise ValueError("training.warmup_steps must be 0")

    if training.get("mixed_precision") != EXPECTED_MIXED_PRECISION:
        raise ValueError("training.mixed_precision must be fp16")

    if training.get("train_text_encoder") is not False:
        raise ValueError("training.train_text_encoder must be false")

    if training.get("prior_preservation") is not False:
        raise ValueError("training.prior_preservation must be false")

    if training.get("lora_dropout") != EXPECTED_LORA_DROPOUT:
        raise ValueError("training.lora_dropout must be 0.0")

    if training.get("lora_target_modules") != EXPECTED_LORA_TARGET_MODULES:
        raise ValueError(
            f"training.lora_target_modules must be {EXPECTED_LORA_TARGET_MODULES}"
        )

    if training.get("center_crop") is not True:
        raise ValueError("training.center_crop must be true")

    if training.get("random_flip") is not False:
        raise ValueError("training.random_flip must be false")

    if not isinstance(training.get("seed"), int):
        raise ValueError("training.seed must be an integer")

    if inference.get("prompt_count") != 8:
        raise ValueError("inference.prompt_count must be 8")

    if inference.get("seeds") != [11, 22, 33, 44]:
        raise ValueError("inference.seeds must be [11, 22, 33, 44]")

    if inference.get("resolution") != 512:
        raise ValueError("inference.resolution must be 512")

    if inference.get("num_inference_steps") != 30:
        raise ValueError("inference.num_inference_steps must be 30")

    if inference.get("guidance_scale") != 7.5:
        raise ValueError("inference.guidance_scale must be 7.5")

    if inference.get("negative_prompt") != "":
        raise ValueError("inference.negative_prompt must be empty")

    if inference.get("lora_scale") != 1.0:
        raise ValueError("inference.lora_scale must be 1.0")
