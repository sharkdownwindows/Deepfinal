"""Configuration loading and validation for experiment runs."""

from pathlib import Path
import re

import yaml


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROTOCOL_VERSION = "v1"
PROMPT_BANK_VERSION = "v1"

ALLOWED_SUBSET_SIZES = {1, 3, 5, 10}
ALLOWED_RANKS = {4, 16, 32}

CORE_CELLS = {
    (1, 16),
    (3, 16),
    (5, 16),
    (10, 16),
    (5, 4),
    (5, 32),
}

EXPECTED_RESOLUTION = 512
EXPECTED_LEARNING_RATE = 1e-4
EXPECTED_MAX_TRAIN_STEPS = 500
EXPECTED_CHECKPOINTING_STEPS = 100
EXPECTED_BATCH_SIZE = 1
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 1
EXPECTED_TRAINING_SCHEDULER = "constant"
EXPECTED_INFERENCE_SCHEDULER = "fixed"
EXPECTED_WARMUP_STEPS = 0
EXPECTED_MIXED_PRECISION = "fp16"
EXPECTED_LORA_DROPOUT = 0.0
EXPECTED_LORA_TARGET_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
]

EXPECTED_PROMPT_COUNT = 8
EXPECTED_INFERENCE_SEEDS = [11, 22, 33, 44]
EXPECTED_INFERENCE_STEPS = 30
EXPECTED_GUIDANCE_SCALE = 7.5
EXPECTED_NEGATIVE_PROMPT = ""
EXPECTED_LORA_SCALE = 1.0


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    return config


def _require_mapping(config: dict, name: str) -> dict:
    value = config.get(name)

    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")

    return value


def _require_string(section: dict, field: str) -> str:
    key = field.split(".")[-1]
    value = section.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")

    return value


def _require_strict_int(section: dict, field: str) -> int:
    key = field.split(".")[-1]
    value = section.get(key)

    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")

    return value


def _require_bool(section: dict, field: str, expected: bool) -> None:
    key = field.split(".")[-1]
    value = section.get(key)

    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")

    if value is not expected:
        raise ValueError(
            f"{field} must be {str(expected).lower()}"
        )


def _validate_run_id(
    run_id: str,
    concept_id: str,
    subset_size: int,
    rank: int,
    seed: int,
) -> None:
    """Validate run ID format and consistency with the core experiment cell."""

    pattern = (
        r"^"
        r"(?P<concept>[A-Za-z][A-Za-z0-9_-]*?)"
        r"_n(?P<n>[0-9]+)"
        r"_r(?P<rank>[0-9]+)"
        r"(?:_ts(?P<seed>[0-9]+))?"
        r"$"
    )

    match = re.fullmatch(pattern, run_id)

    if match is None:
        raise ValueError(
            "run.id must follow '<concept>_n<n>_r<rank>' "
            "or '<concept>_n<n>_r<rank>_ts<seed>'"
        )

    run_n = int(match.group("n"))
    run_rank = int(match.group("rank"))

    if run_n != subset_size:
        raise ValueError(
            "run.id subset size does not match data.subset_size"
        )

    if run_rank != rank:
        raise ValueError(
            "run.id rank does not match training.rank"
        )

    if match.group("concept") != concept_id:
        raise ValueError("run.id concept does not match data.concept_id")

    run_seed = match.group("seed")
    if run_seed is None or int(run_seed) != seed:
        raise ValueError("run.id training seed does not match training.seed")

    if (subset_size, rank) not in CORE_CELLS:
        raise ValueError(
            "Core experiment cell must be one of: "
            "(1,16), (3,16), (5,16), (10,16), (5,4), (5,32)"
        )


def _validate_model(model: dict) -> None:
    model_id = _require_string(model, "model.id")
    revision = _require_string(model, "model.revision")

    if model_id != MODEL_ID:
        raise ValueError(
            f"model.id must be {MODEL_ID}"
        )

    if revision in {
        "main",
        "master",
        "latest",
        "default",
    }:
        raise ValueError(
            "model.revision must be an immutable pinned revision, "
            "not a floating branch or tag"
        )

    if revision.startswith("refs/"):
        raise ValueError(
            "model.revision must be an immutable revision, "
            "not a refs/* value"
        )


def _validate_data(data: dict) -> None:
    _require_string(data, "data.concept_id")
    _require_string(data, "data.dataset_version")
    _require_string(data, "data.manifest")
    _require_string(data, "data.instance_prompt")

    subset_size = _require_strict_int(
        data,
        "data.subset_size",
    )

    if subset_size not in ALLOWED_SUBSET_SIZES:
        raise ValueError(
            f"data.subset_size must be one of "
            f"{sorted(ALLOWED_SUBSET_SIZES)}"
        )


def validate_config(config: dict) -> None:
    """Validate the core experiment configuration contract."""

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")

    required_sections = {
        "run",
        "model",
        "data",
        "training",
        "inference",
    }

    missing = required_sections - config.keys()

    if missing:
        raise ValueError(
            f"Missing sections: {sorted(missing)}"
        )

    run = _require_mapping(config, "run")
    model = _require_mapping(config, "model")
    data = _require_mapping(config, "data")
    training = _require_mapping(config, "training")
    inference = _require_mapping(config, "inference")

    run_id = _require_string(run, "run.id")

    protocol_version = _require_string(
        run,
        "run.protocol_version",
    )

    if protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f"run.protocol_version must be {PROTOCOL_VERSION}"
        )

    _validate_model(model)
    _validate_data(data)

    rank = _require_strict_int(
        training,
        "training.rank",
    )

    alpha = _require_strict_int(
        training,
        "training.alpha",
    )

    subset_size = data["subset_size"]

    if rank not in ALLOWED_RANKS:
        raise ValueError(
            f"training.rank must be one of "
            f"{sorted(ALLOWED_RANKS)}"
        )

    if (subset_size, rank) not in CORE_CELLS:
        raise ValueError(
            "Invalid core experiment cell: "
            f"n{subset_size}-r{rank}. "
            "Allowed cells are n1-r16, n3-r16, n5-r16, "
            "n10-r16, n5-r4, n5-r32."
        )

    if alpha != rank:
        raise ValueError(
            "training.alpha must equal training.rank"
        )

    if training.get("resolution") != EXPECTED_RESOLUTION:
        raise ValueError(
            "training.resolution must be 512"
        )

    if training.get("learning_rate") != EXPECTED_LEARNING_RATE:
        raise ValueError(
            "training.learning_rate must be 1e-4"
        )

    if training.get("max_train_steps") != EXPECTED_MAX_TRAIN_STEPS:
        raise ValueError(
            "training.max_train_steps must be 500"
        )

    if training.get("checkpointing_steps") != EXPECTED_CHECKPOINTING_STEPS:
        raise ValueError(
            "training.checkpointing_steps must be 100"
        )

    if training.get("batch_size") != EXPECTED_BATCH_SIZE:
        raise ValueError(
            "training.batch_size must be 1"
        )

    if (
        training.get("gradient_accumulation_steps")
        != EXPECTED_GRADIENT_ACCUMULATION_STEPS
    ):
        raise ValueError(
            "training.gradient_accumulation_steps must be 1"
        )

    if training.get("scheduler") != EXPECTED_TRAINING_SCHEDULER:
        raise ValueError(
            "training.scheduler must be constant"
        )

    if training.get("warmup_steps") != EXPECTED_WARMUP_STEPS:
        raise ValueError(
            "training.warmup_steps must be 0"
        )

    if training.get("mixed_precision") != EXPECTED_MIXED_PRECISION:
        raise ValueError(
            "training.mixed_precision must be fp16"
        )

    _require_bool(
        training,
        "training.train_text_encoder",
        False,
    )

    _require_bool(
        training,
        "training.prior_preservation",
        False,
    )

    if training.get("lora_dropout") != EXPECTED_LORA_DROPOUT:
        raise ValueError(
            "training.lora_dropout must be 0.0"
        )

    if training.get("lora_target_modules") != EXPECTED_LORA_TARGET_MODULES:
        raise ValueError(
            "training.lora_target_modules must be "
            f"{EXPECTED_LORA_TARGET_MODULES}"
        )

    _require_bool(
        training,
        "training.center_crop",
        True,
    )

    _require_bool(
        training,
        "training.random_flip",
        False,
    )

    seed = _require_strict_int(
        training,
        "training.seed",
    )

    _validate_run_id(
        run_id,
        data["concept_id"],
        subset_size,
        rank,
        seed,
    )

    prompt_bank_version = _require_string(
        inference,
        "inference.prompt_bank_version",
    )

    if prompt_bank_version != PROMPT_BANK_VERSION:
        raise ValueError(
            f"inference.prompt_bank_version must be "
            f"{PROMPT_BANK_VERSION}"
        )

    if inference.get("prompt_count") != EXPECTED_PROMPT_COUNT:
        raise ValueError(
            "inference.prompt_count must be 8"
        )

    if inference.get("seeds") != EXPECTED_INFERENCE_SEEDS:
        raise ValueError(
            "inference.seeds must be [11, 22, 33, 44]"
        )

    if inference.get("resolution") != EXPECTED_RESOLUTION:
        raise ValueError(
            "inference.resolution must be 512"
        )

    if inference.get("num_inference_steps") != EXPECTED_INFERENCE_STEPS:
        raise ValueError(
            "inference.num_inference_steps must be 30"
        )

    if inference.get("guidance_scale") != EXPECTED_GUIDANCE_SCALE:
        raise ValueError(
            "inference.guidance_scale must be 7.5"
        )

    if inference.get("scheduler") != EXPECTED_INFERENCE_SCHEDULER:
        raise ValueError(
            "inference.scheduler must be fixed"
        )

    if inference.get("negative_prompt") != EXPECTED_NEGATIVE_PROMPT:
        raise ValueError(
            "inference.negative_prompt must be empty"
        )

    if inference.get("lora_scale") != EXPECTED_LORA_SCALE:
        raise ValueError(
            "inference.lora_scale must be 1.0"
        )
