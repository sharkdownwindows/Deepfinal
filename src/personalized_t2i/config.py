"""Configuration loading and validation for experiment runs."""

from pathlib import Path
import yaml


ALLOWED_SUBSET_SIZES = {1, 3, 5, 10}
ALLOWED_RANKS = {4, 16, 32}


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    """Validate the core experiment configuration contract."""
    required_sections = {"run", "model", "data", "training"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing sections: {sorted(missing)}")

    run = config["run"]
    model = config["model"]
    data = config["data"]
    training = config["training"]

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

    if not isinstance(training.get("seed"), int):
        raise ValueError("training.seed must be an integer")
