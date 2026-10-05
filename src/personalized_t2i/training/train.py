"""Experiment execution entry point."""

from pathlib import Path

import yaml

from personalized_t2i.config import validate_config
from personalized_t2i.registry import build_run_id, create_run


def prepare_run(
    config_path: str | Path,
    environment: dict,
    artifacts_root: str | Path = "artifacts",
) -> Path:
    """Validate configuration and reserve a new artifact run."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    run_id = build_run_id(config)

    return create_run(
        run_id=run_id,
        config=config,
        environment=environment,
        artifacts_root=artifacts_root,
    )
