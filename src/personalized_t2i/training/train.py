"""Experiment execution entry point."""

from pathlib import Path

import yaml

from personalized_t2i.config import validate_config
from personalized_t2i.preflight import preflight_config
from personalized_t2i.registry import build_run_id, create_run


def prepare_run(
    config_path: str | Path,
    environment: dict,
    artifacts_root: str | Path = "artifacts",
    concepts_registry: str | Path = "data/manifests/concepts.csv",
) -> Path:
    """Validate, preflight, and reserve a new artifact run."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    preflight_config(
        config,
        concepts_registry=concepts_registry,
    )

    run_id = build_run_id(config)

    return create_run(
        run_id=run_id,
        config=config,
        environment=environment,
        artifacts_root=artifacts_root,
    )
