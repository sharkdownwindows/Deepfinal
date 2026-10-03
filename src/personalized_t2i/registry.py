"""Run ID helpers and registry checks."""

import json
from pathlib import Path


def build_run_id(config: dict) -> str:
    """Build the canonical run ID from an experiment configuration."""
    concept_id = config["data"]["concept_id"]
    subset_size = config["data"]["subset_size"]
    rank = config["training"]["rank"]
    seed = config["training"]["seed"]

    return f"{concept_id}_n{subset_size}_r{rank}_ts{seed}"


def ensure_run_available(
    run_id: str,
    artifacts_root: str | Path = "artifacts",
) -> None:
    """Reject a run ID that already exists or has completed."""
    run_dir = Path(artifacts_root) / run_id

    if not run_dir.exists():
        return

    status_file = run_dir / "status.json"

    if status_file.exists():
        with status_file.open("r", encoding="utf-8") as f:
            status = json.load(f).get("status")

        if status == "completed":
            raise FileExistsError(
                f"Completed run cannot be overwritten: {run_id}"
            )

    raise FileExistsError(f"Run ID already exists: {run_id}")