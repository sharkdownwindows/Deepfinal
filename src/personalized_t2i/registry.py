"""Run ID helpers, artifact registry, and status lifecycle."""

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ALLOWED_STATUSES = {"queued", "running", "completed", "failed"}

ALLOWED_TRANSITIONS = {
    "queued": {"running"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}


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
    """Reject a run ID that already exists."""
    run_dir = Path(artifacts_root) / run_id

    if run_dir.exists():
        raise FileExistsError(f"Run ID already exists: {run_id}")


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _status_path(run_id: str, artifacts_root: str | Path) -> Path:
    """Return the status file path for a run."""
    return Path(artifacts_root) / run_id / "status.json"


def create_run(
    run_id: str,
    config: dict,
    environment: dict,
    artifacts_root: str | Path = "artifacts",
) -> Path:
    """Create a new run with queued status and provenance files."""
    ensure_run_available(run_id, artifacts_root)

    run_dir = Path(artifacts_root) / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "logs").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "adapter").mkdir()
    (run_dir / "generations").mkdir()

    with (run_dir / "config.resolved.yaml").open(
        "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with (run_dir / "environment.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(environment, f, indent=2)

    status = {
        "run_id": run_id,
        "status": "queued",
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    with (run_dir / "status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    return run_dir


def update_run_status(
    run_id: str,
    status: str,
    artifacts_root: str | Path = "artifacts",
    error: str | None = None,
) -> None:
    """Update a run status while enforcing the lifecycle."""
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid run status: {status}")

    status_file = _status_path(run_id, artifacts_root)

    if not status_file.exists():
        raise FileNotFoundError(f"Run does not exist: {run_id}")

    with status_file.open("r", encoding="utf-8") as f:
        current = json.load(f)

    current_status = current["status"]

    if status not in ALLOWED_TRANSITIONS[current_status]:
        raise ValueError(
            f"Invalid status transition: "
            f"{current_status} -> {status}"
        )

    if status == "running":
        current["started_at"] = _utc_now()

    if status == "completed":
        current["completed_at"] = _utc_now()
        current["error"] = None

    if status == "failed":
        current["error"] = error or "Run failed"

    current["status"] = status

    with status_file.open("w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)