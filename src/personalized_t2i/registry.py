"""Run ID helpers, artifact registry, and status lifecycle."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

from personalized_t2i.environment import validate_environment_metadata


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


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    temporary_path.replace(path)


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_provenance(config: dict, repository_root: str | Path = ".") -> dict:
    model = config.get("model")
    data = config.get("data")
    training = config.get("training")
    if not isinstance(model, dict) or not isinstance(data, dict):
        raise ValueError("Run config must include model and data mappings")
    for key in ("id", "revision"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError(f"model.{key} is required for run provenance")
    for key in ("concept_id", "dataset_version", "manifest"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"data.{key} is required for run provenance")
    if type(data.get("subset_size")) is not int:
        raise ValueError("data.subset_size is required for run provenance")

    manifest_path = Path(data["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = Path(repository_root) / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")

    dataset = {
        "concept_id": data["concept_id"],
        "dataset_version": data["dataset_version"],
        "subset_size": data["subset_size"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "train_data_dir": data.get("train_data_dir"),
    }
    provenance = {
        "model": {"id": model["id"], "revision": model["revision"]},
        "dataset": dataset,
    }
    if isinstance(training, dict):
        provenance["training"] = {
            key: training[key]
            for key in ("rank", "alpha", "seed", "max_train_steps")
            if key in training
        }
    configured_hash = data.get("manifest_sha256")
    if configured_hash and configured_hash.lower() != dataset["manifest_sha256"]:
        raise ValueError("data.manifest_sha256 does not match the manifest")
    return provenance


def create_run(
    run_id: str,
    config: dict,
    environment: dict,
    artifacts_root: str | Path = "artifacts",
    repository_root: str | Path = ".",
) -> Path:
    """Create a new run with queued status and provenance files."""
    validate_environment_metadata(environment)
    provenance = _build_provenance(config, repository_root)
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

    with (run_dir / "provenance.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(provenance, f, indent=2)

    status = {
        "run_id": run_id,
        "status": "queued",
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    _write_json_atomic(run_dir / "status.json", status)

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

    _write_json_atomic(status_file, current)
