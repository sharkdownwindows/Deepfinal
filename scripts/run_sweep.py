"""Preflight or execute the protocol's 12-cell data-size sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from personalized_t2i.training.runner import run_training, validate_training_config


EXPECTED_SIZES = [1, 3, 5, 10]
REPORT_COLUMNS = [
    "run_id",
    "config_path",
    "status",
    "started_at",
    "finished_at",
    "error",
    "rerun_decision",
]


def classify_existing_artifact(output_dir: Path, expected_config: dict, repo_root: Path) -> tuple[str, str]:
    """Only recognize a completed sweep cell when its recorded evidence matches."""
    try:
        status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((output_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
        provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError):
        return "incomplete_existing_artifact", "review_existing_run; required evidence is missing or invalid"

    if (
        not isinstance(status, dict)
        or status.get("status") != "completed"
        or status.get("run_id") != expected_config["run"]["id"]
    ):
        return "incomplete_existing_artifact", "review_existing_run; status is not completed"
    if not isinstance(resolved, dict) or not isinstance(provenance, dict):
        return "incomplete_existing_artifact", "review_existing_run; config or provenance is invalid"

    expected_run = expected_config["run"]
    actual_run = resolved.get("run", {})
    provenance_run = provenance.get("run", {})
    provenance_dataset = provenance.get("dataset", {})
    if not all(isinstance(value, dict) for value in (actual_run, provenance_run, provenance_dataset)):
        return "incomplete_existing_artifact", "review_existing_run; run or dataset evidence is invalid"
    if (
        actual_run.get("id") != expected_run["id"]
        or actual_run.get("artifact_namespace", "core") != "core"
        or provenance_run.get("id") != expected_run["id"]
        or provenance_run.get("artifact_namespace", "core") != "core"
    ):
        return "incompatible_artifact", "review_existing_run; run id or artifact namespace differs"

    def expected_provenance(config: dict) -> tuple[dict, dict, dict]:
        model = config["model"]
        data = config["data"]
        training = config["training"]
        manifest = Path(data["manifest"])
        if not manifest.is_absolute():
            manifest = repo_root / manifest
        return (
            {"id": model["id"], "revision": model["revision"]},
            {
                "concept_id": data["concept_id"],
                "dataset_version": data["dataset_version"],
                "subset_size": data["subset_size"],
                "manifest": str(manifest.resolve()),
            },
            {key: training[key] for key in ("rank", "alpha", "seed", "max_train_steps")},
        )

    try:
        expected_model, expected_data, expected_training = expected_provenance(expected_config)
        actual_model, actual_data, actual_training = expected_provenance(resolved)
    except (KeyError, TypeError, OSError):
        return "incomplete_existing_artifact", "review_existing_run; resolved config lacks provenance fields"

    try:
        manifest_sha256 = hashlib.sha256(Path(expected_data["manifest"]).read_bytes()).hexdigest()
    except OSError:
        return "incomplete_existing_artifact", "review_existing_run; dataset manifest is unavailable"
    evidence_matches = (
        actual_model == expected_model
        and actual_data == expected_data
        and actual_training == expected_training
        and provenance.get("model") == expected_model
        and all(provenance_dataset.get(key) == value for key, value in expected_data.items())
        and provenance_dataset.get("manifest_sha256") == manifest_sha256
        and provenance.get("training") == expected_training
    )
    if not evidence_matches:
        return "incompatible_artifact", "review_existing_run; resolved config or provenance does not match this cell"
    return "completed_existing", "verified_completed_run; preserve existing evidence"


def expand_sweep_matrix(matrix: dict) -> list[dict]:
    """Build resolved run configs from the versioned sweep matrix YAML."""
    if matrix.get("data_sizes") != EXPECTED_SIZES:
        raise ValueError(f"Sweep data_sizes must equal {EXPECTED_SIZES}")
    concepts = matrix.get("concepts")
    if not isinstance(concepts, dict) or len(concepts) != 3:
        raise ValueError("ML-04 requires exactly three concept configurations")
    for section in ("model", "training", "inference"):
        if not isinstance(matrix.get(section), dict):
            raise ValueError(f"Sweep matrix must define {section}")

    configs = []
    training = deepcopy(matrix["training"])
    training["rank"] = 16
    training["alpha"] = 16
    for concept_id, concept in concepts.items():
        if not isinstance(concept, dict):
            raise ValueError(f"Concept config must be a mapping: {concept_id}")
        token = concept.get("instance_token")
        class_noun = concept.get("class_noun")
        if not all(isinstance(value, str) and value.strip() for value in (token, class_noun)):
            raise ValueError(f"Concept {concept_id} needs instance_token and class_noun")
        for size in EXPECTED_SIZES:
            run_id = f"{concept_id}_n{size}_r16_ts{training.get('seed', 42)}"
            data = {
                "concept_id": concept_id,
                "dataset_version": concept["dataset_version"],
                "manifest": concept["manifest"],
                "subset_size": size,
                "train_data_dir": concept["train_data_dir"],
                "instance_prompt": f"a photo of {token} {class_noun}",
                "instance_token": token,
            }
            inference = deepcopy(matrix["inference"])
            inference["prompt"] = f"a photo of {token} {class_noun} on a wooden table"
            configs.append(
                {
                    "run": {"id": run_id, "protocol_version": matrix.get("protocol_version", "v1")},
                    "model": deepcopy(matrix["model"]),
                    "data": data,
                    "training": deepcopy(training),
                    "inference": inference,
                    "output": {"output_dir": f"{matrix.get('output_root', 'artifacts')}/{run_id}"},
                }
            )
    return configs


def write_status_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("configs/ml04_sweep.yaml"))
    parser.add_argument("--diffusers-script", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate all 12 cells without training")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = args.matrix.resolve()
    with matrix_path.open("r", encoding="utf-8") as stream:
        matrix = yaml.safe_load(stream)
    configs = expand_sweep_matrix(matrix)
    config_dir = repo_root / "artifacts" / "sweep_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    report_path = repo_root / "artifacts" / (
        "ml04_sweep_status_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".csv"
    )
    report_display_path = report_path.relative_to(repo_root)

    rows = []
    configs_to_run = []
    invalid_configs = []
    for config in configs:
        run_id = config["run"]["id"]
        config_path = config_dir / f"{run_id}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        output_value = Path(config["output"]["output_dir"])
        output_dir = output_value if output_value.is_absolute() else repo_root / output_value
        row = {
            "run_id": run_id,
            "config_path": str(config_path),
            "status": "planned",
            "started_at": "",
            "finished_at": "",
            "error": "",
            "rerun_decision": "none",
        }
        if output_dir.exists():
            row["status"], row["rerun_decision"] = classify_existing_artifact(
                output_dir, config, repo_root
            )
        else:
            try:
                validate_training_config(config, repo_root)
                configs_to_run.append((config, config_path, row))
            except Exception as exc:
                row["status"] = "preflight_failed"
                row["error"] = str(exc)
                invalid_configs.append(run_id)
        rows.append(row)

    write_status_report(report_path, rows)
    if invalid_configs:
        print(f"[FAIL] Preflight failed for: {', '.join(invalid_configs)}")
        print(f"[INFO] Report: {report_display_path}")
        raise SystemExit(1)
    print(f"[OK] Preflighted {len(configs)} matrix cells; {len(configs_to_run)} are runnable.")
    print(f"[INFO] Report: {report_display_path}")
    if args.dry_run:
        return
    if args.diffusers_script is None:
        raise SystemExit("Pass --diffusers-script to execute training, or use --dry-run.")

    trainer_script = args.diffusers_script.resolve()
    for config, config_path, row in configs_to_run:
        row["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            run_training(config_path, trainer_script, repo_root)
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            row["rerun_decision"] = "owner_review_required; failed run is retained"
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_status_report(report_path, rows)
        print(f"[{row['status'].upper()}] {row['run_id']}")

    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
