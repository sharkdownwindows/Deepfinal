"""Validation helpers for traceable qualitative failure cases."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


FAILURE_CASE_COLUMNS = (
    "case_id",
    "run_id",
    "concept_id",
    "prompt_id",
    "generation_seed",
    "failure_tag",
    "attribution",
    "severity",
    "lora_image_path",
    "base_image_path",
    "base_model_id",
    "base_model_revision",
    "nearest_training_image",
    "automated_evidence",
    "review_status",
    "reviewer_id",
    "notes",
)

FAILURE_TAGS = frozenset(
    {
        "underfit",
        "identity_drift",
        "background_leakage",
        "pose_copy",
        "prompt_refusal",
        "memorization",
        "structure_artifact",
        "rendering_artifact",
        "other",
    }
)
ATTRIBUTIONS = frozenset(
    {"base_model", "lora", "both", "uncertain"}
)
SEVERITIES = frozenset({"low", "medium", "high"})
REVIEW_STATUSES = frozenset({"confirmed", "needs_review"})


def validate_failure_rows(
    rows: Iterable[dict[str, str]],
    *,
    repo_root: str | Path | None = None,
    check_files: bool = False,
) -> list[str]:
    """Return actionable validation errors without modifying the source rows."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    errors: list[str] = []
    seen_case_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        prefix = f"row {row_number}"
        case_id = row.get("case_id", "").strip()
        if not case_id:
            errors.append(f"{prefix}: case_id is required")
        elif case_id in seen_case_ids:
            errors.append(f"{prefix}: duplicate case_id {case_id!r}")
        seen_case_ids.add(case_id)

        for field in ("run_id", "concept_id", "prompt_id", "generation_seed"):
            if not row.get(field, "").strip():
                errors.append(f"{prefix}: {field} is required for traceability")

        seed = row.get("generation_seed", "").strip()
        if seed:
            try:
                int(seed)
            except ValueError:
                errors.append(f"{prefix}: generation_seed must be an integer")

        tag = row.get("failure_tag", "").strip()
        if tag not in FAILURE_TAGS:
            errors.append(
                f"{prefix}: failure_tag must be one of {', '.join(sorted(FAILURE_TAGS))}"
            )

        attribution = row.get("attribution", "").strip()
        if attribution not in ATTRIBUTIONS:
            errors.append(
                f"{prefix}: attribution must be one of {', '.join(sorted(ATTRIBUTIONS))}"
            )
        elif attribution != "uncertain":
            for field in (
                "lora_image_path",
                "base_image_path",
                "base_model_id",
                "base_model_revision",
            ):
                if not row.get(field, "").strip():
                    errors.append(
                        f"{prefix}: {field} is required to attribute a failure; "
                        "compare base and LoRA outputs for the same prompt and seed"
                    )

        severity = row.get("severity", "").strip()
        if severity not in SEVERITIES:
            errors.append(
                f"{prefix}: severity must be one of {', '.join(sorted(SEVERITIES))}"
            )

        review_status = row.get("review_status", "").strip()
        if review_status not in REVIEW_STATUSES:
            errors.append(
                f"{prefix}: review_status must be one of "
                f"{', '.join(sorted(REVIEW_STATUSES))}"
            )
        if not row.get("reviewer_id", "").strip():
            errors.append(f"{prefix}: reviewer_id is required")
        if not row.get("notes", "").strip():
            errors.append(f"{prefix}: notes must describe the visible evidence")

        if not row.get("lora_image_path", "").strip():
            errors.append(f"{prefix}: lora_image_path is required for visual review")

        if tag == "memorization" and not row.get("nearest_training_image", "").strip():
            errors.append(
                f"{prefix}: nearest_training_image is required for a memorization case"
            )

        if check_files:
            path_fields = ["lora_image_path"]
            if attribution and attribution != "uncertain":
                path_fields.append("base_image_path")
            if tag == "memorization":
                path_fields.append("nearest_training_image")
            for field in path_fields:
                value = row.get(field, "").strip()
                if value and not (root / value).is_file():
                    errors.append(f"{prefix}: {field} does not exist: {value}")

    return errors


def load_failure_cases(
    path: str | Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """Load a failure-case CSV and check that its header follows the contract."""
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in FAILURE_CASE_COLUMNS if column not in header]
        if missing:
            return [], [f"CSV is missing required columns: {', '.join(missing)}"]
        return list(reader), []
