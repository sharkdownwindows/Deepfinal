"""Validate the provenance and evidence contract for EVAL-05 case records."""

from __future__ import annotations

import argparse
from pathlib import Path

from personalized_t2i.evaluation.failures import (
    load_failure_cases,
    validate_failure_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate EVAL-05 failure-case records without changing the CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/failures.csv"),
        help="Failure-case CSV (default: results/failures.csv).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Root used to resolve evidence paths when --check-files is enabled.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Also verify the LoRA/base/training image paths exist under --repo-root.",
    )
    args = parser.parse_args()

    try:
        rows, errors = load_failure_cases(args.input)
    except OSError as exc:
        print(f"Cannot read {args.input}: {exc}")
        return 2

    errors.extend(
        validate_failure_rows(
            rows,
            repo_root=args.repo_root,
            check_files=args.check_files,
        )
    )
    if not rows and not errors:
        errors.append("no representative cases recorded; EVAL-05 evidence is pending")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Invalid failure-case file: {len(errors)} issue(s).")
        return 1

    print(f"Valid failure-case file: {len(rows)} case(s). Source CSV unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
