"""Thin command-line entry point for automated output evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from personalized_t2i.evaluation.fidelity import evaluate_run_dino


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generated samples with DINOv2 subject fidelity."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID whose artifacts/<run_id>/metadata.jsonl will be evaluated.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root directory containing run artifacts.",
    )
    parser.add_argument(
        "--eval-refs-root",
        default="data/eval_refs",
        help="Root directory containing held-out references by concept.",
    )
    parser.add_argument(
        "--output",
        default="results/metrics_per_sample.csv",
        help="Per-sample metrics CSV to update.",
    )

    args = parser.parse_args()

    rows = evaluate_run_dino(
        run_id=args.run_id,
        artifacts_root=args.artifacts_root,
        eval_refs_root=args.eval_refs_root,
        output_path=args.output,
    )

    valid_count = sum(bool(row["valid"]) for row in rows)
    invalid_count = len(rows) - valid_count

    print(f"run_id: {args.run_id}")
    print(f"samples: {len(rows)}")
    print(f"valid: {valid_count}")
    print(f"invalid: {invalid_count}")
    print(f"output: {Path(args.output)}")


if __name__ == "__main__":
    main()