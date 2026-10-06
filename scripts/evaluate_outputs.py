"""Thin command-line entry point for automated output evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from personalized_t2i.evaluation.alignment import evaluate_run_clip
from personalized_t2i.evaluation.fidelity import evaluate_run_dino


def _print_summary(name: str, rows: list[dict]) -> None:
    valid_count = sum(bool(row["valid"]) for row in rows)
    invalid_count = len(rows) - valid_count

    print(f"{name}_samples: {len(rows)}")
    print(f"{name}_valid: {valid_count}")
    print(f"{name}_invalid: {invalid_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generated samples with DINOv2 and CLIP."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID whose artifacts/<run_id>/metadata.jsonl will be evaluated.",
    )
    parser.add_argument(
        "--metric",
        choices=("all", "dino", "clip"),
        default="all",
        help="Metric to run. Default: all.",
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
        "--concept-registry",
        default="data/manifests/concepts.csv",
        help="Concept registry containing class nouns and unique tokens.",
    )
    parser.add_argument(
        "--output",
        default="results/metrics_per_sample.csv",
        help="Per-sample metrics CSV to update.",
    )

    args = parser.parse_args()

    print(f"run_id: {args.run_id}")

    if args.metric in {"all", "dino"}:
        dino_rows = evaluate_run_dino(
            run_id=args.run_id,
            artifacts_root=args.artifacts_root,
            eval_refs_root=args.eval_refs_root,
            output_path=args.output,
        )
        _print_summary("dino", dino_rows)

    if args.metric in {"all", "clip"}:
        clip_rows = evaluate_run_clip(
            run_id=args.run_id,
            artifacts_root=args.artifacts_root,
            concept_registry_path=args.concept_registry,
            output_path=args.output,
        )
        _print_summary("clip", clip_rows)

    print(f"output: {Path(args.output)}")


if __name__ == "__main__":
    main()