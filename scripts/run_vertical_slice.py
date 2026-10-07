"""Run a real configured LoRA pilot and score its adapter sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from personalized_t2i.evaluation.alignment import evaluate_run_clip
from personalized_t2i.evaluation.fidelity import evaluate_run_dino
from personalized_t2i.training.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, reload, generate, and score one SD 1.5 LoRA pilot."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diffusers-script", type=Path, required=True)
    parser.add_argument("--eval-refs-root", type=Path, default=Path("data/eval_refs"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics_per_sample.csv"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    run_id = config["run"]["id"]

    run_dir = run_training(config_path, args.diffusers_script.resolve(), repo_root)
    dino_rows = evaluate_run_dino(
        run_id,
        artifacts_root=run_dir.parent,
        eval_refs_root=args.eval_refs_root,
        output_path=args.output,
        metadata_path=run_dir / "pilot_metadata.jsonl",
    )
    clip_rows = evaluate_run_clip(
        run_id,
        artifacts_root=run_dir.parent,
        output_path=args.output,
        metadata_path=run_dir / "pilot_metadata.jsonl",
    )
    if not dino_rows or not clip_rows or not all(
        row["valid"] for row in (*dino_rows, *clip_rows)
    ):
        raise SystemExit(
            "Vertical slice did not produce valid DINO and CLIP scores; "
            f"inspect {args.output} and artifacts/{run_id}/status.json"
        )
    print(f"[OK] Train/load/generate/score completed for {run_id}")
    print(f"[OK] Metrics: {args.output}")


if __name__ == "__main__":
    main()
