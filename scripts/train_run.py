"""Thin CLI entry point for a config-driven Diffusers DreamBooth-LoRA run."""

import argparse
import os
from pathlib import Path

from personalized_t2i.training.runner import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and smoke-test an SD 1.5 LoRA from a YAML config."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--diffusers-script",
        type=Path,
        default=os.environ.get("DIFFUSERS_TRAIN_DREAMBOOTH_LORA"),
        help=(
            "Path to Diffusers examples/dreambooth/train_dreambooth_lora.py. "
            "May also be set with DIFFUSERS_TRAIN_DREAMBOOTH_LORA."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diffusers_script is None:
        raise SystemExit(
            "Pass --diffusers-script or set DIFFUSERS_TRAIN_DREAMBOOTH_LORA "
            "to the official Diffusers DreamBooth-LoRA example script."
        )
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    run_dir = run_training(config_path, args.diffusers_script, repo_root)
    print(f"Training and adapter inference completed: {run_dir}")


if __name__ == "__main__":
    main()
