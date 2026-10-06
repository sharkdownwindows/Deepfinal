"""Config-driven DreamBooth-LoRA training entry point."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DreamBooth-LoRA training from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training YAML configuration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg = config["run"]
    output_cfg = config["output"]

    run_id = run_cfg["id"]

    artifact_dir = Path(
        output_cfg.get(
            "output_dir",
            f"artifacts/{run_id}",
        )
    )

    checkpoint_dir = artifact_dir / "checkpoint"
    status_path = artifact_dir / "status.json"
    resolved_config_path = (
        artifact_dir / "config.resolved.yaml"
    )

    # Prevent accidental overwrite.
    if status_path.exists():
        raise RuntimeError(
            f"Run '{run_id}' already exists: "
            f"{artifact_dir}"
        )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the resolved configuration.
    with open(
        resolved_config_path,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    # Mark run as running.
    with open(
        status_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "status": "running",
                "run_id": run_id,
            },
            f,
            indent=2,
        )

    # The trainer reads the complete configuration itself.
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "train_custom_lora.py",
        "--config",
        str(config_path),
    ]

    log_path = artifact_dir / "train.log"

    try:
        print("Starting training...")
        print(f"Run ID: {run_id}")
        print(f"Config: {config_path}")
        print(f"Artifact directory: {artifact_dir}")

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)

            return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"Training failed with exit code "
                f"{return_code}"
            )

        if not checkpoint_dir.exists():
            raise RuntimeError(
                "Training finished but checkpoint "
                "directory was not created."
            )

        # Mark run as completed.
        with open(
            status_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "status": "completed",
                    "run_id": run_id,
                },
                f,
                indent=2,
            )

        print("\nTraining completed successfully.")
        print(
            f"Checkpoint directory: "
            f"{checkpoint_dir}"
        )

    except Exception as exc:
        # Mark run as failed.
        with open(
            status_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "status": "failed",
                    "run_id": run_id,
                    "error": str(exc),
                },
                f,
                indent=2,
            )

        raise


if __name__ == "__main__":
    main()