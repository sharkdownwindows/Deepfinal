"""Run the pinned Diffusers DreamBooth-LoRA trainer from a project config."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from personalized_t2i.config import CORE_CELLS, MODEL_ID
from personalized_t2i.environment import collect_environment_metadata
from personalized_t2i.registry import build_run_id, create_run, update_run_status


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ADAPTER_FILENAMES = (
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
)


def resolve_training_subset(config: dict, repo_root: Path) -> list[Path]:
    """Resolve the locked nested subset from the concept manifest."""
    data = config["data"]
    manifest_path = Path(data["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()
    subset_size = data["subset_size"]
    concept_id = data["concept_id"]
    train_data_dir = Path(data["train_data_dir"])
    if not train_data_dir.is_absolute():
        train_data_dir = repo_root / train_data_dir
    train_data_dir = train_data_dir.resolve()

    required = {"file_path", "sha256", "concept_id", "split", "subset_membership"}
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Dataset manifest is missing required columns: {manifest_path}")
        rows = list(reader)

    concept_train_rows = [
        row for row in rows
        if row.get("concept_id") == concept_id and row.get("split") == "train_pool"
    ]
    selected_rows = [
        row for row in concept_train_rows
        if str(subset_size) in {
            value.strip() for value in (row.get("subset_membership") or "").split(",")
        }
    ]
    if len(selected_rows) != subset_size:
        raise ValueError(
            f"Manifest selects {len(selected_rows)} images for {concept_id} n={subset_size}; "
            f"expected {subset_size}"
        )

    source_images = [
        path for path in train_data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(source_images) != len(concept_train_rows):
        raise ValueError(
            f"data.train_data_dir contains {len(source_images)} images; "
            f"manifest lists {len(concept_train_rows)} train_pool images"
        )

    selected_paths = []
    for row in selected_rows:
        source = Path(row["file_path"])
        if not source.is_absolute():
            source = repo_root / source
        source = source.resolve()
        try:
            source.relative_to(train_data_dir)
        except ValueError as exc:
            raise ValueError("Manifest training image is outside data.train_data_dir") from exc
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"Manifest training image is missing: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row["sha256"].strip().lower():
            raise ValueError(f"Training image hash mismatch: {source}")
        selected_paths.append(source)
    return selected_paths


def materialize_training_subset(config: dict, repo_root: Path, run_dir: Path) -> Path:
    """Copy only the manifest-locked subset into the ignored run artifacts."""
    selected = resolve_training_subset(config, repo_root)
    subset_dir = run_dir / "training_data"
    subset_dir.mkdir()
    provenance = []
    for index, source in enumerate(selected, start=1):
        destination = subset_dir / f"{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        provenance.append({
            "source": str(source),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    (run_dir / "training_subset.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return subset_dir


def write_pilot_metadata(
    run_dir: Path,
    config: dict,
    image_path: Path,
    image_size: tuple[int, int],
) -> Path:
    """Write the single adapter validation sample using the evaluation schema."""
    record = {
        "run_id": config["run"]["id"],
        "concept_id": config["data"]["concept_id"],
        "prompt_bank_version": config.get("inference", {}).get("prompt_bank_version", "pilot"),
        "prompt_id": "pilot",
        "prompt_category": "pilot",
        "prompt": config["inference"]["prompt"],
        "seed": config["inference"]["seed"],
        "generation_seed": config["inference"]["seed"],
        "generation_mode": "adapter",
        "lora_scale": config["inference"].get("lora_scale", 1.0),
        "base_model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "adapter_path": str((run_dir / "adapter").resolve()),
        "image_path": str(image_path.resolve()),
        "width": image_size[0],
        "height": image_size[1],
        "num_inference_steps": config["inference"]["num_inference_steps"],
        "guidance_scale": config["inference"].get("guidance_scale", 7.5),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = run_dir / "pilot_metadata.jsonl"
    with metadata_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metadata_path


def validate_training_config(config: dict, repo_root: Path) -> tuple[str, Path]:
    """Validate fields needed by the actual training entry point."""
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")

    for section in ("run", "model", "data", "training", "output", "inference"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")

    run = config["run"]
    model = config["model"]
    data = config["data"]
    training = config["training"]
    output = config["output"]
    inference = config["inference"]

    if model.get("id") != MODEL_ID:
        raise ValueError(f"model.id must be {MODEL_ID}")
    revision = model.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
        raise ValueError("model.revision must be an immutable hexadecimal commit SHA")

    concept_id = data.get("concept_id")
    prompt = data.get("instance_prompt")
    subset_size = data.get("subset_size")
    data_dir_value = data.get("train_data_dir")
    if not isinstance(concept_id, str) or not concept_id:
        raise ValueError("data.concept_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("data.instance_prompt must be a non-empty string")
    if type(subset_size) is not int or subset_size < 1:
        raise ValueError("data.subset_size must be a positive integer")
    rank = training.get("rank")
    if type(rank) is not int or (subset_size, rank) not in CORE_CELLS:
        raise ValueError("data.subset_size and training.rank must be a core experiment cell")
    if not isinstance(data_dir_value, str) or not data_dir_value:
        raise ValueError("data.train_data_dir must be a non-empty path")
    data_dir = Path(data_dir_value)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {data_dir}")
    manifest_value = data.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("data.manifest is required for dataset provenance")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    resolve_training_subset(config, repo_root)

    positive_ints = (
        ("training.rank", training.get("rank")),
        ("training.alpha", training.get("alpha")),
        ("training.resolution", training.get("resolution")),
        ("training.max_train_steps", training.get("max_train_steps")),
        ("training.checkpointing_steps", training.get("checkpointing_steps")),
        ("training.batch_size", training.get("batch_size")),
        ("training.gradient_accumulation_steps", training.get("gradient_accumulation_steps")),
    )
    for name, value in positive_ints:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    seed = training.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("training.seed must be a non-negative integer")
    if training["alpha"] != training["rank"]:
        raise ValueError("training.alpha must equal training.rank")
    learning_rate = training.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be a positive number")
    if training.get("mixed_precision", "fp16") not in {"no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision must be no, fp16, or bf16")
    if training.get("mixed_precision", "fp16") == "fp16" and training.get("train_text_encoder", False):
        raise ValueError("fp16 training of the text encoder is unsupported by this project wrapper")

    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.id must be a non-empty string")
    try:
        expected_run_id = build_run_id(config)
    except (KeyError, TypeError) as exc:
        raise ValueError("run config is missing concept, subset size, rank, or seed") from exc
    if run_id != expected_run_id:
        raise ValueError(
            "run.id must match <concept_id>_n<subset_size>_r<rank>_ts<seed>"
        )

    for field in ("prompt",):
        value = inference.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"inference.{field} must be a non-empty string")
    for field in ("seed", "num_inference_steps"):
        value = inference.get(field)
        if type(value) is not int or value < (0 if field == "seed" else 1):
            raise ValueError(f"inference.{field} has an invalid value")

    output_dir_value = output.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        raise ValueError("output.output_dir must be a non-empty path")
    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    if output_dir.name != run_id:
        raise ValueError("output.output_dir must end with run.id")
    if output_dir.exists():
        raise FileExistsError(f"Run output already exists: {output_dir}")

    return run_id, output_dir


def build_trainer_command(
    accelerate: str,
    trainer_script: Path,
    config: dict,
    trainer_output: Path,
    data_dir: Path,
) -> list[str]:
    """Translate project config into the official Diffusers script CLI."""
    model = config["model"]
    data = config["data"]
    training = config["training"]
    command = [
        accelerate,
        "launch",
        "--mixed_precision",
        training.get("mixed_precision", "fp16"),
        str(trainer_script),
        # The Diffusers script uses its own argument to upcast trainable LoRA
        # parameters to float32. Passing precision only to `accelerate launch`
        # leaves those parameters in fp16 and GradScaler refuses to unscale them.
        "--mixed_precision",
        training.get("mixed_precision", "fp16"),
        "--pretrained_model_name_or_path",
        model["id"],
        "--revision",
        model["revision"],
        "--instance_data_dir",
        str(data_dir),
        "--instance_prompt",
        data["instance_prompt"],
        "--output_dir",
        str(trainer_output),
        "--resolution",
        str(training["resolution"]),
        "--train_batch_size",
        str(training["batch_size"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--lr_scheduler",
        training.get("scheduler", "constant"),
        "--lr_warmup_steps",
        str(training.get("warmup_steps", 0)),
        "--max_train_steps",
        str(training["max_train_steps"]),
        "--checkpointing_steps",
        str(training["checkpointing_steps"]),
        "--seed",
        str(training["seed"]),
        "--rank",
        str(training["rank"]),
    ]
    if training.get("gradient_checkpointing", True):
        command.append("--gradient_checkpointing")
    return command


def _gpu_snapshot() -> tuple[int | None, int | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None
    if result.returncode != 0 or not result.stdout.strip():
        return None, None
    try:
        used, total = result.stdout.splitlines()[0].split(",")
        return int(used.strip()), int(total.strip())
    except (ValueError, IndexError):
        return None, None


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


class _GpuMemoryMonitor:
    def __init__(self) -> None:
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            used, _ = _gpu_snapshot()
            if used is not None:
                self.samples.append(used)
            self.stop_event.wait(1.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


def run_training(config_path: Path, trainer_script: Path, repo_root: Path) -> Path:
    """Train via the official Diffusers script, load the adapter, and sample once."""
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    run_id, output_dir = validate_training_config(config, repo_root)

    trainer_script = trainer_script.resolve()
    if trainer_script.name != "train_dreambooth_lora.py":
        raise ValueError("Use Diffusers' official examples/dreambooth/train_dreambooth_lora.py")
    if not trainer_script.is_file():
        raise FileNotFoundError(f"Diffusers trainer script not found: {trainer_script}")
    if not ("dreambooth" in trainer_script.parent.name and "examples" in trainer_script.parent.parent.name):
        raise ValueError("Trainer must come from Diffusers examples/dreambooth")

    accelerate = shutil.which("accelerate")
    if accelerate is None:
        raise RuntimeError("accelerate CLI is missing; install the project's ML dependencies in the runtime")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ML-01 requires a CUDA GPU; select a GPU runtime (for example, Colab T4)")

    trainer_output = output_dir / "trainer_output"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    diffusers_root = trainer_script.parent.parent.parent
    environment = collect_environment_metadata(repo_root)
    environment["diffusers"] = {
        "git_commit": _git_revision(diffusers_root),
        "trainer_script": str(trainer_script),
    }
    run_dir = create_run(
        run_id,
        config,
        environment,
        artifacts_root=output_dir.parent,
        repository_root=repo_root,
    )
    update_run_status(run_id, "running", artifacts_root=output_dir.parent)

    training_started = time.monotonic()
    log_path = run_dir / "logs" / "train.log"
    monitor = _GpuMemoryMonitor()
    monitor.__enter__()
    try:
        data_dir = materialize_training_subset(config, repo_root, run_dir)
        command = build_trainer_command(
            accelerate, trainer_script, config, trainer_output, data_dir
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=trainer_script.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Diffusers trainer exited with code {return_code}")
        training_seconds = time.monotonic() - training_started

        adapter_path = next(
            (trainer_output / name for name in ADAPTER_FILENAMES if (trainer_output / name).is_file()),
            None,
        )
        if adapter_path is None:
            raise FileNotFoundError("Diffusers completed without a LoRA weight file")
        saved_adapter = run_dir / "adapter" / adapter_path.name
        shutil.copy2(adapter_path, saved_adapter)

        inference_started = time.monotonic()
        dtype = torch.float16 if config["training"].get("mixed_precision", "fp16") == "fp16" else torch.float32
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            config["model"]["id"],
            revision=config["model"]["revision"],
            torch_dtype=dtype,
        ).to("cuda")
        pipe.load_lora_weights(str(trainer_output), weight_name=adapter_path.name)
        generator = torch.Generator(device="cuda").manual_seed(config["inference"]["seed"])
        generated = pipe(
            prompt=config["inference"]["prompt"],
            num_inference_steps=config["inference"]["num_inference_steps"],
            guidance_scale=config["inference"].get("guidance_scale", 7.5),
            generator=generator,
            cross_attention_kwargs={"scale": config["inference"].get("lora_scale", 1.0)},
        ).images[0]
        generated_path = run_dir / "generations" / "adapter_pilot.png"
        generated.save(generated_path)
        write_pilot_metadata(
            run_dir,
            config,
            generated_path,
            generated.size,
        )
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inference_seconds = time.monotonic() - inference_started

        used_mib, total_mib = _gpu_snapshot()
        metrics = {
            "run_id": run_id,
            "training_steps": config["training"]["max_train_steps"],
            "training_wall_seconds": round(training_seconds, 3),
            "inference_wall_seconds": round(inference_seconds, 3),
            "total_wall_seconds": round(time.monotonic() - training_started, 3),
            "observed_peak_gpu_memory_used_mib": max(monitor.samples) if monitor.samples else None,
            "gpu_memory_total_mib": total_mib,
            "gpu_memory_note": "nvidia-smi samples include memory used by other processes",
            "final_gpu_memory_used_mib": used_mib,
            "adapter_path": str(saved_adapter.relative_to(run_dir)),
            "generation_path": str(generated_path.relative_to(run_dir)),
            "inference_seed": config["inference"]["seed"],
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        update_run_status(run_id, "completed", artifacts_root=output_dir.parent)
        return run_dir
    except Exception as exc:
        update_run_status(run_id, "failed", artifacts_root=output_dir.parent, error=str(exc))
        raise
    finally:
        monitor.__exit__()
