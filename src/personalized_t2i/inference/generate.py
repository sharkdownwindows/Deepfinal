"""Evaluation image generation module."""

import json
from datetime import datetime, timezone
from pathlib import Path
import yaml

from personalized_t2i.config import (
    EXPECTED_GUIDANCE_SCALE,
    EXPECTED_INFERENCE_SEEDS,
    EXPECTED_INFERENCE_STEPS,
    EXPECTED_PROMPT_COUNT,
    PROMPT_BANK_VERSION,
)


def load_prompt_bank(
    prompt_path: str = "prompt_bank/evaluation_prompts.yaml",
    concept_id: str = None,
):
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if (
        not isinstance(data, dict)
        or data.get("version") != PROMPT_BANK_VERSION
    ):
        raise ValueError(
            f"Prompt bank version must be {PROMPT_BANK_VERSION}: {prompt_path}"
        )

    concepts = data.get("concepts", {})
    if not isinstance(concepts, dict) or not concepts:
        raise ValueError(f"No concepts found in prompt bank: {prompt_path}")

    if concept_id is None:
        concept_id = list(concepts.keys())[0]
    elif concept_id not in concepts:
        raise ValueError(f"Unknown concept in prompt bank: {concept_id}")

    concept_prompts = concepts[concept_id]
    if (
        not isinstance(concept_prompts, list)
        or len(concept_prompts) != EXPECTED_PROMPT_COUNT
        or any(not isinstance(item, dict) for item in concept_prompts)
    ):
        raise ValueError(
            f"{concept_id} must have {EXPECTED_PROMPT_COUNT} prompts"
        )

    prompt_ids = [item.get("prompt_id") for item in concept_prompts]
    required_categories = {
        "simple",
        "new_background",
        "viewpoint_action",
        "style",
        "challenging_composition",
    }
    prompt_categories = [
        item.get("category")
        for item in concept_prompts
    ]
    valid_prompt_ids = all(
        isinstance(prompt_id, str) and prompt_id.strip()
        for prompt_id in prompt_ids
    )
    valid_categories = all(
        isinstance(category, str) and category.strip()
        for category in prompt_categories
    )
    categories = set(prompt_categories) if valid_categories else set()
    if (
        not valid_prompt_ids
        or len(set(prompt_ids)) != EXPECTED_PROMPT_COUNT
        or not valid_categories
        or not required_categories.issubset(categories)
    ):
        raise ValueError(
            f"{concept_id} must have unique prompt IDs and all evaluation categories"
        )
    if any(
        not isinstance(item.get("prompt"), str) or not item["prompt"].strip()
        for item in concept_prompts
    ):
        raise ValueError(f"{concept_id} contains an empty prompt")

    prompts = []

    for item in concept_prompts:
        prompts.append(
            {
                "id": item.get("prompt_id"),
                "category": item.get("category"),
                "text": item.get("prompt"),
            }
        )

    return prompts, concept_id


def load_seeds(seed_path: str = "prompt_bank/generation_seeds.yaml"):
    with open(seed_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or data.get("version") != PROMPT_BANK_VERSION:
        raise ValueError(
            f"Seed bank version must be {PROMPT_BANK_VERSION}: {seed_path}"
        )

    seeds = data.get("seeds")
    if seeds != EXPECTED_INFERENCE_SEEDS:
        raise ValueError(
            f"Seed bank must contain {EXPECTED_INFERENCE_SEEDS}: {seed_path}"
        )

    return seeds


def generate_evaluation_batch(
    run_id: str,
    base_model_id: str = None,
    adapter_path: str = None,
    resolved_config_path: str = None,
    lora_scale: float = None,
    artifacts_root: str = "artifacts",
    prompt_bank_path: str = "prompt_bank/evaluation_prompts.yaml",
    seed_path: str = "prompt_bank/generation_seeds.yaml",
    device: str = None,
    pipeline_factory=None,
    generator_factory=None,
):
    """Generate the fixed prompt-by-seed matrix for base and optional LoRA."""
    artifact_dir = Path(artifacts_root) / run_id
    config_path = Path(resolved_config_path) if resolved_config_path else artifact_dir / "config.resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved run config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Resolved run config must be a YAML mapping")

    model = config.get("model", {})
    model_id = model.get("id")
    if base_model_id is not None and base_model_id != model_id:
        raise ValueError("base_model_id must match the resolved config model.id")
    revision = model.get("revision")
    concept_id = config.get("data", {}).get("concept_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Resolved config must define model.id")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Resolved config must define model.revision")
    if not isinstance(concept_id, str) or not concept_id.strip():
        raise ValueError("Resolved config must define data.concept_id")

    prompts, resolved_concept_id = load_prompt_bank(prompt_bank_path, concept_id)
    seeds = load_seeds(seed_path)
    inference = config.get("inference", {})
    configured_seeds = inference.get("seeds")
    if configured_seeds is not None and configured_seeds != seeds:
        raise ValueError(f"Resolved config inference.seeds must equal {seeds}")
    width = inference.get("resolution", 512)
    height = width
    steps = inference.get("num_inference_steps", EXPECTED_INFERENCE_STEPS)
    guidance = inference.get("guidance_scale", EXPECTED_GUIDANCE_SCALE)
    negative_prompt = inference.get("negative_prompt", "")
    if type(width) is not int or width <= 0:
        raise ValueError("inference.resolution must be a positive integer")
    if type(steps) is not int or steps <= 0:
        raise ValueError("inference.num_inference_steps must be a positive integer")
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)):
        raise ValueError("inference.guidance_scale must be numeric")
    if not isinstance(negative_prompt, str):
        raise ValueError("inference.negative_prompt must be a string")
    if lora_scale is None:
        lora_scale = inference.get("lora_scale", 1.0)
    if isinstance(lora_scale, bool) or not isinstance(lora_scale, (int, float)) or lora_scale < 0:
        raise ValueError("lora_scale must be non-negative")
    if inference.get("scheduler", "fixed") != "fixed":
        raise ValueError("Only the protocol's fixed inference scheduler is supported")

    resolved_adapter_path = Path(adapter_path) if adapter_path else None
    if resolved_adapter_path is not None and not resolved_adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter does not exist: {resolved_adapter_path}")
    modes = ["base", "adapter"] if resolved_adapter_path else ["base"]
    generation_root = artifact_dir / "generations" / "evaluation"
    metadata_file = artifact_dir / "metadata.jsonl"
    if metadata_file.exists():
        raise FileExistsError(f"Generation metadata already exists: {metadata_file}")

    expected_count = len(prompts) * len(seeds) * len(modes)
    output_paths = [
        generation_root / mode / f"{run_id}__{prompt['id']}__gs{seed}.png"
        for mode in modes
        for prompt in prompts
        for seed in seeds
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Generation outputs already exist: {existing[0]}")

    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
    if pipeline_factory is None:
        from diffusers import StableDiffusionPipeline

        def pipeline_factory(model_name, model_revision, torch_dtype, target_device):
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                revision=model_revision,
                torch_dtype=torch_dtype,
            )
            return pipeline.to(target_device)

    if generator_factory is None:
        generator_factory = lambda sample_seed: torch.Generator(device=resolved_device).manual_seed(sample_seed)

    pipe = pipeline_factory(model_id, revision, dtype, resolved_device)
    metadata_records = []
    try:
        for mode in modes:
            active_adapter = None
            active_scale = 0.0
            if mode == "adapter":
                active_adapter = str(resolved_adapter_path.resolve())
                if resolved_adapter_path.is_dir():
                    pipe.load_lora_weights(active_adapter)
                else:
                    pipe.load_lora_weights(
                        str(resolved_adapter_path.parent.resolve()),
                        weight_name=resolved_adapter_path.name,
                    )
                active_scale = lora_scale

            for prompt in prompts:
                for seed in seeds:
                    # A fresh RNG is required for each prompt/seed sample.
                    generator = generator_factory(seed)
                    result = pipe(
                        prompt=prompt["text"],
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=float(guidance),
                        generator=generator,
                        cross_attention_kwargs={"scale": active_scale} if active_adapter else None,
                    )
                    images = getattr(result, "images", None)
                    if not images or len(images) != 1:
                        raise RuntimeError("Diffusers must return exactly one image per sample")

                    image_path = generation_root / mode / f"{run_id}__{prompt['id']}__gs{seed}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    images[0].save(image_path)
                    metadata_records.append(
                        {
                            "run_id": run_id,
                            "concept_id": resolved_concept_id,
                            "prompt_bank_version": PROMPT_BANK_VERSION,
                            "prompt_id": prompt["id"],
                            "prompt_category": prompt["category"],
                            "prompt": prompt["text"],
                            "seed": seed,
                            "generation_seed": seed,
                            "generation_mode": mode,
                            "lora_scale": active_scale,
                            "base_model_id": model_id,
                            "model_revision": revision,
                            "adapter_path": active_adapter,
                            "image_path": str(image_path.resolve()),
                            "width": width,
                            "height": height,
                            "num_inference_steps": steps,
                            "guidance_scale": float(guidance),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

        validate_generation_completeness(
            metadata_records,
            prompts=prompts,
            seeds=seeds,
            modes=modes,
            run_id=run_id,
        )
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_metadata = metadata_file.with_suffix(".jsonl.tmp")
        with temporary_metadata.open("w", encoding="utf-8") as stream:
            for record in metadata_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_metadata.replace(metadata_file)
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[OK] Generated {len(metadata_records)} of {expected_count} expected samples.")
    print(f"[OK] Metadata: {metadata_file}")
    return str(metadata_file)


def validate_generation_completeness(
    records: list[dict],
    *,
    prompts: list[dict],
    seeds: list[int],
    modes: list[str],
    run_id: str,
) -> None:
    """Reject duplicate, missing, or nonexistent prompt/seed/mode outputs."""
    expected = {
        (mode, prompt["id"], seed)
        for mode in modes
        for prompt in prompts
        for seed in seeds
    }
    actual = set()
    for record in records:
        if record.get("run_id") != run_id:
            raise ValueError("Generation metadata contains a different run_id")
        required_fields = (
            "concept_id",
            "prompt_id",
            "prompt",
            "generation_mode",
            "base_model_id",
            "model_revision",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "image_path",
        )
        missing_fields = [
            key
            for key in required_fields
            if key not in record
            or record[key] is None
            or (isinstance(record[key], str) and not record[key].strip())
        ]
        if missing_fields:
            raise ValueError(
                f"Generation metadata is missing required fields: {missing_fields}"
            )
        key = (
            record.get("generation_mode"),
            record.get("prompt_id"),
            record.get("generation_seed", record.get("seed")),
        )
        if key in actual:
            raise ValueError(f"Duplicate generated sample: {key}")
        actual.add(key)
        image_path = record.get("image_path")
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"Generated sample is missing: {image_path}")
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"Generation matrix incomplete: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
