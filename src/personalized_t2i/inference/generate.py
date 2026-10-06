"""Evaluation image generation module."""

import json
import yaml
from pathlib import Path
from datetime import datetime, timezone

from PIL import Image, ImageDraw

from personalized_t2i.config import (
    EXPECTED_INFERENCE_SEEDS,
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
    base_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    adapter_path: str = None,
    resolved_config_path: str = None,
    lora_scale: float = 1.0,
):
    """Generate deterministic placeholder images for QA-01 CPU smoke testing."""

    print(f"[*] Starting evaluation generation for: {run_id}")

    artifact_dir = Path("artifacts") / run_id
    gen_dir = artifact_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Try to determine concept from resolved config.
    concept_id = None

    if resolved_config_path:
        config_path = Path(resolved_config_path)
    else:
        config_path = artifact_dir / "config.resolved.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        token = (
            config.get("data", {})
            .get("instance_token")
        )

        token_to_concept = {
            "zzobj01": "cat_mug",
            "zzobj02": "dog_plush",
            "zzobj03": "blue_white_vase",
        }

        concept_id = token_to_concept.get(token)

    prompts, resolved_concept_id = load_prompt_bank(
        concept_id=concept_id
    )

    seeds = load_seeds()

    metadata_records = []

    print(f"[*] Concept: {resolved_concept_id}")
    print(f"[*] Prompts: {len(prompts)}")
    print(f"[*] Seeds: {len(seeds)}")
    print(f"[*] Expected images: {len(prompts) * len(seeds)}")

    for prompt_index, prompt in enumerate(prompts):
        prompt_id = prompt["id"]
        prompt_text = prompt["text"]

        prompt_dir = gen_dir / str(prompt_id)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            # Deterministic simple colour pattern.
            r = (seed * 17 + prompt_index * 31) % 256
            g = (seed * 29 + prompt_index * 19) % 256
            b = (seed * 43 + prompt_index * 11) % 256

            image = Image.new("RGB", (512, 512), (r, g, b))
            draw = ImageDraw.Draw(image)

            draw.rectangle(
                (20, 20, 492, 492),
                outline=(255, 255, 255),
                width=4,
            )

            draw.text(
                (35, 35),
                f"QA-01 SMOKE\n{resolved_concept_id}\n"
                f"{prompt_id}\nseed={seed}",
                fill=(255, 255, 255),
            )

            image_path = prompt_dir / f"{seed}.png"
            image.save(image_path)

            metadata_records.append(
                {
                    "run_id": run_id,
                    "concept_id": resolved_concept_id,
                    "prompt_bank_version": PROMPT_BANK_VERSION,
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "seed": seed,
                    "lora_scale": lora_scale if adapter_path else 0.0,
                    "base_model_id": base_model_id,
                    "adapter_path": adapter_path,
                    "image_path": str(image_path),
                    "generation_mode": "cpu_smoke_placeholder",
                    "created_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )

    metadata_file = gen_dir / "metadata.jsonl"

    with open(metadata_file, "w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print("[OK] Evaluation generation completed.")
    print(f"[OK] Images generated: {len(metadata_records)}")
    print(f"[OK] Metadata: {metadata_file}")

    return str(metadata_file)
