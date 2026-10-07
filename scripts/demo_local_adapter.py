"""Generate one local SD 1.5 sample with or without the completed LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DEFAULT_ADAPTER = Path(
    r"B:\lora-local\runs\dog_plush_n5_r4_256_ts42\pytorch_lora_weights.safetensors"
)
DEFAULT_OUTPUT_DIR = Path(r"B:\lora-local\demo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a same-prompt/same-seed base or LoRA SD 1.5 demo."
    )
    parser.add_argument(
        "--adapter", type=Path, default=DEFAULT_ADAPTER,
        help="Path to pytorch_lora_weights.safetensors.",
    )
    parser.add_argument(
        "--base-only", action="store_true",
        help="Generate the base-model comparison without loading the adapter.",
    )
    parser.add_argument(
        "--prompt", default="a photo of zzobj02 plush toy sitting on a wooden table",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resolution < 64 or args.resolution % 8:
        parser.error("--resolution must be at least 64 and divisible by 8")
    if args.steps < 1:
        parser.error("--steps must be positive")
    return args


def load_base_pipeline() -> StableDiffusionPipeline:
    """Load from the complete pipeline snapshot, or assemble cached components."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot = (
        hf_home
        / "hub"
        / "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
        / "snapshots"
        / MODEL_REVISION
    )
    model_index = snapshot / "model_index.json"
    if model_index.is_file():
        return StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )

    required = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]
    missing = [name for name in required if not (snapshot / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Model cache is incomplete at {snapshot}; missing components: {missing}. "
            "Check HF_HOME and the model revision."
        )

    # Training cached these components individually; a root model_index.json was
    # never needed by the trainer. The optional safety-checker weights are absent.
    print("Note: safety checker is disabled; its component is absent from this cache.")
    scheduler = DDPMScheduler.from_pretrained(
        snapshot / "scheduler", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        snapshot / "text_encoder",
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        snapshot / "tokenizer", local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        snapshot / "unet",
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae",
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    return StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the local Conda environment")
    if not args.base_only and not args.adapter.is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {args.adapter}")

    mode = "base" if args.base_only else "lora"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"dog_plush_{mode}_seed{args.seed}.png"
    metadata_path = image_path.with_suffix(".json")
    if not args.overwrite and (image_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Demo output exists: {image_path}. Choose another seed/path or pass --overwrite."
        )

    pipe = load_base_pipeline()
    if not args.base_only:
        pipe.load_lora_weights(str(args.adapter.parent), weight_name=args.adapter.name)

    # Keep model weights on CPU until needed; this fits a 4 GB GPU better.
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=False)

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    image = result.images[0]
    image.save(image_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "safety_checker_enabled": False,
        "adapter_path": str(args.adapter.resolve()) if not args.base_only else None,
        "adapter_sha256": sha256_file(args.adapter) if not args.base_only else None,
        "prompt": args.prompt,
        "seed": args.seed,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "python": sys.version,
        "torch": torch.__version__,
        "diffusers": __import__("diffusers").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "image_path": str(image_path.resolve()),
        "image_sha256": sha256_file(image_path),
        "platform": platform.platform(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Image: {image_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Peak allocated VRAM: {metadata['peak_allocated_vram_bytes'] / 1024**3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
