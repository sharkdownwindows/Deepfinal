"""Evaluation image generation module (ML-03)."""
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

def load_prompt_bank(prompt_path: str = "prompt_bank/evaluation_prompts.yaml", concept_id: str = None):
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    concepts = data.get("concepts", {})
    if not concepts:
        raise ValueError(f"No concepts found in prompt bank: {prompt_path}")
    
    # Nếu không truyền concept_id cụ thể, mặc định lấy concept đầu tiên trong file yaml
    if not concept_id or concept_id not in concepts:
        concept_id = list(concepts.keys())[0]
        print(f"[*] concept_id not specified or found, defaulting to: {concept_id}")
        
    raw_prompts = concepts[concept_id]
    
    # Chuẩn hóa lại key để khớp với logic xử lý bên dưới (id, category, text)
    formatted_prompts = []
    for p in raw_prompts:
        formatted_prompts.append({
            "id": p.get("prompt_id"),
            "category": p.get("category"),
            "text": p.get("prompt")
        })
    return formatted_prompts, concept_id

def load_seeds(seed_path: str = "prompt_bank/generation_seeds.yaml"):
    with open(seed_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("seeds", [11, 22, 33, 44])

def generate_evaluation_batch(
    run_id: str,
    base_model_id: str,
    adapter_path: str = None,  # None nếu chạy base-model baseline
    resolved_config_path: str = None,
    output_dir: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lora_scale: float = 1.0
):
    """
    Thực hiện sinh batch ảnh đánh giá cho một run cụ thể hoặc cho base model baseline.
    Đáp ứng yêu cầu FR-05, FR-06 và US-04 trong tài liệu PRD.
    """
    print(f"[*] Starting batch generation for run_id: {run_id}")
    
    # 1. Xác định đường dẫn artifact
    if output_dir is None:
        output_dir = Path(f"artifacts/{run_id}")
    else:
        output_dir = Path(output_dir)
        
    gen_dir = output_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "metadata.jsonl"

    # 2. Load cấu hình đã resolve nếu có để lấy thông tin concept/version
    concept_id = None
    dataset_version = "v1"
    base_model_revision = "main"
    if resolved_config_path and os.path.exists(resolved_config_path):
        with open(resolved_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            concept_id = cfg.get("data", {}).get("concept_id", concept_id)
            dataset_version = cfg.get("data", {}).get("dataset_version", dataset_version)
            base_model_revision = cfg.get("model", {}).get("revision", base_model_revision)

    # 3. Khởi tạo pipeline Stable Diffusion v1.5
    print(f"[*] Loading base model: {base_model_id} (revision: {base_model_revision})")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        revision=base_model_revision,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    
    # 4. Nạp LoRA adapter nếu không phải là base baseline
    if adapter_path and os.path.exists(adapter_path):
        print(f"[*] Loading LoRA adapter from: {adapter_path}")
        pipeline.load_lora_weights(adapter_path)
    else:
        print("[*] Running as Base Model Baseline (No LoRA attached)")

    pipeline.to(device)
    pipeline.enable_attention_slicing()

    # 5. Load prompts (tự động khớp với concept_id) và seeds
    prompts, resolved_concept_id = load_prompt_bank(concept_id=concept_id)
    seeds = load_seeds()

    metadata_records = []
    inference_steps = 30
    guidance_scale = 7.5
    width, height = 512, 512

    # 6. Vòng lặp sinh ảnh (Prompt x Seed matrix)
    for p in prompts:
        prompt_id = p["id"]
        prompt_text = p["text"]
        
        prompt_out_dir = gen_dir / prompt_id
        prompt_out_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            # Tạo generator riêng biệt cho từng sample để đảm bảo tính độc lập & tái lập
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # Thực hiện inference
            with torch.autocast(device) if device == "cuda" else nullcontext():
                image = pipeline(
                    prompt=prompt_text,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    cross_attention_kwargs={"scale": lora_scale} if adapter_path else None
                ).images[0]

            # Lưu file ảnh theo định dạng chuẩn: .png
            image_filename = f"{seed}.png"
            image_path = prompt_out_dir / image_filename
            image.save(image_path)

            # Xây dựng record metadata theo đúng chuẩn contract
            record = {
                "run_id": run_id,
                "concept_id": resolved_concept_id,
                "dataset_version": dataset_version,
                "checkpoint_step": 500,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "seed": seed,
                "lora_scale": lora_scale if adapter_path else 0.0,
                "base_model_revision": base_model_revision,
                "image_path": str(image_path),
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
            metadata_records.append(record)

    # 7. Ghi file metadata.jsonl
    with open(metadata_file, "w", encoding="utf-8") as mf:
        for rec in metadata_records:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[✓] Batch generation completed successfully for {run_id}. Total images: {len(metadata_records)}")
    print(f"[✓] Metadata saved to: {metadata_file}")
    return metadata_file