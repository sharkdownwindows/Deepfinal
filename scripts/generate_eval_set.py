import argparse
from personalized_t2i.inference.generate import generate_evaluation_batch

def main():
    parser = argparse.ArgumentParser(description="Batch generate evaluation images for a LoRA run or baseline.")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier (e.g., toy01_n5_r16_ts42)")
    parser.add_argument("--base-model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Hugging Face model ID")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to directory containing adapter weights (safetensors)")
    parser.add_argument("--resolved-config", type=str, default=None, help="Path to resolved YAML config file")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA mixing scale")
    
    args = parser.parse_args()

    generate_evaluation_batch(
        run_id=args.run_id,
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        resolved_config_path=args.resolved_config,
        lora_scale=args.lora_scale
    )

if __name__ == "__main__":
    main()