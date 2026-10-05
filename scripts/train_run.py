"""Thin single-run training entry point; implementation is pending."""

import argparse
import os
import subprocess
import yaml
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run DreamBooth-LoRA training pipeline via config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config tại: {config_path}")

    # 1. Đọc file config gốc
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Thiết lập run_id và thư mục artifacts theo contract chuẩn
    run_id = config.get("run_id", "default_run")
    artifact_dir = Path(f"artifacts/{run_id}")
    
    # Chống overwrite: Nếu run đã hoàn thành, từ chối ghi đè để bảo vệ evidence
    status_path = artifact_dir / "status.json"
    if artifact_dir.exists() and status_path.exists():
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = yaml.safe_load(f)
            if status_data.get("status") == "completed":
                print(f"Cảnh báo: Run ID '{run_id}' đã hoàn thành trước đó. Từ chối ghi đè.")
                return

    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 3. Lưu file resolved config vào artifacts//config.resolved.yaml
    resolved_config_path = artifact_dir / "config.resolved.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Đã lưu resolved config tại: {resolved_config_path}")

    # 4. Ghi trạng thái ban đầu là running
    with open(status_path, "w", encoding="utf-8") as f:
        yaml.dump({"status": "running", "run_id": run_id}, f)

    try:
        print(f"Đang thực thi Official Diffusers Trainer cho run_id: {run_id} theo chuẩn manifest...")
        
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        
        output_dir = artifact_dir / "checkpoint"
        
        # Xây dựng câu lệnh gọi Official Diffusers DreamBooth-LoRA trainer
        cmd = [
            "accelerate", "launch", "examples/dreambooth/train_dreambooth_lora.py",
            f"--pretrained_model_name_or_path={model_cfg.get('pretrained_model_name_or_path', 'stable-diffusion-v1-5/stable-diffusion-v1-5')}",
            f"--revision={model_cfg.get('revision', '451f4fe')}",
            f"--instance_data_dir={data_cfg.get('train_data_dir', 'data/processed/v1/train_pool/')}",
            f"--instance_prompt={data_cfg.get('instance_prompt', 'a photo of zzobj02 plush toy')}",
            f"--output_dir={output_dir}",
            f"--resolution={train_cfg.get('resolution', 512)}",
            f"--train_batch_size={train_cfg.get('train_batch_size', 1)}",
            f"--gradient_accumulation_steps=1",
            f"--learning_rate={train_cfg.get('learning_rate', 1e-4)}",
            f"--lr_scheduler=constant",
            f"--lr_warmup_steps=0",
            f"--max_train_steps={train_cfg.get('max_train_steps', 500)}",
            f"--checkpointing_steps={train_cfg.get('checkpointing_steps', 250)}",
            f"--seed={train_cfg.get('seed', 42)}",
            "--push_to_hub=False"
        ]
        
        # Thực thi lệnh train thông qua subprocess
        subprocess.run(cmd, check=True)
        
        # Đánh dấu hoàn thành
        with open(status_path, "w", encoding="utf-8") as f:
            yaml.dump({"status": "completed", "run_id": run_id}, f)
        print("Training thành công và đã lưu artifacts đầy đủ!")
        
    except Exception as e:
        # Nếu lỗi ngắt quãng, bắt buộc giữ lại failed status/log
        with open(status_path, "w", encoding="utf-8") as f:
            yaml.dump({"status": "failed", "error": str(e)}, f)
        print(f"Training thất bại. Đã lưu trạng thái lỗi vào {status_path}")
        raise e

if __name__ == "__main__":
    main()