"""Thin single-run training entry point."""

import argparse
import os
import subprocess
import sys
import json
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

    # 1. Đọc file config gốc theo đúng schema lồng nhau
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Lấy thông số chuẩn theo nested schema
    run_cfg = config.get("run", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    run_id = run_cfg.get("id", "default_run")
    artifact_dir = Path(output_cfg.get("output_dir", f"artifacts/{run_id}"))
    
    # 2. Chống overwrite nghiêm ngặt: Chặn mọi run đã tồn tại (không chỉ mỗi completed)
    status_path = artifact_dir / "status.json"
    if artifact_dir.exists() and status_path.exists():
        print(f"Cảnh báo: Run ID '{run_id}' đã tồn tại thư mục và trạng thái. Từ chối ghi đè.")
        sys.exit(1)

    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 3. Lưu file resolved config vào artifacts//config.resolved.yaml
    resolved_config_path = artifact_dir / "config.resolved.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Đã lưu resolved config tại: {resolved_config_path}")

    # 4. Ghi trạng thái ban đầu là running
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump({"status": "running", "run_id": run_id}, f)

    try:
        print(f"Đang thực thi Official Diffusers Trainer cho run_id: {run_id}...")
        
        output_dir = artifact_dir / "checkpoint"
        
        # Xây dựng câu lệnh gọi Official Diffusers DreamBooth-LoRA trainer (ĐÃ XÓA --push_to_hub=False)
        cmd = [
            "accelerate", "launch", "examples/dreambooth/train_dreambooth_lora.py",
            "--pretrained_model_name_or_path", model_cfg.get('id', 'stable-diffusion-v1-5/stable-diffusion-v1-5'),
            "--revision", model_cfg.get('revision', '451f4fe'),
            "--instance_data_dir", data_cfg.get('train_data_dir', 'data/raw/dog_plush/v1/train_pool/'),
            "--instance_prompt", data_cfg.get('instance_prompt', 'a photo of zzobj02 plush toy'),
            "--output_dir", str(output_dir),
            "--train_text_encoder",
            "--resolution", str(train_cfg.get('resolution', 512)),
            "--train_batch_size", str(train_cfg.get('batch_size', 1)),
            "--gradient_accumulation_steps", "1",
            "--learning_rate", str(train_cfg.get('learning_rate', 1e-4)),
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--max_train_steps", str(train_cfg.get('max_train_steps', 500)),
            "--checkpointing_steps", str(train_cfg.get('checkpointing_steps', 100)),
            "--seed", str(train_cfg.get('seed', 42))
        ]
        
        # 5. Thực thi và capture stdout/stderr vào file log riêng
        log_path = artifact_dir / "train.log"
        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            process.wait()

        # Kiểm tra xem checkpoint thực tế có được sinh ra không
        if process.returncode == 0 and output_dir.exists():
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump({"status": "completed", "run_id": run_id}, f)
            print("Training thành công và đã lưu artifacts đầy đủ!")
        else:
            raise RuntimeError("Tiến trình huấn luyện kết thúc nhưng không tạo được checkpoint hợp lệ.")
        
    except Exception as e:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"status": "failed", "error": str(e)}, f)
        print(f"Training thất bại. Đã lưu trạng thái lỗi vào {status_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()