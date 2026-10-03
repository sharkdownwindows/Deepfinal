import argparse
import os
import subprocess
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Config-driven LoRA Training Wrapper for Diffusers")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base.yaml", 
        help="Đường dẫn tới file cấu hình YAML"
    )
    args = parser.parse_args()

    # 1. Đọc config
    print(f"[*] Đang đọc file cấu hình từ: {args.config}")
    config = load_config(args.config)

    # Trích xuất thông số
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    pretrained_model = model_cfg.get("pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")
    train_data_dir = train_cfg.get("train_data_dir", "data/train_images")
    output_dir = output_cfg.get("output_dir", "artifacts/lora_output")
    resolution = train_cfg.get("resolution", 512)
    learning_rate = train_cfg.get("learning_rate", 1e-4)
    max_train_steps = train_cfg.get("max_train_steps", 500)
    train_batch_size = train_cfg.get("train_batch_size", 1)
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 4)
    mixed_precision = train_cfg.get("mixed_precision", "fp16")
    save_steps = train_cfg.get("save_steps", 100)

    # 2. Đảm bảo thư mục output tồn tại và lưu resolved config (đúng Acceptance Criteria)
    os.makedirs(output_dir, exist_ok=True)
    resolved_config_path = Path(output_dir) / "resolved_config.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f"[*] Đã lưu resolved config tại: {resolved_config_path}")

    # 3. Chuẩn bị câu lệnh gọi script train chính thức của Diffusers
    # Lưu ý: Script này thường nằm trong kho chứa examples của diffusers hoặc cài đặt qua môi trường.
    # Đảm bảo bạn đã clone thư mục diffusers hoặc sử dụng script train_text_to_image_lora.py chuẩn.
    cmd = [
        "accelerate", "launch", "train_text_to_image_lora.py",
        f"--pretrained_model_name_or_path={pretrained_model}",
        f"--train_data_dir={train_data_dir}",
        f"--output_dir={output_dir}",
        f"--resolution={resolution}",
        f"--learning_rate={learning_rate}",
        f"--max_train_steps={max_train_steps}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--mixed_precision={mixed_precision}",
        f"--save_steps={save_steps}",
        "--image_column=image",
        "--caption_column=text",
        "--adam_weight_decay=1e-2",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--seed=42"
    ]

    print("[*] Đang khởi động quá trình huấn luyện LoRA với câu lệnh:")
    print(" ".join(cmd))

    # 4. Thực thi lệnh
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[+] Huấn luyện thành công! Artifacts được lưu tại: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Lỗi trong quá trình huấn luyện: {e}")
        raise e

if __name__ == "__main__":
    main()