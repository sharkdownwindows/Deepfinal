import os
import yaml
from pathlib import Path
import subprocess

def main():
    # Danh sách các concepts (ví dụ bạn có 3 concepts)
    concepts = [
        {"name": "dog_plush", "token": "zzobj01", "prompt": "a photo of zzobj01 plush toy"}
        
    ]
    
    # Các mức data size theo yêu cầu ML-04
    data_sizes = [1, 3, 5, 10]
    
    config_dir = Path("configs/data_sweep")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for concept in concepts:
        c_name = concept["name"]
        token = concept["token"]
        prompt = concept["prompt"]
        
        for n in data_sizes:
            run_id = f"{c_name}_n{n}_r16"
            output_dir = f"artifacts/{run_id}"
            train_data_dir = f"data/raw/{c_name}/v1_n{n}/train_pool/"
            
            # Cấu hình YAML cho từng run
            sweep_config = {
                "run": {"id": run_id},
                "model": {
                    "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "revision": "451f4fe"
                },
                "training": {
                    "learning_rate": 0.0001,
                    "max_train_steps": 500,
                    "checkpointing_steps": 100,
                    "resolution": 512,
                    "batch_size": 1,
                    "seed": 42,
                    "rank": 16,
                    "alpha": 16
                },
                "data": {
                    "dataset_name": "custom",
                    "train_data_dir": train_data_dir,
                    "instance_prompt": prompt,
                    "instance_token": token
                },
                "output": {
                    "output_dir": output_dir
                }
            }
            
            # Ghi file config ra thư mục configs/data_sweep/
            config_file_path = config_dir / f"{run_id}.yaml"
            with open(config_file_path, "w", encoding="utf-8") as f:
                yaml.dump(sweep_config, f, default_flow_style=False)
                
            print(f"Đã tạo config: {config_file_path}")
            
            # Tự động gọi train_run.py cho từng config vừa sinh
            cmd = ["venv/Scripts/python.exe", "scripts/train_run.py", "--config", str(config_file_path)]
            print(f"Đang thực thi: {' '.join(cmd)}")
            
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Cảnh báo: Run {run_id} gặp lỗi hoặc bị dừng.")
            else:
                print(f"Hoàn tất thành công run: {run_id}\n")

if __name__ == "__main__":
    main()
