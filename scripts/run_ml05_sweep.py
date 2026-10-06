import os
import yaml
from pathlib import Path
import subprocess
import time
import json

def main():
    # Chỉ định concept duy nhất
    concepts = [
        {"name": "dog_plush", "token": "zzobj01", "prompt": "a photo of zzobj01 plush toy"}
    ]
    
    # Các mức rank theo yêu cầu ML-05, cố định n = 5
    ranks = [4, 16, 32]
    fixed_n = 5
    
    config_dir = Path("configs/ml05_sweep")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_results = []

    for concept in concepts:
        c_name = concept["name"]
        token = concept["token"]
        prompt = concept["prompt"]
        
        for r in ranks:
            run_id = f"{c_name}_n{fixed_n}_r{r}"
            output_dir = Path(f"artifacts/{run_id}")
            status_path = output_dir / "status.json"
            
            # Tiêu chí ML-05: Không chạy lại nếu n5-r16 (hoặc run tương ứng) đã hoàn tất
            if output_dir.exists() and status_path.exists():
                with open(status_path, "r", encoding="utf-8") as sf:
                    try:
                        st_data = json.load(sf)
                        if st_data.get("status") == "completed":
                            print(f"Bỏ qua '{run_id}' vì đã hoàn thành trước đó (Giữ nguyên kết quả cũ).")
                            
                            safetensors_file = output_dir / "checkpoint" / "pytorch_lora_weights.safetensors"
                            adapter_size_mb = round(safetensors_file.stat().st_size / (1024 * 1024), 2) if safetensors_file.exists() else 0
                            
                            sweep_results.append({
                                "run_id": run_id,
                                "concept": c_name,
                                "rank": r,
                                "n": fixed_n,
                                "time_seconds": 0,
                                "adapter_size_mb": adapter_size_mb,
                                "success": True,
                                "skipped": True
                            })
                            continue
                    except Exception:
                        pass

            train_data_dir = f"data/raw/{c_name}/v1_n{fixed_n}/train_pool/"
            
            # Cấu hình YAML cho từng mức rank
            sweep_config = {
                "run": {"id": run_id},
                "model": {
                    "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "revision": "451f4fe"
                },
                "training": {
                    "learning_rate": 0.0001,
                    "max_train_steps": 1,
                    "checkpointing_steps": 100,
                    "resolution": 512,
                    "batch_size": 1,
                    "seed": 42,
                    "rank": r,
                    "alpha": r
                },
                "data": {
                    "dataset_name": "custom",
                    "train_data_dir": train_data_dir,
                    "instance_prompt": prompt,
                    "instance_token": token
                },
                "output": {
                    "output_dir": str(output_dir)
                }
            }
            
            config_file_path = config_dir / f"{run_id}.yaml"
            with open(config_file_path, "w", encoding="utf-8") as f:
                yaml.dump(sweep_config, f, default_flow_style=False)
                
            print(f"\n--- Bắt đầu chạy: {run_id} ---")
            cmd = ["venv/Scripts/python.exe", "scripts/train_run.py", "--config", str(config_file_path)]
            
            start_time = time.time()
            result = subprocess.run(cmd)
            elapsed_time = time.time() - start_time
            
            # Đo lường thời gian và dung lượng adapter (.safetensors)
            adapter_size_mb = 0
            checkpoint_dir = output_dir / "checkpoint"
            safetensors_file = checkpoint_dir / "pytorch_lora_weights.safetensors"
            
            if result.returncode == 0 and safetensors_file.exists():
                adapter_size_mb = round(safetensors_file.stat().st_size / (1024 * 1024), 2)
                print(f"Thành công! Thời gian: {elapsed_time:.2f}s | Kích thước adapter: {adapter_size_mb} MB")
            else:
                print(f"Cảnh báo: Run {run_id} gặp lỗi.")

            sweep_results.append({
                "run_id": run_id,
                "concept": c_name,
                "rank": r,
                "n": fixed_n,
                "time_seconds": round(elapsed_time, 2),
                "adapter_size_mb": adapter_size_mb,
                "success": result.returncode == 0,
                "skipped": False
            })

    # Xuất báo cáo tổng kết ra file JSON để nghiệm thu task ML-05
    summary_path = Path("artifacts/ml05_summary_report.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(sweep_results, sf, indent=4, ensure_ascii=False)
    print(f"\nĐã xuất báo cáo tổng kết ML-05 tại: {summary_path}")

if __name__ == "__main__":
    main()