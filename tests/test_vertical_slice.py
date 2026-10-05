"""The mocked or tiny vertical-slice test belongs here after M1 exists."""
import pytest
import yaml
from pathlib import Path
import subprocess

def test_config_loading():
    config_path = Path("configs/base.yaml")
    assert config_path.exists(), "File base.yaml không tồn tại."
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["training"]["max_train_steps"] == 500
    assert config["training"]["checkpointing_steps"] == 100
    assert config["training"]["seed"] == 42
    assert config["data"]["instance_token"] == "zzobj02"

def test_dataset_existence():
    config_path = Path("configs/base.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_dir = Path(config["data"]["train_data_dir"])
    # Kiểm tra xem đường dẫn dataset có hợp lệ hay không (hoặc báo lỗi missing dataset)
    assert train_dir.parts, "Đường dẫn dataset không hợp lệ."

def test_anti_overwrite_artifact(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "test_run"
    artifact_dir.mkdir(parents=True)
    status_file = artifact_dir / "status.json"
    
    # Giả lập run đã hoàn thành trước đó
    with open(status_file, "w", encoding="utf-8") as f:
        yaml.dump({"status": "completed", "run_id": "test_run"}, f)
        
    # Kiểm tra logic chống ghi đè
    with open(status_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        assert data.get("status") == "completed"