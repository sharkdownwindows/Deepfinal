from safetensors import safe_open
import os

# Thay đổi đường dẫn này trỏ tới file .safetensors của bạn
# (Ví dụ: pytorch_lora_weights.safetensors hoặc model.safetensors)
safetensors_path = "artifacts/dog_plush_run_01/checkpoint/adapter_model.safetensors"

if not os.path.exists(safetensors_path):
    # Tìm tự động file safetensors trong thư mục nếu bạn chưa rõ tên file chính xác
    folder = "artifacts/dog_plush_run_01/checkpoint"
    for f in os.listdir(folder):
        if f.endswith(".safetensors"):
            safetensors_path = os.path.join(folder, f)
            break

print(f"đang đọc file: {safetensors_path}")

try:
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"\n Tổng số keys: {len(keys)}")
        print("\n 10 key đầu tiên trong file:")
        for k in keys[:10]:
            print(f"  - {k}")
            
        # Kiểm tra xem có chứa tiền tố 'unet.' hay không
        has_unet_prefix = any(k.startswith("unet.") for k in keys)
        print(f"\n Các key có bắt đầu bằng chữ 'unet.' không? {'CÓ' if has_unet_prefix else 'KHÔNG (chỉ có các block trực tiếp)'}")
        
except Exception as e:
    print(f" Lỗi khi đọc file: {e}")