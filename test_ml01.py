import time
import torch
from diffusers import StableDiffusionPipeline

# 1. Bắt đầu bấm giờ (Wall time)
start_time = time.time()

print("--- KIỂM TRA MÔI TRƯỜNG CHO ML-01 ---")
print("PyTorch version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())

# Khởi tạo biến image mặc định
image = None
BACKBONE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BACKBONE_REVISION = "main" # Pin model revision để đảm bảo tính ổn định

print(f"\n[BACKBONE DECISION] Đã chốt backbone: {BACKBONE_MODEL_ID}")
print(f"[MODEL REVISION] Đã pin revision: {BACKBONE_REVISION}")
print(f"Lý do: SDXL bị loại bỏ do không đạt gate phần cứng VRAM hiện tại.\n")

if torch.cuda.is_available():
    # Reset thống kê VRAM đỉnh trước khi chạy
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    print("GPU Device:", torch.cuda.get_device_name(0))

    # Tải mô hình nền Stable Diffusion v1.5
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"\nĐang thử tải mô hình {model_id}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=BACKBONE_REVISION,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    print("Tải mô hình thành công và đưa vào VRAM GPU hoàn tất!")

    # Test sinh một ảnh nhỏ để kiểm tra thông lượng bộ nhớ
    prompt = "A cute cat wearing a spacesuit, digital art"
    print(f"Đang thử nghiệm sinh ảnh với prompt: '{prompt}'...")
    
    image = pipe(prompt, num_inference_steps=15).images[0]
    print("Sinh ảnh thử nghiệm thành công mà không gặp lỗi tràn bộ nhớ (OOM)!")
    
    # Lưu lại ảnh test
    image.save("astronaut_cat.png")
    print("Đã lưu ảnh thành công vào file 'astronaut_cat.png'!")
else:
    print("Cảnh báo: Không tìm thấy GPU, vui lòng kiểm tra lại cấu hình CUDA!")

# 2. Tính toán thời gian và Peak VRAM
end_time = time.time()
wall_time = end_time - start_time

print("\n--- BÁO CÁO HIỆU NĂNG (ML-01) ---")
print(f"Tổng thời gian chạy (Wall time): {wall_time:.2f} giây")

if torch.cuda.is_available():
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024 * 1024)
    print(f"Mức tiêu thụ VRAM đỉnh (Peak VRAM): {peak_vram_mb:.2f} MB")
else:
    print("Chạy trên CPU (Không đo được VRAM).")
    
   