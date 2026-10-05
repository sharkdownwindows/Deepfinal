import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Cấu hình chuẩn giống hệt file hôm qua của bạn
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
revision = "451f4fe"
adapter_path = "artifacts/dog_plush_run_01/checkpoint"
prompt = "A photo of zzobj02 plush toy in a modern room"
seed = 42

print(f"Đang nạp backbone từ {model_id}...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.float32  # Dùng float32 để không bao giờ bị lỗi ảnh đen
).to("cuda")

print(f"Đang nạp custom LoRA adapter từ: {adapter_path}")
pipe.load_lora_weights(adapter_path)

generator = torch.Generator(device="cuda").manual_seed(seed)

print(f"Đang sinh ảnh với prompt: '{prompt}'...")
image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]

output_path = Path("test_single.png")
image.save(output_path)
print(f"Thành công! Đã lưu ảnh tại '{output_path}'.")