import torch
from diffusers import StableDiffusionPipeline

# 1. Load backbone chuẩn từ ML-01
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    revision="451f4fe",
    torch_dtype=torch.float32
).to("cuda")
pipe.safety_checker = None

# 2. Nạp file trọng số LoRA custom vừa train
print("Đang nạp custom LoRA adapter...")
pipe.load_lora_weights("outputs/custom_lora_dog_plush")

# 3. Sinh ảnh kiểm chứng với từ khóa định danh
prompt = "A photo of sks dog_plush in a modern room"
image = pipe(prompt, num_inference_steps=25).images[0]

image.save("custom_verification_output.png")
print("Thành công! Đã load adapter thủ công và lưu ảnh tại 'custom_verification_output.png'.")
