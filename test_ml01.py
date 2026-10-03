import torch
from diffusers import StableDiffusionPipeline

print("--- KIỂM TRA MÔI TRƯỜNG CHO ML-01 ---")
print("PyTorch version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    
    # Tải mô hình nền Stable Diffusion v1.5
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"\nĐang thử tải mô hình {model_id}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    print("Tải mô hình thành công và đưa vào VRAM GPU hoàn tất!")
    
    # Test sinh một ảnh nhỏ để kiểm tra thông lượng bộ nhớ
    prompt = "A cute cat wearing a spacesuit, digital art"
    print(f"Đang thử nghiệm sinh ảnh với prompt: '{prompt}'...")
    image = pipe(prompt, num_inference_steps=15).images[0]
    print("Sinh ảnh thử nghiệm thành công mà không gặp lỗi tràn bộ nhớ (OOM)!")
else:
    print("Cảnh báo: Không tìm thấy GPU, vui lòng kiểm tra lại cấu hình CUDA!")
    image.save("astronaut_cat.png")
    print("Đã lưu ảnh thành công vào file 'astronaut_cat.png'!")