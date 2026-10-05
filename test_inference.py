import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

def run_inference():
    # Cấu hình thông tin run & seed để tái lập (reproducible)
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision = "451f4fe"
    adapter_path = "outputs/custom_lora_dog_plush"
    prompt = "A photo of zzobj02 plush toy in a modern room"
    seed = 42

    print(f"Đang nạp backbone từ {model_id} (revision: {revision})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32
    ).to("cuda")
    
    # Giữ an toàn theo chuẩn protocol (hoặc cấu hình lại nếu cần)
    # pipe.safety_checker = ... 

    print(f"Đang nạp custom LoRA adapter từ: {adapter_path}")
    pipe.load_lora_weights(adapter_path)

    # Cố định seed để tái lập kết quả sinh ảnh
    generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Đang sinh ảnh với prompt: '{prompt}' (seed: {seed})...")
    image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]

    # Assertion kiểm tra ảnh được sinh thành công và không trống
    assert image is not None, "Quá trình inference thất bại, không trả về ảnh."
    
    output_path = Path("custom_verification_output.png")
    image.save(output_path)
    print(f"Thành công! Đã lưu ảnh kiểm chứng tại '{output_path}'.")

if __name__ == "__main__":
    run_inference()


