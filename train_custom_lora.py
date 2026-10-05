import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms

# --- BƯỚC 1: CẤU HÌNH & LOAD BACKBONE (Kế thừa từ ML-01) ---
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe"
INSTANCE_DIR = "data/raw/dog_plush/v1/train_pool" # Đường dẫn từ DATA-02
INSTANCE_PROMPT = "a photo of zzobj02 plush toy"
OUTPUT_DIR = "outputs/custom_lora_dog_plush"

print("Đang tải mô hình gốc từ ML-01...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=torch.float16
).to("cuda")

# Đóng băng các thành phần không train để tiết kiệm VRAM
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)

# Lấy các thành phần chính
unet = pipe.unet
vae = pipe.vae
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# --- BƯỚC 2: CHÈN LORA BẰNG PEFT ---
lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none",
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# --- BƯỚC 3: TẠO DATASET TÙY CHỈNH ---
class ConceptDataset(Dataset):
    def __init__(self, image_dir, tokenizer, instance_prompt, size=512):
        self.image_paths = [
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.tokenizer = tokenizer
        self.prompt = instance_prompt
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # Tokenize prompt
        inputs = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image_tensor,
            "input_ids": inputs.input_ids[0]
        }

train_dataset = ConceptDataset(INSTANCE_DIR, tokenizer, INSTANCE_PROMPT)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# --- BƯỚC 4: TỐI ƯU HÓA & VÒNG LẶP TRAINING (TRAINING LOOP) ---
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=1e-4
)

unet.train()
EPOCHS = 100

print(f"\nBắt đầu tự huấn luyện LoRA với {len(train_dataset)} ảnh...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in train_dataloader:
        # Chuyển dữ liệu lên GPU và ép kiểu fp16
        images = batch["pixel_values"].to("cuda", dtype=torch.float16)
        input_ids = batch["input_ids"].to("cuda")

        # 1. Chuyển ảnh qua VAE để lấy latent z
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # 2. Tạo nhiễu ngẫu nhiên
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device="cuda").long()

        # 3. Thêm nhiễu vào latent theo lịch trình (forward diffusion)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 4. Lấy text embedding từ Prompt
        encoder_hidden_states = text_encoder(input_ids)[0]

        # 5. Dự đoán phần nhiễu bằng UNet (đã gắn LoRA)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 6. Tính MSE Loss giữa nhiễu dự đoán và nhiễu thực tế
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        # 7. Lan truyền ngược và cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss / len(train_dataloader):.4f}")

# --- BƯỚC 5: LƯU ADAPTER SAU KHI TRAIN ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
unet.save_pretrained(OUTPUT_DIR)
print(f"\nHuấn luyện thủ công hoàn tất! Đã lưu LoRA adapter tại: {OUTPUT_DIR}")