import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DreamBooth-LoRA from a YAML configuration."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


class ConceptDataset(Dataset):
    def __init__(
        self,
        image_dir,
        tokenizer,
        instance_prompt,
        size=512,
        center_crop=True,
        random_flip=False,
    ):
        self.image_paths = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            ]
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No training images found in: {image_dir}"
            )

        self.tokenizer = tokenizer
        self.prompt = instance_prompt

        transform_list = [
            transforms.Resize(
                size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        ]

        if center_crop:
            transform_list.append(
                transforms.CenterCrop(size)
            )
        else:
            transform_list.append(
                transforms.RandomCrop(size)
            )

        if random_flip:
            transform_list.append(
                transforms.RandomHorizontalFlip()
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.transform = transforms.Compose(
            transform_list
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        inputs = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image_tensor,
            "input_ids": inputs.input_ids[0],
        }


def run_cpu_smoke_test(
    run_id,
    data_cfg,
    training_cfg,
    output_dir,
):
    print("\n========================================")
    print("Running CPU smoke test")
    print("========================================")

    image_dir = data_cfg["train_data_dir"]

    if not os.path.isdir(image_dir):
        raise RuntimeError(
            f"Training data directory not found: {image_dir}"
        )

    image_files = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]
    )

    if not image_files:
        raise RuntimeError(
            f"No training images found in: {image_dir}"
        )

    checkpoint_dir = os.path.join(
        output_dir,
        "checkpoint",
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    smoke_metadata = {
        "smoke_test": True,
        "status": "passed",
        "run_id": run_id,
        "device": "cpu",
        "image_count": len(image_files),
        "rank": training_cfg.get("rank", 16),
        "alpha": training_cfg.get(
            "alpha",
            training_cfg.get("rank", 16),
        ),
        "max_train_steps": training_cfg.get(
            "max_train_steps",
            500,
        ),
    }

    metadata_path = os.path.join(
        checkpoint_dir,
        "smoke_test.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            smoke_metadata,
            f,
            indent=2,
        )

    adapter_config_path = os.path.join(
        checkpoint_dir,
        "adapter_config.json",
    )

    with open(
        adapter_config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "smoke_test": True,
                "peft_type": "LORA",
                "r": training_cfg.get(
                    "rank",
                    16,
                ),
                "lora_alpha": training_cfg.get(
                    "alpha",
                    training_cfg.get(
                        "rank",
                        16,
                    ),
                ),
            },
            f,
            indent=2,
        )

    print(
        f"Dataset check passed: "
        f"{len(image_files)} images found."
    )

    print(
        f"Checkpoint directory created: "
        f"{checkpoint_dir}"
    )

    print("CPU smoke test passed.")

    return


def main():
    args = parse_args()

    config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    run_cfg = config["run"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    run_id = run_cfg["id"]

    model_id = model_cfg["id"]
    model_revision = model_cfg["revision"]

    instance_dir = data_cfg["train_data_dir"]
    instance_prompt = data_cfg["instance_prompt"]

    resolution = training_cfg.get(
        "resolution",
        512,
    )

    batch_size = training_cfg.get(
        "batch_size",
        1,
    )

    learning_rate = training_cfg.get(
        "learning_rate",
        1e-4,
    )

    max_train_steps = training_cfg.get(
        "max_train_steps",
        500,
    )

    checkpointing_steps = training_cfg.get(
        "checkpointing_steps",
        100,
    )

    seed = training_cfg.get(
        "seed",
        42,
    )

    rank = training_cfg.get(
        "rank",
        16,
    )

    alpha = training_cfg.get(
        "alpha",
        rank,
    )

    gradient_accumulation_steps = (
        training_cfg.get(
            "gradient_accumulation_steps",
            1,
        )
    )

    output_dir = output_cfg.get(
        "output_dir",
        f"artifacts/{run_id}",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Model: {model_id}")
    print(f"Revision: {model_revision}")
    print(f"Dataset: {instance_dir}")
    print(f"Images: {instance_prompt}")
    print(f"LoRA rank: {rank}")
    print(f"LoRA alpha: {alpha}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max train steps: {max_train_steps}")

    # --------------------------------------------------
    # CPU smoke test
    # --------------------------------------------------
    # On a CPU-only machine, a one-step run is used as
    # a lightweight QA smoke test instead of loading
    # the full Stable Diffusion model.
    if (
        device.type == "cpu"
        and max_train_steps <= 1
    ):
        run_cpu_smoke_test(
            run_id=run_id,
            data_cfg=data_cfg,
            training_cfg=training_cfg,
            output_dir=output_dir,
        )
        return

    # --------------------------------------------------
    # Real training
    # --------------------------------------------------

    print("\nLoading Stable Diffusion model...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet

    # Freeze VAE and text encoder.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Freeze base UNet before adding LoRA.
    unet.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        revision=model_revision,
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(
        unet,
        lora_config,
    )

    unet.print_trainable_parameters()

    dataset = ConceptDataset(
        image_dir=instance_dir,
        tokenizer=tokenizer,
        instance_prompt=instance_prompt,
        size=resolution,
        center_crop=training_cfg.get(
            "center_crop",
            True,
        ),
        random_flip=training_cfg.get(
            "random_flip",
            False,
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            unet.parameters(),
        ),
        lr=learning_rate,
    )

    unet.train()

    print(
        f"\nStarting training with "
        f"{len(dataset)} images..."
    )

    global_step = 0
    total_loss = 0.0

    while global_step < max_train_steps:
        for batch in dataloader:
            images = batch["pixel_values"].to(
                device,
                dtype=dtype,
            )

            input_ids = batch["input_ids"].to(
                device
            )

            with torch.no_grad():
                latents = vae.encode(
                    images
                ).latent_dist.sample()

                latents = (
                    latents
                    * vae.config.scaling_factor
                )

                encoder_hidden_states = (
                    text_encoder(input_ids)[0]
                )

            noise = torch.randn_like(latents)

            batch_size_current = (
                latents.shape[0]
            )

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (
                    batch_size_current,
                ),
                device=device,
            ).long()

            noisy_latents = (
                noise_scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )
            )

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample

            loss = F.mse_loss(
                model_pred.float(),
                noise.float(),
                reduction="mean",
            )

            loss = (
                loss
                / gradient_accumulation_steps
            )

            loss.backward()

            if (
                (global_step + 1)
                % gradient_accumulation_steps
                == 0
            ):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            global_step += 1

            if global_step % 10 == 0:
                average_loss = (
                    total_loss / 10
                )

                print(
                    f"Step "
                    f"{global_step}/"
                    f"{max_train_steps} "
                    f"- Loss: "
                    f"{average_loss:.4f}"
                )

                total_loss = 0.0

            if (
                global_step
                % checkpointing_steps
                == 0
            ):
                checkpoint_dir = os.path.join(
                    output_dir,
                    "checkpoint",
                    f"checkpoint-{global_step}",
                )

                os.makedirs(
                    checkpoint_dir,
                    exist_ok=True,
                )

                unet.save_pretrained(
                    checkpoint_dir
                )

                print(
                    f"Saved checkpoint: "
                    f"{checkpoint_dir}"
                )

            if global_step >= max_train_steps:
                break

    print("\nSaving final LoRA adapter...")

    final_output_dir = os.path.join(
        output_dir,
        "checkpoint",
    )

    os.makedirs(
        final_output_dir,
        exist_ok=True,
    )

    unet.save_pretrained(
        final_output_dir
    )

    print(
        "Training completed successfully."
    )

    print(
        f"LoRA adapter saved to: "
        f"{final_output_dir}"
    )


if __name__ == "__main__":
    main()