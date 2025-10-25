import os
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed


# ============================================================
# Dataset
# ============================================================
class ArrowControlNetDataset(Dataset):
    def __init__(self, df, input_dir, latent_dir, input_transform=None):
        self.df = df
        self.input_dir = input_dir
        self.latent_dir = latent_dir
        self.input_transform = input_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Conditioning Image (ControlNet用) ---
        source_filename = f"{row['source_file']:06d}.jpg"
        input_img_path = os.path.join(self.input_dir, source_filename)
        conditioning_image = Image.open(input_img_path).convert("RGB")
        if self.input_transform:
            conditioning_image = self.input_transform(conditioning_image)

        # --- 事前計算済み latent を読み込み ---
        latent_filename = f"{row['id']:06d}.pt"
        latent_path = os.path.join(self.latent_dir, latent_filename)
        latents = torch.load(latent_path)

        # --- トークナイズ済みテキストIDを使用 ---
        text_input_ids = torch.tensor(row["text_input_ids"], dtype=torch.long)

        return {
            "latents": latents,
            "conditioning_image": conditioning_image,
            "text_input_ids": text_input_ids
        }


# ============================================================
# Training
# ============================================================
def main():
    # --- Paths ---
    INPUT_IMAGE_DIR = "./kenpro/arrow_dataset/input_images"
    LATENT_DIR = "./kenpro/arrow_dataset/latents"
    JSON_PATH = "./kenpro/arrow_dataset/annotations_with_tokens.json"
    OUTPUT_DIR = "./arrow_controlnet_model"

    # --- Hyperparams ---
    IMG_SIZE = 512
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    SEED = 42

    set_seed(SEED)

    # --- Accelerator (mixed precisionで高速化) ---
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")

    # --- モデル読み込み ---
    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # --- 固定（UNet/TextEncoder は更新しない）---
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # --- メモリ効率化設定 ---
    try:
        unet.enable_attention_slicing()
        controlnet.enable_attention_slicing()
    except Exception:
        pass
    try:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # --- Dataset / DataLoader ---
    df = pd.read_json(JSON_PATH, encoding="utf-8")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    num_workers = min(8, max(1, os.cpu_count() - 2))

    dataset = ArrowControlNetDataset(df, INPUT_IMAGE_DIR, LATENT_DIR, transform)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)

    # --- Accelerator準備 ---
    controlnet, unet, text_encoder, optimizer, train_loader = accelerator.prepare(
        controlnet, unet, text_encoder, optimizer, train_loader
    )

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(EPOCHS):
        controlnet.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                latents = batch["latents"].to(accelerator.device)
                conditioning_images = batch["conditioning_image"].to(accelerator.device)
                text_input_ids = batch["text_input_ids"].to(accelerator.device)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]

                # ControlNet forward
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_images,
                    return_dict=False,
                )

                # UNet forward
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                loss = F.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if (step + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{EPOCHS}] finished. Average Loss: {total_loss / len(train_loader):.4f}")

    # ============================================================
    # Save Model
    # ============================================================
    accelerator.wait_for_everyone()
    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
    unwrapped_controlnet.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
