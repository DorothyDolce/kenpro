import os
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 修正箇所: VAE_NETWORK_NAME -> AutoencoderKL
from diffusers import DDPMScheduler, UNet2DConditionModel, ControlNetModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

class ArrowControlNetDataset(Dataset):
    def __init__(self, df, input_dir, target_dir, tokenizer, input_transform=None, target_transform=None):
        self.df = df
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.tokenizer = tokenizer
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df) # 1エポックに何回のループが必要かを計算

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_prompt = row['coaching_text']
        text_inputs = self.tokenizer(
            text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.squeeze()
        source_filename = f"{row['source_file']:06d}.jpg"
        target_filename = f"{row['id']:06d}.jpg"
        input_img_path = os.path.join(self.input_dir, source_filename)
        target_img_path = os.path.join(self.target_dir, target_filename)
        conditioning_image = Image.open(input_img_path).convert("RGB")
        image = Image.open(target_img_path).convert("RGB")
        if self.input_transform:
            conditioning_image = self.input_transform(conditioning_image)
        if self.target_transform:
            image = self.target_transform(image)
        return {
            "image": image,
            "conditioning_image": conditioning_image,
            "text_input_ids": text_input_ids
        }

def main():
    INPUT_IMAGE_DIR = './arrow_dataset/input_images'
    TARGET_IMAGE_DIR = './arrow_dataset/target_images'
    JSON_PATH = './arrow_dataset/annotations.json'
    OUTPUT_DIR = "./arrow_controlnet_model"
    IMG_SIZE = 512
    BATCH_SIZE = 2
    EPOCHS = 1
    LEARNING_RATE = 1e-6
    SEED = 42
    set_seed(SEED)
    accelerator = Accelerator(gradient_accumulation_steps=1)
    model_id = "runwayml/stable-diffusion-v1-5"

    # 転移学習のため、事前学習済みモデルをロード
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer") # 辞書
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet") # ノイズ除去を行うネットワーク
    controlnet = ControlNetModel.from_unet(unet)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    df = pd.read_json(JSON_PATH)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = ArrowControlNetDataset(df, INPUT_IMAGE_DIR, TARGET_IMAGE_DIR, tokenizer, transform, transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)
    
    controlnet, unet, text_encoder, vae, optimizer, train_loader = accelerator.prepare(
        controlnet, unet, text_encoder, vae, optimizer, train_loader
    )

    for epoch in range(EPOCHS):
        controlnet.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                clean_images = batch["image"]
                conditioning_images = batch["conditioning_image"]
                text_input_ids = batch["text_input_ids"]
                latents = vae.encode(clean_images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_images,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents, # 現在のノイズ化された画像
                    timesteps, # ノイズ量
                    encoder_hidden_states=encoder_hidden_states, # tokenizerが変換したテキスト情報
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

    accelerator.wait_for_everyone()
    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
    unwrapped_controlnet.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()