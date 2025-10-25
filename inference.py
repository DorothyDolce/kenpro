import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image

# --- 1. パイプラインと学習済みControlNetの準備 ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
base_model_id = "runwayml/stable-diffusion-v1-5"
controlnet_path = "./arrow_controlnet_model"

# 学習済みControlNetを読み込む
controlnet = ControlNetModel.from_pretrained(controlnet_path)

# torch_dtypeの指定を削除して、より安定したfloat32でモデルを読み込む
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
).to(device)

# 推論を高速化するスケジューラに変更
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# もしメモリが足りなくなった場合は、以下のコメントを外して有効化してください
# pipe.enable_model_cpu_offload()

print("モデルの読み込みが完了しました。")

# --- 2. 推論の準備：入力データを用意する ---
prompt = "move head to the right"
negative_prompt = "low quality, blurry, text, watermark, bad anatomy"
input_image_path = "./test_image/000000.jpg" 
input_image = Image.open(input_image_path).convert("RGB")
generator = torch.Generator(device=device).manual_seed(42)

# --- 3. 推論の実行と保存 ---
print("推論を開始します...")
image = pipe(
    prompt,
    image=input_image,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=25,
).images[0]

output_path = "arrow_output.png"
image.save(output_path)
print(f"生成が完了し、画像を {output_path} に保存しました。")