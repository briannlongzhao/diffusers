from diffusers import StableDiffusionPipeline
import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--num_images", type=int)
args =parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for i in range(args.num_images):
    image = pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    save_path = os.path.join(args.save_dir, f"{args.prompt}_{i}.png")
    image.save(save_path)
