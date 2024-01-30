from diffusers import StableDiffusionPipeline
import torch
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from edit_prompts import edit_prompts

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--token", type=str, required=True)
parser.add_argument("--class_name", type=str, required=True)
parser.add_argument("--num_images", type=int, default=4)
args = parser.parse_args()
args.class_name = args.class_name.replace('_', ' ')

pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")
os.makedirs(args.save_dir, exist_ok=True)

for prompt in edit_prompts:
    prompt_dir = os.path.join(args.save_dir, prompt.format(args.class_name))
    os.makedirs(prompt_dir, exist_ok=True)
    i = len(os.listdir(prompt_dir))
    while i < args.num_images:
        image = pipe(prompt.format(f"{args.token} {args.class_name}"), num_inference_steps=50, guidance_scale=7.5).images[0]
        save_path = os.path.join(prompt_dir, f"{prompt.format(args.class_name)}_{i}.png")
        image.save(save_path)
        print(save_path)
        i += 1

