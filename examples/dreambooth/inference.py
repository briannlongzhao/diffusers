from diffusers import StableDiffusionPipeline
import torch
import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--num_images", type=int)
parser.add_argument("--idx", type=int)
args = parser.parse_args()

if args.idx is not None:
    sys.path.insert(0,str(Path(__file__).parent.parent.parent))
    from imagenet_classes import subset100, wnid2classname_simple
    args.wnid = subset100[args.idx]
    args.class_name = wnid2classname_simple[args.wnid]
    args.prompt = f"{args.prompt.strip()} {args.class_name}" 
    args.model_path = os.path.join(args.model_path, str(args.idx))
    args.save_dir = os.path.join(args.save_dir, args.class_name)

pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

i = len(os.listdir(args.save_dir))
while i < args.num_images:
    image = pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    save_path = os.path.join(args.save_dir, f"{args.prompt}_{i}.png")
    image.save(save_path)
    print(save_path)
    i += 1
