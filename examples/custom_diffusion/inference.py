from diffusers import StableDiffusionPipeline
import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--token", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--num_images", type=int)
args =parser.parse_args()

assert args.token in args.prompt

pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs(args.model_path, weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion(args.model_path, weight_name=f"{args.token}.bin")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for i in range(args.num_images):
    image = pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    save_path = os.path.join(args.save_dir, f"{args.prompt}_{i}.png")
    image.save(save_path)
