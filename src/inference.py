import os
import glob
import random
from pathlib import Path
import argparse
import utils as utils
from guided_inference import GuidedInference

import torch
import torch.utils.checkpoint
from torchvision import transforms

import diffusers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datasets
import transformers

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command-line arguments for sampling."""
    parser = argparse.ArgumentParser(description="Single-GPU sampling script for ProCreate.")
    # model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="pt-sk/stable-diffusion-1.5",
                        help="path to base pretrained model")
    parser.add_argument("--unet_ckpt_dir", type=str, default="", help="directory of saved unet ckpt from train.py")
    # dataset
    parser.add_argument("--dataset_dir", type=str, required=True, help="local dataset directory")
    parser.add_argument("--num_train_sample", type=int, default=10,
                        help="data size to randomly sample for the training set")
    parser.add_argument("--prompt", type=str, required=True, help="prompt to generate images with")
    # generation parameters
    parser.add_argument("--msla_step_size", type=int, default=5, help="multi-step look ahead step size")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="number of inference steps for producing sample images")
    parser.add_argument("--prompt_w", type=float, default=7.0, help="cfg toward prompt")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max guidance gradient norm")
    parser.add_argument("--dreamsim_w", type=float, help="dreamsim guidance scale", default=0.0)
    parser.add_argument("--num_gen_per_prompt", type=int, default=4, help="number of samples to generate per prompt")
    # optimization
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="whether or not to use xformers")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="whether or not to allow TF32 on Ampere GPUs to speed up training")
    # others
    parser.add_argument("--output_dir", type=str, default="sample_output",
                        help="the output directory where the model predictions will be written")
    parser.add_argument("--seed", type=int, default=42, help="a seed for reproducible dataset splitting")

    args = parser.parse_args()
    return args


args = parse_args()

# logging
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()
diffusers.utils.logging.set_verbosity_warning()
os.makedirs(args.output_dir, exist_ok=True)

# seed
utils.set_seed(args.seed)

# tf32 for ampere gpus
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

# load stable diffusion pipeline and unet ckpt (float16 to fit in memory)
pipe = StableDiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    use_safetensors=True,
    torch_dtype=torch.float16
)
if args.unet_ckpt_dir != "":
    pipe.unet = UNet2DConditionModel.from_pretrained(args.unet_ckpt_dir, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)
pipe.safety_checker = None

# memory efficiency
if args.enable_xformers_memory_efficient_attention:
    pipe.enable_xformers_memory_efficient_attention()

# load dataset
image_paths = glob.glob(f"{args.dataset_dir}/*.jpg")
image_paths = sorted(image_paths, key=lambda p: int(Path(p).stem)) # sort by image number
json_path = os.path.join(args.dataset_dir, "metadata.jsonl")
dataset = datasets.load_dataset("imagefolder", data_files=image_paths + [json_path]) # auto-loaded to train split

# randomly sample train dataset and get reference images to guide away from
image_transforms = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # to [-1, 1]
    ]
)
train_indices = sorted(random.sample(range(len(dataset["train"])), k=args.num_train_sample))
train_dataset = dataset["train"].select(train_indices)
train_dataset = utils.preprocess_dataset(train_dataset, image_transforms, pipe.tokenizer)
ref_images = [x["image"].convert("RGB") for x in train_dataset]

# generated and save images
guided_inference = GuidedInference(
    pipe=pipe,
    ref_images=ref_images,
    msla_step_size=args.msla_step_size,
    num_inference_steps=args.num_inference_steps,
    dreamsim_w=args.dreamsim_w,
    max_grad_norm=args.max_grad_norm,
    prompt_w=args.prompt_w,
)
# NOTE: only experimented with batch size 1 generation
for repeat_i in range(args.num_gen_per_prompt):
    img = guided_inference([args.prompt], verbose=True)[0]
    out_path = os.path.join(args.output_dir, f"{repeat_i}.jpg")
    img.save(out_path)
    print("saved to", out_path)
