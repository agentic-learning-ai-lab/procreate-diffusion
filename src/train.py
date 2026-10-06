import os
import glob
import random
from pathlib import Path
import argparse
from tqdm.auto import tqdm
import utils

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms

import diffusers
from diffusers.optimization import get_scheduler
from diffusers import StableDiffusionPipeline, DDIMScheduler
import datasets
import transformers

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Single-GPU training script for ProCreate")
    # model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="pt-sk/stable-diffusion-1.5",
                        help="path to base pretrained model")
    # dataset
    parser.add_argument("--dataset_dir", type=str, required=True, help="local dataset directory")
    parser.add_argument("--num_train_sample", type=int, default=10,
                        help="data size to randomly sample for the training set")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size of the training dataloader")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="number of subprocesses to use for data loading")
    # training
    parser.add_argument("--train_steps", type=int, default=2500, help="total number of training steps to perform")
    parser.add_argument("--checkpointing_steps", type=int, default=250,
                        help="save a checkpoint of the training state every X updates")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max backpropagation gradient norm")
    # optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="initial learning rate (after the potential warmup period) to use")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="scheduler: linear, cosine, polynomial, constant, constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="number of steps for the warmup in the lr scheduler")
    # optimization
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="whether or not to use xformers")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="whether or not to allow TF32 on Ampere GPUs to speed up training")
    # others
    parser.add_argument("--output_dir", type=str, required=True,
                        help="the output directory where the model checkpoints will be written")
    parser.add_argument("--seed", type=int, default=42, help="a seed for reproducible training")

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

# load stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, use_safetensors=True)
pipeline.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
pipeline = pipeline.to(device)
pipeline.set_progress_bar_config(disable=True)
pipeline.safety_checker = None

# break pipeline into components
noise_scheduler = pipeline.scheduler
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
vae = pipeline.vae
unet = pipeline.unet
unet.train()

# freeze parameters
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# memory efficiency
if args.enable_xformers_memory_efficient_attention:
    unet.enable_xformers_memory_efficient_attention()

# load dataset
image_paths = glob.glob(f"{args.dataset_dir}/*.jpg")
image_paths = sorted(image_paths, key=lambda p: int(Path(p).stem)) # sort by image number
json_path = os.path.join(args.dataset_dir, "metadata.jsonl")
dataset = datasets.load_dataset("imagefolder", data_files=image_paths + [json_path]) # auto-loaded to train split

# randomly sample train dataset and make dataloader
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
train_dataset = utils.preprocess_dataset(train_dataset, image_transforms, tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=utils.collate_fn,
    batch_size=args.batch_size,
    num_workers=args.dataloader_num_workers,
)

# optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

# lr scheduler
lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=args.train_steps,
)

# train info
print("***** Running training *****")
print(f"  Num train examples = {len(train_dataset)}")
print(f"  Num train iters = {args.train_steps}")
print(f"  Batch size = {args.batch_size}")

# progress bar
global_step = 0
progress_bar = tqdm(range(global_step, args.train_steps))
progress_bar.set_description("Steps")

for _ in tqdm(range(args.train_steps)):
    # sample a random batch from train_dataloader
    batch = next(iter(train_dataloader))
    pixel_values = batch["pixel_values"].to(device=device)
    input_ids = batch["input_ids"].to(device=device)

    # prepare text embeddings and image latents
    encoder_hidden_states = text_encoder(input_ids)[0]
    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

    # sample timestep and noise then do forward diffusion
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.batch_size,), device=device)
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # predict noise and compute loss
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    # backprop, clip, update
    loss.backward()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # save model sometimes
    progress_bar.update(1)
    global_step += 1
    if global_step % args.checkpointing_steps == 0:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        unet.save_pretrained(os.path.join(save_path))
        print(f"Saved state to {save_path}")

# save final model
save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
if not os.path.exists(save_path):
    unet.save_pretrained(os.path.join(save_path))
    print(f"Saved state to {save_path}")
