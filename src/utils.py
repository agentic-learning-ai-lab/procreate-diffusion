import random
import warnings
import numpy as np
from dreamsim import dreamsim

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms


def pairwise_cossim(x, y):
    """Compute pairwise cosine similarity between the vectors in x and y."""
    x_normalized = F.normalize(x, p=2, dim=1)
    y_normalized = F.normalize(y, p=2, dim=1)
    return torch.mm(x_normalized, y_normalized.t())


def set_seed(seed):
    """Seed the Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_dataset(dataset, image_transforms, tokenizer):
    """Apply image transforms and caption tokenization to a dataset."""
    def tokenize_captions(examples, tokenizer):
        """Collect and tokenize captions."""
        captions = [c for c in examples['text']]
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_examples(examples):
        """Transform images and tokenize captions."""
        images = [image.convert("RGB") for image in examples['image']]
        examples["pixel_values"] = [image_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples, tokenizer)
        return examples

    assert dataset.column_names == ['image', 'text']
    return dataset.with_transform(preprocess_examples)


def collate_fn(examples):
    """Collate a batch of images and tokenized captions."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    texts = [example['text'] for example in examples]
    return {"pixel_values": pixel_values, "input_ids": input_ids, 'texts': texts}


def load_dreamsim(device):
    """Load the DreamSim model, its preprocessing function, and a transform for decoded VAE latents."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dreamsim_base_model, dreamsim_preprocess = dreamsim(pretrained=True, cache_dir="dreamsim_ckpt")
        dreamsim_base_model = dreamsim_base_model.to(device)
        dreamsim_model = lambda x: dreamsim_base_model.embed(x)
        # [0, 1] latent to dreamsim input
        dreamsim_latent_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.functional.InterpolationMode.BICUBIC, antialias=True),
        ])
    return dreamsim_model, dreamsim_preprocess, dreamsim_latent_transform
