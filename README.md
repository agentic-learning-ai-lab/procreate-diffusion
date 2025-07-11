# ProCreate Diffusion

### [Paper](https://arxiv.org/abs/2408.02226) | [Project Page](https://procreate-diffusion.github.io/) | [FSCG-8 Dataset](https://huggingface.co/datasets/Jacklu0831/few-shot-creative-generation-8)

Public code release for the paper "ProCreate, Don’t Reproduce! Propulsive Energy Diffusion for Creative Generation" (ECCV 2024).

![Teaser Figure](assets/_teaser.png)

## Setup

Download and set up the repo:
```bash
git clone https://github.com/agentic-learning-ai-lab/procreate-diffusion.git
cd procreate-diffusion
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Dataset FSCG-8

![Dataset Figure](assets/_dataset.png)
The FSCG-8 dataset is locally stored under `few-shot-creative-generation-8`. FSCG-8 can also be loaded from Huggingface as [Jacklu0831/few-shot-creative-generation-8](https://huggingface.co/datasets/Jacklu0831/few-shot-creative-generation-8).

## Training

The following command fine-tunes a Stable Diffusion checkpoint on Amedeo Modigliani paintings. To fine-tune on other FSCG-8 categories, see commands in `commands_train.sh`. You can also collect your own few-shot caption-image pairs and put them under `dataset_dir`.

```bash
python src/train.py \
    --output_dir temp \
    --dataset_dir few-shot-creative-generation-8/amedeo_modigliani \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32
```

## Sampling

The following command runs ProCreate sampling for the prompt "*a Amedeo Modigliani painting of a boy in a suit and hat*" from a Stable Diffusion checkpoint that is fine-tuned on Amedeo Modigliani paintings. `src/inference.py` automatically downloads the [model weights from Huggingface](https://huggingface.co/Jacklu0831/procreate-diffusion-amedeo-modigliani).

```bash
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/amedeo_modigliani \
    --unet_ckpt_dir Jacklu0831/procreate-diffusion-amedeo-modigliani \
    --prompt "a Amedeo Modigliani painting of a boy in a suit and hat" \
    --dreamsim_w 250 \
    --max_grad_norm 0.5 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32
```

Sample outputs:
<p float="left">
  <img src="assets/amedeo0.jpg" width="200" />
  <img src="assets/amedeo1.jpg" width="200" />
  <img src="assets/amedeo2.jpg" width="200" />
  <img src="assets/amedeo3.jpg" width="200" />
</p>

We also provide a trained checkpoint for each other category in FSCG-8. We use commands in `commands_inference.sh` to display ProCreate samples from each checkpoint.

[Apple checkpoint](https://huggingface.co/Jacklu0831/procreate-diffusion-apple) with prompt "*an Apple VR headset*":
<p float="left">
  <img src="assets/apple0.jpg" width="200" />
  <img src="assets/apple1.jpg" width="200" />
  <img src="assets/apple2.jpg" width="200" />
  <img src="assets/apple3.jpg" width="200" />
</p>

[Burberry checkpoint](https://huggingface.co/Jacklu0831/procreate-diffusion-burberry) with prompt "*a Burberry stuffed bear*":
<p float="left">
  <img src="assets/burberry0.jpg" width="200" />
  <img src="assets/burberry1.jpg" width="200" />
  <img src="assets/burberry2.jpg" width="200" />
  <img src="assets/burberry3.jpg" width="200" />
</p>

[Frank Gehry checkpoint](https://huggingface.co/Jacklu0831/procreate-diffusion-frank-gehry) with prompt "*a twisting tall apartment building, designed by Frank Gehry*":
<p float="left">
  <img src="assets/frank0.jpg" width="200" />
  <img src="assets/frank1.jpg" width="200" />
  <img src="assets/frank2.jpg" width="200" />
  <img src="assets/frank3.jpg" width="200" />
</p>

[Nouns checkpoint](https://huggingface.co/Jacklu0831/procreate-diffusion-nouns) with prompt "*a mountain Nouns character*":
<p float="left">
  <img src="assets/nouns0.jpg" width="200" />
  <img src="assets/nouns1.jpg" width="200" />
  <img src="assets/nouns2.jpg" width="200" />
  <img src="assets/nouns3.jpg" width="200" />
</p>

[One Piece checkpoint](https://huggingface.co/Jacklu0831/procreate-diffusion-one-piece) with prompt "*a One Piece man in a purple robe*":
<p float="left">
  <img src="assets/one_piece0.jpg" width="200" />
  <img src="assets/one_piece1.jpg" width="200" />
  <img src="assets/one_piece2.jpg" width="200" />
  <img src="assets/one_piece3.jpg" width="200" />
</p>

[Pokemon checkpoint](https://huggingface.co/Jacklu0831/procreate-diffusion-pokemon) with prompt "*a jellyfish Pokemon*":
<p float="left">
  <img src="assets/pokemon0.jpg" width="200" />
  <img src="assets/pokemon1.jpg" width="200" />
  <img src="assets/pokemon2.jpg" width="200" />
  <img src="assets/pokemon3.jpg" width="200" />
</p>

[Rococo checkpoint](https://huggingface.co/Jacklu0831/procreate-diffusion-rococo) with prompt "*a Rococo style bed*":
<p float="left">
  <img src="assets/rococo0.jpg" width="200" />
  <img src="assets/rococo1.jpg" width="200" />
  <img src="assets/rococo2.jpg" width="200" />
  <img src="assets/rococo3.jpg" width="200" />
</p>

## Acknowledgments

The code is adapted from [Huggingface Diffusers](https://github.com/huggingface/diffusers).

## Citation
If you have any questions or find any bugs, please feel free to contact Jack Lu (yl11330@nyu.edu).
If you found our work helpful, please cite it:
```
@InProceedings{procreate,
    author="Lu, Jack and Teehan, Ryan and Ren, Mengye",
    title="ProCreate, Don’t Reproduce! Propulsive Energy Diffusion for Creative Generation",
    booktitle="Computer Vision -- ECCV 2024",
    year="2024",
    publisher="Springer Nature Switzerland",
}
```
