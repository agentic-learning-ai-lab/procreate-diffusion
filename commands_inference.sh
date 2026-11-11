
# produce samples from trained amedeo_modigliani checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/amedeo_modigliani \
    --unet_ckpt_dir Jacklu0831/procreate-diffusion \
    --unet_subfolder amedeo_modigliani \
    --prompt "a Amedeo Modigliani painting of a boy in a suit and hat" \
    --dreamsim_w 250 \
    --max_grad_norm 0.5 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# produce samples from trained apple checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/apple \
    --unet_ckpt_dir Jacklu0831/procreate-diffusion \
    --unet_subfolder apple \
    --prompt "an Apple VR headset" \
    --dreamsim_w 250 \
    --max_grad_norm 0.3 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# produce samples from trained burberry checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/burberry \
    --unet_ckpt_dir train_output/burberry/checkpoint-2000 \
    --prompt "a Burberry stuffed bear" \
    --dreamsim_w 100 \
    --max_grad_norm 0.5 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# produce samples from trained frank_gehry checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/frank_gehry \
    --unet_ckpt_dir train_output/frank_gehry/checkpoint-2000 \
    --prompt "a twisting tall apartment building, designed by Frank Gehry" \
    --dreamsim_w 250 \
    --max_grad_norm 0.3 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# produce samples from trained nouns checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/nouns \
    --unet_ckpt_dir train_output/nouns/checkpoint-2000 \
    --prompt "a mountain Nouns character" \
    --dreamsim_w 100 \
    --max_grad_norm 0.3 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# produce samples from trained one_piece checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/one_piece \
    --unet_ckpt_dir train_output/one_piece/checkpoint-2000 \
    --prompt "a One Piece man in a purple robe" \
    --dreamsim_w 100 \
    --max_grad_norm 0.5 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# produce samples from trained pokemon checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/pokemon \
    --unet_ckpt_dir train_output/pokemon/checkpoint-2000 \
    --prompt "a jellyfish Pokemon" \
    --dreamsim_w 100 \
    --max_grad_norm 0.3 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# produce samples from trained rococo checkpoint with a sample prompt
python src/inference.py \
    --dataset_dir few-shot-creative-generation-8/rococo \
    --unet_ckpt_dir train_output/rococo/checkpoint-2000 \
    --prompt "a Rococo style bed" \
    --dreamsim_w 250 \
    --max_grad_norm 0.3 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32
