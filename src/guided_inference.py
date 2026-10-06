import math
from tqdm.auto import tqdm
from PIL import ImageOps
import numpy as np
import utils

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GuidedInference:
    def __init__(self, pipe, ref_images, msla_step_size, num_inference_steps, dreamsim_w, max_grad_norm, prompt_w):
        """Store the sampling settings, load DreamSim, and embed the reference images."""
        self.pipe = pipe
        self.ref_images = ref_images
        self.msla_step_size = msla_step_size
        self.num_inference_steps = num_inference_steps
        self.dreamsim_w = dreamsim_w
        self.max_grad_norm = max_grad_norm
        self.prompt_w = prompt_w

        # load dreamsim and compute reference embeddings
        self.dreamsim_model, self.dreamsim_preprocess, self.dreamsim_latent_transform = utils.load_dreamsim(device)
        self.collect_ref_embs()


    @torch.no_grad()
    def collect_ref_embs(self):
        """Collect DreamSim embeddings of the reference images to guide away from."""
        dreamsim_embs = []
        for image in self.ref_images:
            for img in [image, ImageOps.mirror(image)]:
                emb = self.dreamsim_model(self.dreamsim_preprocess(img).to(device))
                dreamsim_embs.append(emb[0])
        self.dreamsim_embs = torch.stack(dreamsim_embs, dim=0) if dreamsim_embs else None


    def ddim_step(self, model_output, timestep, sample, prev_timestep=None):
        """Run a single DDIM denoising step."""
        if prev_timestep is None:
            step_size = self.pipe.scheduler.config.num_train_timesteps // self.pipe.scheduler.num_inference_steps
            prev_timestep = timestep - step_size

        # setup coefficients
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.pipe.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep < 0:
            alpha_prod_t_prev = self.pipe.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # cf: DDIM paper https://arxiv.org/pdf/2010.02502
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample, pred_original_sample


    def estimate_x0_with_msla(self, latents, timestep, text_embeddings):
        """Predict x0 with multiple DDIM sampling steps (multi-step look-ahead)."""
        # get msla timestep schedule
        timestep = timestep.cpu()
        step_size = math.ceil(timestep / self.msla_step_size)
        timestep_schedule = np.round(np.arange(timestep, 0, -step_size))
        timestep_schedule = list(timestep_schedule)
        timestep_schedule.append(0)
        # perform msla estimation of x0
        for curr_t, prev_t in zip(timestep_schedule[:-1], timestep_schedule[1:]):
            noise_pred = self.unet_forward(latents, curr_t, text_embeddings)
            latents, _ = self.ddim_step(noise_pred, curr_t, latents, prev_timestep=prev_t)
        return latents


    def __call__(self, prompts, verbose=True):
        """Sample an image with DreamSim-guided propulsive guidance."""
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device)

        # encode prompt (unconditional followed by conditional) with shape (batchsize x 2, 77, 768)
        text_embeddings = self.pipe._encode_prompt(prompt=prompts,
                                                   device=device,
                                                   num_images_per_prompt=1,
                                                   do_classifier_free_guidance=True)
        text_embeddings = text_embeddings.to(self.pipe.unet.dtype).detach().clone()

        # initialize noise latent
        batch_size = len(prompts)
        latents = torch.empty((batch_size, 4, 64, 64), device=device, dtype=self.pipe.unet.dtype).normal_()
        latents *= self.pipe.scheduler.init_noise_sigma

        # iterator based on verbosity
        iterator = self.pipe.scheduler.timesteps
        if verbose:
            iterator = tqdm(iterator)

        for timestep in iterator:
            if self.dreamsim_w != 0.0:
                latents = latents.detach().requires_grad_()

            # sample noise pred with cfg
            noise_pred = self.unet_forward(latents, timestep, text_embeddings)
            if self.dreamsim_w == 0.0:
                prev_latents = self.pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
            else:
                # compute an estimate for x0 in latent space (optionally with MSLA)
                if self.msla_step_size == -1:
                    pred_x0 = self.pipe.scheduler.step(noise_pred, timestep, latents).pred_original_sample
                else:
                    pred_x0 = self.estimate_x0_with_msla(
                        latents=latents,
                        timestep=timestep,
                        text_embeddings=text_embeddings
                    )

                # compute dreamsim loss and backprop
                dreamsim_loss = self.guidance_loss_function(pred_x0=pred_x0)
                total_grad = torch.autograd.grad(dreamsim_loss, latents)[0]
                total_grad = torch.clamp(total_grad, min=-self.dreamsim_w, max=self.dreamsim_w)

                # clip grad norm
                total_grad_norm = torch.norm(total_grad)
                max_weighted_guide_grad_norm = self.dreamsim_w * self.max_grad_norm
                if total_grad_norm > max_weighted_guide_grad_norm:
                    total_grad = total_grad / total_grad_norm * max_weighted_guide_grad_norm

                # apply propulsive guidance to the noise prediction
                noise_pred = noise_pred.detach().clone()
                alphas_cumprod_t = self.pipe.scheduler.alphas_cumprod[timestep]
                noise_pred = noise_pred + (1.0 - alphas_cumprod_t) ** 0.5 * total_grad

                # estimate a previous latent
                latents = latents.detach().clone()
                prev_latents = self.pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

            # update latent
            latents = prev_latents.detach()

        # decode latent and return PIL image
        with torch.no_grad():
            images = self.pipe.decode_latents(latents.detach().to(dtype=self.pipe.vae.dtype))
        return self.pipe.numpy_to_pil(images)


    def unet_forward(self, latents, timestep, text_embeddings):
        """Compute the UNet noise prediction."""
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, timestep=timestep)
        noise_pred = self.pipe.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.prompt_w * (noise_pred_text - noise_pred_uncond)
        return noise_pred


    def guidance_loss_function(self, pred_x0):
        """Compute the guidance loss between the estimated pred_x0 and the reference images."""
        # get predicted x0's dreamsim embeddings
        pred_x0_img = self.decode_latents(pred_x0)
        dreamsim_pred_embs = self.dreamsim_model(self.dreamsim_latent_transform(pred_x0_img.float()))

        # compute matching reference image(s) to pred_x0
        with torch.no_grad():
            cossims = utils.pairwise_cossim(dreamsim_pred_embs.float(), self.dreamsim_embs.float())
            closest_idxs = torch.topk(cossims, k=1).indices

        # compute cosine similarity between dreamsim embeddings
        dreamsim_loss = torch.mean(
            F.cosine_similarity(
                dreamsim_pred_embs.float(),
                self.dreamsim_embs[closest_idxs[0]].float(), dim=-1,
            )
        )
        return dreamsim_loss * self.dreamsim_w


    def decode_latents(self, pred_x0):
        """Decode latents into images with values in [0, 1]."""
        # decode latent into images
        pred_x0_img = self.pipe.vae.decode(
            1.0 / self.pipe.vae.config.scaling_factor * pred_x0.to(self.pipe.vae.dtype),
            return_dict=False,
        )[0]
        return (pred_x0_img / 2.0 + 0.5).clamp(0, 1)
