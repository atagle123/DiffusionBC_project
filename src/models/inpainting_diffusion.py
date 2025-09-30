import numpy as np
import torch.nn.functional as F
import torch
from .diffusion import Sample, GaussianDiffusion
from tqdm import tqdm

from .helpers import (
    extract,
)

class GaussianInpaintingDiffusion(GaussianDiffusion):
    """
    Gaussian diffusion model for inpainting
    """

    def __init__(self, model, data_dim: int, schedule: str="cosine", n_timesteps: int=15, cond_drop_prob: float=0.0, pad_value: float=0, noise_cond: bool=True, compile: bool=False):
        assert not (cond_drop_prob > 0.0 and noise_cond), "Cannot use conditional guidance with noised conditioning"

        super().__init__(model, data_dim, schedule, n_timesteps, cond_drop_prob, pad_value, compile)

        self.noise_cond = noise_cond

    def setup_sampling(
        self,
        clip_denoised=True,
        temperature=1,
        disable_progess_bar=True,
        return_chain=False,
        horizon = 16, # TODO this should be set in the model...
        guidance_scale=0, # if 0 no class free guidance... 
        **kwargs,
    ):

        self.temperature = temperature
        self.clip_denoised = clip_denoised
        self.disable_progess_bar = disable_progess_bar
        self.return_chain = return_chain
        self.horizon = horizon
        self.guidance_scale = guidance_scale

    @torch.inference_mode()
    def p_mean_variance(self, x, condition, t):
        '''
        x: [batch, horizon, transition]
        condition: [batch, history, transition]
        '''

        _, cond_len, _ = condition.shape
        x_cond = x.clone()
        x_cond[:,:cond_len,:] = condition

        epsilon = self.model(x=x_cond, time=t, training=False)
        if self.guidance_scale > 0.0:
            # Classifier-free guidance
            epsilon_uncond = self.model(x=x, time=t, training=False) 
            epsilon = epsilon + self.guidance_scale * (epsilon - epsilon_uncond)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance
    
    def noise_condition(self, condition):
        device = condition.device
        noised_condition = torch.zeros(self.n_timesteps, *(condition.shape), device=device)
        noised_condition[0] = condition

        for t in range(1,self.n_timesteps):
            noised_condition[t] = self.noise_step(noised_condition[t-1], t)
        
        return noised_condition
    
    def noise_step(self, x, t):
        b, *_, device = *x.shape, x.device
        t = torch.full((b,), t, device=device, dtype=torch.long)
        return extract(self.sqrt_alphas, t, x.shape) * x + extract(self.betas, t, x.shape) * torch.randn_like(x)

    def p_sample_loop(self, condition, shape):
        """
        Classical DDPM sampling algorithm

            Parameters:
                condition
                shape,

            Returns:
            sample

        """
        device = self.betas.device

        x = torch.randn(shape, device=device) * self.temperature

        chain = [x] if self.return_chain else None

        if self.noise_cond:
            condition = self.noise_condition(condition)

        for t in tqdm(
            reversed(range(0, self.n_timesteps)),
            desc="sampling loop time step",
            total=self.n_timesteps,
            disable=self.disable_progess_bar,
        ):
            cond = condition[t] if self.noise_cond else condition
            x = self.p_sample(x=x, condition=cond, t=t)
            
            if self.return_chain:
                chain.append(x)

        if self.return_chain:
            chain = torch.stack(chain, dim=1)

        if self.clip_denoised:
            x.clamp_(-1.0, 1.0)

        return Sample(x, chain)
    
        
    @torch.no_grad()
    def forward(self, condition):
        """ """
        batch_size = condition.shape[0]

        return self.p_sample_loop(condition, shape=(batch_size, self.horizon, self.data_dim))
    
    # ------------------------------------------ training ------------------------------------------#

    def p_losses(self, x_start, mask, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_noisy[mask] = x_start[mask] # apply mask
        pred_epsilon = self.model(x_noisy, t, training=True)

        assert noise.shape == pred_epsilon.shape

        loss = F.mse_loss(pred_epsilon[~mask], noise[~mask])

        return loss, {"loss": loss}

    def loss(self, x, mask):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, mask, t)