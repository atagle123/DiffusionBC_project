from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .helpers import (
    cosine_beta_schedule,
    extract,
)


Sample = namedtuple("Sample", "sample chains")


class GaussianDiffusion(nn.Module):
    """
    Base Gaussian diffusion model
    """

    def __init__(self, model, data_dim: int, schedule: str="cosine", n_timesteps: int=15, cond_drop_prob: float= 0.1, pad_value: float = 0):
        super().__init__()

        self.model = model
        self.data_dim = data_dim
        self.cond_drop_prob = cond_drop_prob
        self.pad_value = pad_value # used for classifier free guidance

        schedulers = {
            "cosine": cosine_beta_schedule,
        }

        if schedule in schedulers:
            betas = schedulers[schedule](n_timesteps)
        else:
            raise ValueError(
                f"Unknown schedule: {schedule}. Available options are 'vp', 'cosine', or 'linear'."
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )  # helper function to register buffer from float64 to float32

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------ sampling ------------------------------------------#
    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, : self.action_dim] = action_weight
        return loss_weights
    
    def predict_start_from_noise(self, x_t, t, noise):
        """ """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.inference_mode()
    def p_mean_variance(self, x, condition, t):

        epsilon = self.model(x=x, condition=condition, time=t, training=False)
        if self.guidance_scale > 0.0:
            # Classifier-free guidance
            epsilon_uncond = self.model(x=x, condition=torch.full_like(condition, fill_value=self.pad_value), time=t, training=False) # precompute the torch full like matrix...
            epsilon = epsilon + self.guidance_scale * (epsilon - epsilon_uncond)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, condition, t):
        b, *_, device = *x.shape, x.device

        batched_time = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, condition=condition, t=batched_time
        )

        if t > 0:
            noise = torch.randn_like(x) * self.temperature
        else:
            noise = 0

        x_pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return x_pred

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

        for t in tqdm(
            reversed(range(0, self.n_timesteps)),
            desc="sampling loop time step",
            total=self.n_timesteps,
            disable=self.disable_progess_bar,
        ):
            x = self.p_sample(x=x, condition=condition, t=t)
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

        return self.p_sample_loop(condition, shape=(batch_size, self.data_dim))

    def setup_sampling(
        self,
        clip_denoised=True,
        temperature=1,
        disable_progess_bar=True,
        return_chain=False,
        guidance_scale=0, # if 0 no class free guidance... 
        **kwargs,
    ):

        self.temperature = temperature
        self.clip_denoised = clip_denoised
        self.disable_progess_bar = disable_progess_bar
        self.return_chain = return_chain
        self.guidance_scale = guidance_scale

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise):

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, condition, t):

        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        pred_epsilon = self.model(x_noisy, condition, t, training=True)

        assert noise.shape == pred_epsilon.shape

        loss = F.mse_loss(pred_epsilon, noise, reduction="none").mean()

        return loss, {"loss": loss}

    def loss(self, x, condition):

        batch_size = len(x)

        cond_mask = torch.rand(batch_size) > self.cond_drop_prob
        cond_mask = cond_mask.to(condition.device)
        condition = torch.where(cond_mask[:, None], condition, torch.full_like(condition, fill_value=self.pad_value)) # drop condiitons for classifier free guidance

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.p_losses(x, condition, t)



class FiLMGaussianDiffusion(GaussianDiffusion):

    #------------------------------------------ sampling ------------------------------------------#

    def p_mean_variance(self, x, condition, t):
        assert condition is not None  # TODO there must be a condition, in the worst case scenario it consists of only one state. 
        self.model.condition_diffusion(condition)

        epsilon = self.model(x,t)

        if self.guidance_scale > 0.0: # TODO test 
            # Classifier-free guidance
            self.model.clear_conditioning()
            self.model.condition_diffusion(torch.full_like(condition, fill_value=self.pad_value))# maybe save two models one unconditional and one conditional
            epsilon_uncond = self.model(x,t)
            epsilon = epsilon + self.guidance_scale * (epsilon - epsilon_uncond)
            
        self.model.clear_conditioning()
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError() # NOTE

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, condition, shape, return_chain=False):
        device = self.betas.device

        x = torch.randn(shape, device=device)

        #self.model.condition_diffusion(condition) TODO 

        chain = [x] if return_chain else None

        for t in tqdm(
            reversed(range(0, self.n_timesteps)),
            desc="sampling loop time step",
            total=self.n_timesteps,
            disable=self.disable_progess_bar,
        ):
            x = self.p_sample(x=x, condition = condition, t=t) # TODO fix proeblem with conditioning... 

            if return_chain: 
                chain.append(x)

        if return_chain: 
            chain = torch.stack(chain, dim=1)

        if self.clip_denoised:
            x.clamp_(-1.0, 1.0)
        return Sample(x, chain)

    def setup_sampling(
        self,
        clip_denoised=True,
        temperature=1,
        disable_progess_bar=True,
        return_chain=False,
        horizon = 16,
        guidance_scale=0, # if 0 no class free guidance... 
        **kwargs,
    ):

        self.temperature = temperature
        self.clip_denoised = clip_denoised
        self.disable_progess_bar = disable_progess_bar
        self.return_chain = return_chain
        self.horizon = horizon
        self.guidance_scale = guidance_scale
    
    @torch.no_grad()
    def forward(self, condition):
        """ """
        batch_size = condition.shape[0]

        return self.p_sample_loop(condition, shape=(batch_size, self.horizon, self.data_dim))

    #------------------------------------------ training ------------------------------------------#

    def p_losses(self, x_start, condition, t):
        noise = torch.randn_like(x_start)

        self.model.condition_diffusion(condition)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t)

        assert noise.shape == x_recon.shape

        loss = F.mse_loss(x_recon, noise, reduction="none").mean() # TODO add weighted loss

        return loss, {"loss": loss}

    def loss(self, x, condition):
        assert not self.model.conditioning_set(), "Model conditioned with a pre-existing history. Cannot pre-condition model for loss computations."
        batch_size = len(x)

        cond_mask = torch.rand(batch_size) > self.cond_drop_prob 
        cond_mask = cond_mask.to(x.device) # This could be improved using always the same mask

        condition = torch.where(cond_mask[:, None,  None], condition, torch.full_like(condition, fill_value=self.pad_value)) # drop condiitons for classifier free guidance

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        losses = self.p_losses(x, condition, t)
        self.model.clear_conditioning()
        return losses