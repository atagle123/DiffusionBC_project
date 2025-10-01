import math
import numpy as np
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from einops import rearrange


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def divisible_by(numer, denom):
    return (numer % denom) == 0


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, learnable=True):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=learnable)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (
            loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
        ).mean()
        return weighted_loss, {"a0_loss": a0_loss}


### Unet models


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------#
# --------------------------------- attention ---------------------------------#
# -----------------------------------------------------------------------------#


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.g = nn.Parameter(torch.ones(1, dim, 1))
            self.b = nn.Parameter(torch.zeros(1, dim, 1))
            self.out = lambda z: z * self.g + self.b
        else:
            self.out = lambda z: z

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        z = (x - mean) / (var + self.eps).sqrt()

        return self.out(z)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, Norm=LayerNorm, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.norm = Norm(dim, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv
        )
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c d -> b (h c) d")
        return self.to_out(out)


class ScaledResidual(Residual):
    def forward(self, x, scale, *args, **kwargs):
        return scale * self.fn(x, *args, **kwargs) + x


class FiLMedLayerNorm(LayerNorm):
    def __init__(self, dim, eps=1e-5):
        super().__init__(dim, eps, affine=False)

    def forward(self, x, g, b):
        z = super().forward(x)
        return z * (1 + g) + b


class FiLMedPreNorm(PreNorm):
    def __init__(self, dim, fn):
        super().__init__(dim, fn, Norm=FiLMedLayerNorm)

    def forward(self, x, g, b):
        x = self.norm(x, g, b)
        return self.fn(x)


class FiLMedLinearAttention(LinearAttention):
    """
    Applies feature-wise linear modulation (FiLM) at the level of transformer heads.
    """

    def forward(self, x, g, b):
        """
        x : [ batch x transition x horizon ]
        g : [ batch x heads]
        b : [ batch x heads]
        """

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv
        )
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        g, b = g[:, :, None, None], b[:, :, None, None]
        out = (1 + g) * out + b  # apply FiLM on the transformer heads
        out = einops.rearrange(out, "b h c d -> b (h c) d")
        return self.to_out(out)
