import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

from typing import Sequence, Callable, Optional

from src.models.helpers import (
    RandomOrLearnedSinusoidalPosEmb,
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    ScaledResidual,
    FiLMedPreNorm,
    FiLMedLinearAttention,
)


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        scale : [batch_size x out_channels x 1]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ScaledResidualTemporalBlock(ResidualTemporalBlock):
    def forward(self, x, t, scale):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        scale : [batch_size x out_channels x 1]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return scale * out + self.residual_conv(x)


class ResidualEncodingBlock(nn.Module):
    def __init__(self, inp_channels, channels, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([])
        in_ch = inp_channels
        for out_ch in channels:
            self.blocks.append(Conv1dBlock(in_ch, out_ch, kernel_size))
            in_ch = out_ch

        self.residual_conv = (
            nn.Conv1d(inp_channels, channels[-1], 1)
            if inp_channels != channels[-1]
            else nn.Identity()
        )

    def forward(self, x):
        """
        x -> [ batch x transition x horizon]
        """
        out = x
        for block in self.blocks:
            out = block(out)
        return out + self.residual_conv(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims: Sequence[int],
        activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
        activate_final: bool = False,
        use_layer_norm: bool = False,
        scale_final: Optional[float] = None,
        dropout_rate: Optional[float] = None,
    ):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.scale_final = scale_final
        self.dropout_rate = dropout_rate

        # Layer normalization (if used)
        self.layer_norm = nn.LayerNorm(in_dim) if use_layer_norm else None

        # Create a list of layers
        layers = []

        for i, size in enumerate(self.hidden_dims):
            # Add linear layers
            layers.append(nn.Linear(in_dim, size))

            # Add activation function if not final layer or if activate_final is True
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                layers.append(self.activations)

            # Add dropout if specified
            if self.dropout_rate is not None and self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))

            in_dim = size  # Update the input dimension for the next layer

        # Combine the layers into a sequential container
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # Forward pass through the layers
        return self.model(x)


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        act: Callable,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
    ):
        super(MLPResNetBlock, self).__init__()
        self.in_dim = in_dim
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        # Layers
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.layer_norm = nn.LayerNorm(in_dim) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        residual = x

        # Dropout (if specified)
        if self.dropout is not None:
            x = self.dropout(x) if training else x

        # Layer normalization (if specified)
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # MLP forward pass with activation
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        # Adjust residual if needed (for shape mismatch)
        if residual.shape != x.shape:
            residual = self.dense1(residual)  # Project residual to match shape

        # Return the residual connection
        return residual + x


class MLPResNet(nn.Module):
    """MLPResNet network."""

    def __init__(
        self,
        num_blocks: int,
        in_dim: int,
        out_dim: int,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: Callable = F.relu,
    ):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        # Initial dense layer
        self.dense_input = nn.Linear(in_dim, hidden_dim)

        # MLP ResNet Blocks
        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    hidden_dim=hidden_dim * 4,
                    act=self.activations,
                    use_layer_norm=self.use_layer_norm,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output layer
        self.dense_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        # First dense layer
        x = self.dense_input(x)

        # Pass through MLP ResNet blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Apply activation and output layer
        x = self.activations(x)
        x = self.dense_output(x)

        return x


class ScoreModel_test(nn.Module):
    def __init__(
        self,
        data_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        time_emb: int = 128,
        num_blocks: int = 3,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.time_encoder = RandomOrLearnedSinusoidalPosEmb(
            dim=time_emb, learnable=True
        )
        self.cond_encoder = MLP(
            in_dim=time_emb + 1,
            hidden_dims=(time_emb * 2, time_emb * 2),
            activations=nn.SiLU(),
            activate_final=False,
        )

        self.base_model = MLPResNet(
            in_dim=time_emb * 2 + state_dim + data_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            out_dim=data_dim,
            activations=nn.SiLU(),
        )

    def forward(self, x, condition, time, training=True):
        t = self.time_encoder(time)
        cond_emb = self.cond_encoder(t)
        reverse_input = torch.cat([x, condition, cond_emb], dim=-1)
        out = self.base_model(reverse_input, training)

        return out


#### FiLM 1D unet models


class ParentTemporalUnet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        heads=4,
        affine=True,  # ignored if FiLMLayerNorm is True
        scaleResidual=False,
        FiLMLayerNorm=False,
        FiLMAttn=False,
    ):
        super().__init__()

        ResTransform = ScaledResidual if scaleResidual else Residual
        ResTempBlock = (
            ScaledResidualTemporalBlock if scaleResidual else ResidualTemporalBlock
        )
        AttnBlock = FiLMedLinearAttention if FiLMAttn else LinearAttention

        if FiLMLayerNorm:
            PreNormTransform = FiLMedPreNorm
            norm_kwargs = {}
        else:
            PreNormTransform = PreNorm
            norm_kwargs = {"affine": affine}

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {self.in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(self.in_out)

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResTempBlock(
                            dim_in, dim_out, embed_dim=time_dim, horizon=horizon
                        ),
                        ResTempBlock(
                            dim_out, dim_out, embed_dim=time_dim, horizon=horizon
                        ),
                        ResTransform(
                            PreNormTransform(
                                dim_out, AttnBlock(dim_out, heads=heads), **norm_kwargs
                            )
                        )
                        if attention
                        else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResTempBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon
        )
        self.mid_attn = (
            ResTransform(
                PreNormTransform(
                    mid_dim, AttnBlock(mid_dim, heads=heads), **norm_kwargs
                )
            )
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResTempBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResTempBlock(
                            dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon
                        ),
                        ResTempBlock(
                            dim_in, dim_in, embed_dim=time_dim, horizon=horizon
                        ),
                        ResTransform(
                            PreNormTransform(
                                dim_in, AttnBlock(dim_in, heads=heads), **norm_kwargs
                            )
                        )
                        if attention
                        else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time):
        """
        x : [ batch x horizon x transition ]
        """

        # process input x as before
        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)
        h = []

        film_idx = 0
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x


class FiLMTemporalUnet(ParentTemporalUnet):
    def __init__(
        self,
        horizon,
        history_len,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        cond_dim_mults=(1, 2),
        cond_kernel=5,
        *args,
        **kwargs,
    ):
        super().__init__(
            horizon,
            transition_dim,
            cond_dim,
            dim,
            dim_mults,
            *args,
            affine=False,
            **kwargs,
        )

        out_channels = [dim * m for m in cond_dim_mults]
        self.cond_encoder = ResidualEncodingBlock(
            transition_dim, out_channels, cond_kernel
        )
        self.cond_out = history_len * out_channels[-1]

        self.FiLM_layers = nn.ModuleList([])
        for _, dim_out in self.in_out:
            self.FiLM_layers.append(
                nn.Sequential(
                    nn.Linear(self.cond_out, dim_out),
                    nn.Mish(),
                    nn.Linear(dim_out, 2 * dim_out),
                )
            )

        mid_dim = dim_mults[-1] * dim
        self.FiLM_layers.append(
            nn.Sequential(
                nn.Linear(self.cond_out, mid_dim),
                nn.Mish(),
                nn.Linear(mid_dim, 2 * mid_dim),
            )
        )

        for dim_in, _ in reversed(self.in_out[1:]):
            self.FiLM_layers.append(
                nn.Sequential(
                    nn.Linear(self.cond_out, dim_in),
                    nn.Mish(),
                    nn.Linear(dim_in, 2 * dim_in),
                )
            )

        self.FiLM_outputs = []

    def condition_diffusion(self, history):
        """
        Conditioning only needs to be set once through an entire reverse process.
        history : [ batch x cond_horizon x transition ]
        """
        history = einops.rearrange(history, "b h t -> b t h")

        # encode history with temporal convs
        h_embedding = self.cond_encoder(history)
        h_embedding = h_embedding.flatten(start_dim=1)

        # compute gamma and beta
        self.FiLM_outputs = []
        for film in self.FiLM_layers:
            g, b = film(h_embedding)[:, :, None].chunk(chunks=2, dim=1)
            self.FiLM_outputs.append((g, b))

    def conditioning_set(self):
        return len(self.FiLM_outputs) > 0

    def clear_conditioning(self):
        self.FiLM_outputs = []

    def forward(self, x, time):
        """
        x : [ batch x horizon x transition ]
        """

        assert len(self.FiLM_outputs) == len(self.FiLM_layers), (
            "Error: conditioning must be set first."
        )

        # process input x as before
        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)
        h = []

        film_idx = 0
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

            # apply FiLM transformation
            g, b = self.FiLM_outputs[film_idx]
            x = (1 + g) * x + b
            film_idx += 1

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # apply FiLM transformation
        g, b = self.FiLM_outputs[film_idx]
        x = (1 + g) * x + b
        film_idx += 1

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

            # apply FiLM transformation
            g, b = self.FiLM_outputs[film_idx]
            x = (1 + g) * x + b
            film_idx += 1

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x
