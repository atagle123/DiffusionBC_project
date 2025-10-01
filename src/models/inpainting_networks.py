import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from src.utils.arrays import DEVICE
import math

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
)


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=3):
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
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


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
        horizon: int,
        transition_dim: int,
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
            in_dim=time_emb * 2 + horizon * transition_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            out_dim=horizon * transition_dim,
            activations=nn.SiLU(),
        )

        self.horizon = horizon
        self.transition_dim = transition_dim

    def forward(self, x, time, training=True):
        x = x.flatten(start_dim=1)
        t = self.time_encoder(time)
        cond_emb = self.cond_encoder(t)
        reverse_input = torch.cat([x, cond_emb], dim=-1)
        out = self.base_model(reverse_input, training)
        out = out.unflatten(dim=1, sizes=(self.horizon, self.transition_dim))

        return out


class TemporalUnet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        dim=256,
        dim_mults=(1, 2, 4, 8),
        kernel_size=3,
        attention=False,
        heads=4,
        affine=True,
    ):
        super().__init__()

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
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=time_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=time_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                        ),
                        Residual(
                            PreNorm(
                                dim_out,
                                LinearAttention(dim_out, heads=heads),
                                **norm_kwargs,
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
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=time_dim,
            horizon=horizon,
            kernel_size=kernel_size,
        )
        self.mid_attn = (
            Residual(
                PreNorm(mid_dim, LinearAttention(mid_dim, heads=heads), **norm_kwargs)
            )
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=time_dim,
            horizon=horizon,
            kernel_size=kernel_size,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=time_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=time_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                        ),
                        Residual(
                            PreNorm(
                                dim_in,
                                LinearAttention(dim_in, heads=heads),
                                **norm_kwargs,
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
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, *args, **kwargs):
        """
        x : [ batch x horizon x transition ]
        """

        # process input x as before
        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)
        h = []

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


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, p_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, bias=False)
        # self.norm1 = nn.LayerNorm(d_model, bias=True)
        self.t_linear1 = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )  # no dropout on attention weights
        self.dropout1 = nn.Dropout(p=p_drop)

        self.norm2 = nn.LayerNorm(d_model, bias=False)
        # self.norm2 = nn.LayerNorm(d_model, bias=True)
        self.t_linear2 = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(p=p_drop)

    def forward(self, x, t):
        # Attention sublayer
        out = self.norm1(x) + self.t_linear1(t)
        out, _ = self.attn(out, out, out, need_weights=False)
        out = self.dropout1(out) + x

        # Feedforward sublayer
        x = out
        out = self.norm2(out) + self.t_linear2(t)
        out = self.linear2(self.activation(self.linear1(out)))
        out = self.dropout2(out) + x

        return out

    # def forward(self, x):

    #     # Attention sublayer
    #     out = self.norm1(x)
    #     out, _ = self.attn(out, out, out, need_weights=False)
    #     out = self.dropout1(out) + x

    #     # Feedforward sublayer
    #     x = out
    #     out = self.norm2(out)
    #     out = self.linear2(self.activation(self.linear1(out)))
    #     out = self.dropout2(out) + x

    #     return out


class DiT(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        d_model=1024,
        n_heads=8,
        n_layers=3,
        d_ff=None,
        p_drop=0.0,
        learnable_pos=False,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"Transformer error: the number of heads {n_heads} does not evenly the hidden dimension {d_model}"
        )

        d_ff = 4 * d_model if d_ff is None else d_ff

        self.embed = nn.Linear(transition_dim, d_model)
        # self.encoder_blocks = nn.Sequential(*[nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=p_drop, activation="gelu", norm_first=True, batch_first=True) for _ in range(n_layers)])
        self.encoder_blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, d_ff, p_drop) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, transition_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_ff),
            nn.Mish(),
            nn.Linear(d_ff, d_model),
        )

        self.horizon = horizon
        self.d_model = d_model
        self.learnable_pos = learnable_pos

        self.pos_embeddings()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Mish,
            nn.Dropout,
            nn.GELU,
            DiTBlock,
            SinusoidalPosEmb,
            nn.Sequential,
            nn.ModuleList,
            DiT,
        )

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = (
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "out_proj",
            )
            for name in weight_names:
                weights = getattr(module, name)
                if name == "out_proj":
                    weights = weights.weight
                if weights is not None:
                    nn.init.normal_(weights, mean=0.0, std=0.02)

            bias_names = ("in_proj_bias", "bias_k", "bias_v", "out_proj")
            for name in bias_names:
                bias = getattr(module, name)
                if name == "out_proj":
                    bias = bias.bias
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unrecognised module {module} for initialisation")

        if self.learnable_pos:
            nn.init.normal_(
                self.pos_emb, mean=0.0, std=0.02
            )  # initialise the learnable positional embedding

    def forward(self, x, time, *args, **kwargs):
        """
        x : [ batch x horizon x transition ]
        """
        assert x.shape[1] == self.horizon, (
            f"Input sequence length {x.shape[1]} does not match context window length {self.horizon}"
        )

        x = self.embed(x) + self.pos_emb

        time = time[:, None]
        time = self.time_mlp(time)

        for block in self.encoder_blocks:
            x = block(x, time)
        # x = x + time
        # for block in self.encoder_blocks:
        #     x = block(x)

        x = self.out_norm(x)
        x = self.decoder(x)

        return x

    def pos_embeddings(self):
        if self.learnable_pos:
            self.pos_emb = nn.Parameter(
                torch.zeros((1, self.horizon, self.d_model), device=DEVICE)
            )
        else:
            assert self.d_model % 2 == 0
            half_dim = self.d_model // 2
            pos_idx = torch.arange(self.horizon)

            emb = math.log(10000) / half_dim
            emb = torch.exp(torch.arange(half_dim) * -emb)
            emb = pos_idx[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

            self.register_buffer("pos_emb", emb[None, :, :])
