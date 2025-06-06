import torch
import torch.nn as nn

from src.models.helpers import (
    RandomOrLearnedSinusoidalPosEmb,
)
from typing import Sequence, Callable, Optional
import torch.nn.functional as F


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


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

    def forward(self, x, state, time, training=True):
        t = self.time_encoder(time)
        cond_emb = self.cond_encoder(t)
        reverse_input = torch.cat([x, state, cond_emb], dim=-1)
        out = self.base_model(reverse_input, training)

        return out
