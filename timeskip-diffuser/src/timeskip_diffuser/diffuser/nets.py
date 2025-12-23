"""
Trajectory network components for diffusion planning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim):
        """
        Initialize sinusoidal positional embedding.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for positional embedding.

        Args:
            x: Input tensor of shape (batch,)

        Returns:
            Positional embeddings of shape (batch, dim)
        """
        half_dim = self.dim // 2
        emb = np.log(1000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LayerNorm1d(nn.Module):
    """Layer normalization for 1D convolutions."""

    def __init__(self, dim, eps=1e-5):
        """
        Initialize 1D layer normalization.

        Args:
            dim: Number of channels
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        """
        Forward pass for 1D layer norm.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Normalized tensor of shape (batch, channels, length)
        """
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ResidualBlock(nn.Module):
    """Residual block with group normalization and FiLM conditioning."""

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        """
        Initialize residual block.

        Args:
            dim: Input channels
            dim_out: Output channels
            time_emb_dim: Time embedding dimension
            groups: Number of groups for group normalization
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim), nn.SiLU(), nn.Conv1d(dim, dim_out, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv1d(dim_out, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        """
        Forward pass for residual block.

        Args:
            x: Input tensor of shape (batch, channels, length)
            time_emb: Time embedding of shape (batch, time_emb_dim)

        Returns:
            Output tensor of shape (batch, dim_out, length)
        """
        h = self.block1(x)

        # FiLM conditioning
        scale, shift = self.mlp(time_emb).chunk(2, dim=-1)
        scale = scale[:, :, None]
        shift = shift[:, :, None]
        h = h * (scale + 1) + shift

        h = self.block2(h)
        return h + self.res_conv(x)


class LocalResidualBlock(nn.Module):
    """
    Residual block with LOCAL receptive field.
    Uses small kernels and no downsampling to maintain shift equivariance.
    """

    def __init__(self, dim, dim_out, time_emb_dim, kernel_size=3):
        """
        Initialize local residual block.

        Args:
            dim: Input channels
            dim_out: Output channels
            time_emb_dim: Time embedding dimension
            kernel_size: Size of convolutional kernels
        """
        super().__init__()

        # Time embedding MLP
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))

        # Two conv layers with small kernels (LOCAL receptiveness)
        padding = kernel_size // 2
        self.block1 = nn.Sequential(
            LayerNorm1d(dim),
            nn.Mish(),
            nn.Conv1d(
                dim, dim_out, kernel_size, padding=padding, padding_mode="replicate"
            ),
        )

        self.block2 = nn.Sequential(
            LayerNorm1d(dim_out),
            nn.Mish(),
            nn.Conv1d(
                dim_out, dim_out, kernel_size, padding=padding, padding_mode="replicate"
            ),
        )

        # Residual connection
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        """
        Forward pass for local residual block.

        Args:
            x: Input tensor of shape (batch, channels, length)
            time_emb: Time embedding of shape (batch, time_emb_dim)

        Returns:
            Output tensor of shape (batch, dim_out, length)
        """
        h = self.block1(x)

        # Add time conditioning
        time_cond = self.mlp(time_emb)
        h = h + time_cond[:, :, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class TrajNet(nn.Module):
    """
    Base class for trajectory neural networks.

    Defines the common interface for trajectory denoising models that take
    noisy trajectories and diffusion timesteps as input.
    """

    def __init__(self, state_dim):
        """
        Initialize TrajNet.

        Args:
            state_dim: Dimension of the state space
        """
        super().__init__()
        self.state_dim = state_dim

    def forward(self, x, time):
        """
        Forward pass for trajectory denoising.

        Args:
            x: Noisy trajectory of shape (batch, horizon, state_dim)
            time: Diffusion timestep of shape (batch,)

        Returns:
            Denoised trajectory of shape (batch, horizon, state_dim)

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward method")


class EqNet(TrajNet):
    """
    Shift-Equivariant Network with Local Receptiveness for trajectory composition.
    """

    def __init__(self, state_dim, hidden_dim=128, time_dim=32, n_layers=10):
        """
        Initialize EqNet.

        Args:
            state_dim: Dimension of the state space
            hidden_dim: Hidden layer dimension
            time_dim: Time embedding dimension
            n_layers: Number of residual layers
        """
        super().__init__(state_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

        # Initial projection
        self.init_conv = nn.Conv1d(
            state_dim,
            hidden_dim,
            3,
            padding=1,
            padding_mode="replicate",
        )

        # Stack of local residual blocks (NO downsampling!)
        self.blocks = nn.ModuleList(
            [
                LocalResidualBlock(hidden_dim, hidden_dim, time_dim * 4, kernel_size=3)
                for _ in range(n_layers // 2)
            ]
            + [
                LocalResidualBlock(hidden_dim, hidden_dim, time_dim * 4, kernel_size=5)
                for _ in range(n_layers // 2)
            ]
        )

        # Final projection
        self.final_conv = nn.Sequential(
            LayerNorm1d(hidden_dim), nn.Mish(), nn.Conv1d(hidden_dim, state_dim, 1)
        )

    def forward(self, x, time):
        """
        Eq-Net forward pass.

        Args:
            x: Noisy trajectory of shape (batch, horizon, state_dim)
            time: Diffusion timestep of shape (batch,)

        Returns:
            Denoised trajectory of shape (batch, horizon, state_dim)
        """
        # Transpose to (batch, state_dim, horizon) for 1D conv
        x = x.transpose(1, 2)

        # Time embedding
        t_emb = self.time_mlp(time)

        # Initial projection
        h = self.init_conv(x)

        # Apply residual blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Final projection
        out = self.final_conv(h)

        # Transpose back to (batch, horizon, state_dim)
        return out.transpose(1, 2)


class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension."""

    def __init__(self, dim, heads=4):
        """
        Initialize temporal attention module.

        Args:
            dim: Feature dimension
            heads: Number of attention heads
        """
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Forward pass for temporal attention.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Output tensor of shape (batch, channels, time)
        """
        b, c, t = x.shape  # pylint: disable=unused-variable
        x = rearrange(x, "b c t -> b t c")

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda tensor: rearrange(tensor, "b t (h d) -> b h t d", h=self.heads), qkv
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.to_out(out)

        return rearrange(out, "b t c -> b c t")


class TemporalUNet(TrajNet):
    """1D U-Net for STATE trajectory denoising."""

    def __init__(self, state_dim, hidden_dims=None, time_dim=64):
        """
        Initialize TemporalUNet.

        Args:
            state_dim: Dimension of the state space
            hidden_dims: List of hidden dimensions for each level
                (default: [128, 256, 512])
            time_dim: Time embedding dimension
        """
        super().__init__(state_dim)

        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

        # Initial projection (input is now just state_dim, not state+action)
        self.init_conv = nn.Conv1d(state_dim, hidden_dims[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList([])
        self.encoder_attns = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])

        dims = [hidden_dims[0]] + list(hidden_dims)
        for i in range(len(hidden_dims)):
            self.encoder_blocks.append(
                ResidualBlock(dims[i], dims[i + 1], time_dim * 4)
            )
            self.encoder_attns.append(TemporalAttention(dims[i + 1]))
            self.downsamples.append(nn.Conv1d(dims[i + 1], dims[i + 1], 4, 2, 1))

        # Bottleneck
        mid_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim * 4)
        self.mid_attn = TemporalAttention(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim * 4)

        # Decoder
        self.decoder_blocks = nn.ModuleList([])
        self.decoder_attns = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])

        for i in reversed(range(len(hidden_dims))):
            self.upsamples.append(nn.ConvTranspose1d(dims[i + 1], dims[i + 1], 4, 2, 1))
            self.decoder_blocks.append(
                ResidualBlock(dims[i + 1] * 2, dims[i], time_dim * 4)
            )
            self.decoder_attns.append(TemporalAttention(dims[i]))

        # Final projection (output is state_dim)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[0], state_dim, 3, padding=1),
        )

    def forward(self, x, time):
        """
        Forward pass for TemporalUNet.

        Args:
            x: Noisy trajectory of shape (batch, horizon, state_dim)
            time: Diffusion timestep of shape (batch,)

        Returns:
            Denoised trajectory of shape (batch, horizon, state_dim)
        """
        x = rearrange(x, "b t c -> b c t")

        t_emb = self.time_mlp(time)

        x = self.init_conv(x)

        # Encoder
        skips = []
        for block, attn, downsample in zip(
            self.encoder_blocks, self.encoder_attns, self.downsamples
        ):
            x = block(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for block, attn, upsample in zip(
            self.decoder_blocks, self.decoder_attns, self.upsamples
        ):
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
            x = attn(x)

        x = self.final_conv(x)

        return rearrange(x, "b c t -> b t c")
