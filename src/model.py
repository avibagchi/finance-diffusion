"""
Factor-based Conditional Diffusion Transformer (FactorDiT) for portfolio optimization.
Adapts DiT architecture with token-wise conditioning on asset-specific factors.
"""
import torch
import torch.nn as nn
import math
import numpy as np


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FactorEmbedder(nn.Module):
    """Embeds asset-specific factor vectors. Token-wise conditioning."""

    def __init__(self, num_factors, hidden_size, num_layers=2):
        super().__init__()
        layers = []
        in_dim = num_factors
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.SiLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, hidden_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, factors):
        """
        factors: (batch, num_assets, num_factors)
        returns: (batch, num_assets, hidden_size)
        """
        return self.mlp(factors)


class MultiheadAttention(nn.Module):
    """Multi-head self-attention for cross-asset dependencies."""

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FactorDiTBlock(nn.Module):
    """DiT block with token-wise AdaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # Token-wise conditioning: c is (B, N, D) -> 6 modulations per token
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        """
        x: (batch, num_assets, hidden_size)
        c: (batch, num_assets, hidden_size) - token-wise condition
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FactorDiTFinalLayer(nn.Module):
    """Final layer with token-wise AdaLN."""

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 1, bias=True)  # Output 1 value per asset (noise prediction)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # (batch, num_assets, 1)
        return x.squeeze(-1)  # (batch, num_assets)


class FactorDiT(nn.Module):
    """
    Factor-based Conditional Diffusion Transformer for stock return generation.
    Each asset is a token; conditioning is token-wise from asset-specific factors.
    """

    def __init__(
        self,
        num_assets,
        num_factors,
        hidden_size=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_factors = num_factors
        self.hidden_size = hidden_size

        # Input: each asset's noisy return -> embed to hidden_size
        self.input_proj = nn.Linear(1, hidden_size)

        # Token-wise conditioning: factor vector -> condition embedding
        self.factor_embedder = FactorEmbedder(num_factors, hidden_size)
        self.timestep_embedder = TimestepEmbedder(hidden_size)

        # Optional learnable position encoding for asset order
        self.pos_embed = nn.Parameter(torch.zeros(1, num_assets, hidden_size))

        self.blocks = nn.ModuleList([
            FactorDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FactorDiTFinalLayer(hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        nn.init.normal_(self.factor_embedder.mlp[-1].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-init AdaLN modulation (AdaLN-Zero)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, noisy_returns, t, factors):
        """
        noisy_returns: (batch, num_assets) - noisy return at diffusion step
        t: (batch,) - diffusion timestep
        factors: (batch, num_assets, num_factors) - asset factors at time t

        returns: (batch, num_assets) - predicted noise epsilon
        """
        B, D = noisy_returns.shape
        # Embed noisy returns: (B, D) -> (B, D, 1) -> (B, D, hidden)
        x = self.input_proj(noisy_returns.unsqueeze(-1))
        x = x + self.pos_embed

        # Token-wise condition: c_i = MLP(x_i) + embed(t)
        t_emb = self.timestep_embedder(t)  # (B, hidden)
        factor_emb = self.factor_embedder(factors)  # (B, D, hidden)
        # Broadcast t_emb to each token: (B, hidden) -> (B, 1, hidden) -> (B, D, hidden)
        c = factor_emb + t_emb.unsqueeze(1)

        for block in self.blocks:
            x = block(x, c)

        return self.final_layer(x, c)
