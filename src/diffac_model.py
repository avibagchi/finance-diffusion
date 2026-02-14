"""
Diffac-style implicit diffusion with score decomposition.
Implements the architecture from Chen et al. (2025) "Diffusion Factor Models":
  s_θ(r,t) = α_t D_t V · g_ζ(V' D_t r, t) - D_t r
  - Subspace score: α_t D_t V g(...)  (k-dimensional nonlinear part)
  - Complement score: -D_t r  (linear, skip connection)
"""
import torch
import torch.nn as nn
import math
import numpy as np


def orthogonal_init_(tensor):
    """Initialize tensor with orthonormal rows (for V with orthonormal columns)."""
    with torch.no_grad():
        nn.init.orthogonal_(tensor)


class DiffacImplicitDiT(nn.Module):
    """
    Implicit diffusion model with score decomposition (diffac/Diffusion Factor Model).
    Score: s_θ(r,t) = α_t D_t V · g_ζ(V' D_t r, t) - D_t r
    Outputs predicted noise ε for DDPM compatibility: pred_eps = -s_θ * sqrt(h_t)
    """

    def __init__(
        self,
        num_assets,
        num_factors,
        hidden_size=256,
        timesteps=1000,
        beta_schedule="linear",
        sigma_max=1.0,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_factors = num_factors
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.sigma_max = sigma_max

        # Factor loading V ∈ R^{d×k} with orthonormal columns
        self.V = nn.Parameter(torch.randn(num_assets, num_factors))
        # Per-asset idiosyncratic variance c ∈ [0, sigma_max]^d (Λ_t = diag(h_t + α_t² c_i))
        self.log_c = nn.Parameter(torch.zeros(num_assets))

        # Subspace MLP g_ζ: R^k × time -> R^k
        self.time_embed = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.subspace_mlp = nn.Sequential(
            nn.Linear(num_factors + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_factors),
        )

        # Precompute diffusion schedule
        self._setup_schedule(timesteps, beta_schedule)
        self.initialize_weights()

    def _setup_schedule(self, timesteps, beta_schedule):
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, timesteps)
        else:
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((t / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        h_t = 1.0 - alphas_cumprod  # variance of added noise
        alpha_t = torch.sqrt(alphas_cumprod)  # signal retention

        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("h_t", h_t)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _timestep_embed(self, t):
        """Sinusoidal embedding for continuous t in [0,1]."""
        half_dim = 32
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.time_embed(emb)

    def _ensure_V_orthonormal(self):
        """Apply Gram-Schmidt to keep V orthonormal during training."""
        with torch.no_grad():
            V = self.V
            for j in range(self.num_factors):
                v = V[:, j]
                for i in range(j):
                    v = v - (V[:, i] @ v) * V[:, i]
                v = v / (torch.norm(v) + 1e-8)
                V[:, j] = v

    def forward(self, noisy_returns, t):
        """
        noisy_returns: (batch, num_assets)
        t: (batch,) - diffusion timestep in [0, timesteps-1]
        returns: (batch, num_assets) - predicted noise epsilon (for DDPM)
        """
        B, D = noisy_returns.shape
        device = noisy_returns.device

        # Clamp t to valid range
        t_idx = t.long().clamp(0, self.timesteps - 1)
        alpha_t = self.alpha_t[t_idx]  # (B,)
        h_t = self.h_t[t_idx]  # (B,)
        sqrt_h_t = torch.sqrt(h_t + 1e-8)

        # Ensure V orthonormal
        self._ensure_V_orthonormal()
        V = self.V  # (D, k)

        # c_i ∈ [ε, sigma_max] for stability
        c = torch.sigmoid(self.log_c) * self.sigma_max + 1e-4  # (D,)

        # D_t = diag(1 / (h_t + α_t² c_i))
        # alpha_t, h_t are (B,), c is (D,) -> we need (B, D)
        alpha_t_2 = alpha_t.unsqueeze(-1) ** 2  # (B, 1)
        h_t_exp = h_t.unsqueeze(-1)  # (B, 1)
        denom = h_t_exp + alpha_t_2 * c.unsqueeze(0)  # (B, D)
        D_t = 1.0 / (denom + 1e-8)  # (B, D)

        # Encoder: z = V' D_t r  ->  (B, k)
        r_scaled = noisy_returns * D_t  # (B, D)
        z = r_scaled @ V  # (B, k)

        # Time embedding
        t_norm = t_idx.float() / max(self.timesteps - 1, 1)  # [0, 1]
        t_emb = self._timestep_embed(t_norm)  # (B, hidden)

        # Subspace network: g(z, t) -> (B, k)
        g_in = torch.cat([z, t_emb], dim=-1)
        g_out = self.subspace_mlp(g_in)  # (B, k)

        # Decoder: α_t D_t V g  ->  (B, D)
        alpha_t_b = alpha_t.unsqueeze(-1)
        subspace_score = alpha_t_b * D_t * (g_out @ V.T)  # (B, D)

        # Complement score: -D_t r
        complement_score = -D_t * noisy_returns

        # Full score: s_θ = subspace + complement
        score = subspace_score + complement_score

        # Convert score to predicted noise for DDPM: ε = -score * sqrt(h_t)
        pred_eps = -score * sqrt_h_t.unsqueeze(-1)

        return pred_eps

    def initialize_weights(self):
        orthogonal_init_(self.V)
        nn.init.zeros_(self.log_c)  # Start with small c
        for m in self.subspace_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.time_embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
