"""
DDPM forward and reverse diffusion process for stock returns.
"""
import torch
import torch.nn as nn
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in Improved DDPM."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear variance schedule (original DDPM)."""
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    """
    DDPM for stock return generation.
    Forward: R^(n) = sqrt(alpha_bar_n) * R^(0) + sqrt(1-alpha_bar_n) * epsilon
    """

    def __init__(self, timesteps=1000, beta_schedule="linear"):
        super().__init__()
        self.timesteps = timesteps
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to x_start.
        x_start: (batch, num_assets) - clean returns
        t: (batch,) - timesteps
        returns: (batch, num_assets) - noisy returns
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        # Reshape for broadcasting: (batch,) -> (batch, 1)
        if x_start.dim() == 2:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_loss(self, model, x_start, factors=None, t=None):
        """
        Training loss: MSE between true noise and predicted noise.
        factors: optional (batch, num_assets, num_factors). If None, model is unconditional (UncondDiT).
        """
        batch_size = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()

        noise = torch.randn_like(x_start, device=x_start.device)
        x_noisy = self.q_sample(x_start, t, noise)

        if factors is None:
            predicted_noise = model(x_noisy, t.float())
        else:
            predicted_noise = model(x_noisy, t.float(), factors)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t, factors=None):
        """
        Single reverse diffusion step.
        factors: optional. If None, model is unconditional.
        """
        betas = self.betas[t]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas = self.sqrt_recip_alphas[t]

        if x.dim() == 2:
            betas = betas.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
            sqrt_recip_alphas = sqrt_recip_alphas.unsqueeze(-1)

        if factors is None:
            pred = model(x, t.float())
        else:
            pred = model(x, t.float(), factors)
        model_mean = sqrt_recip_alphas * (
            x - betas * pred / sqrt_one_minus_alphas_cumprod
        )

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance = self.posterior_variance[t]
            if x.dim() == 2:
                posterior_variance = posterior_variance.unsqueeze(-1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample(self, model, factors, num_assets, device):
        """
        Full reverse process: sample from learned distribution (conditional).
        factors: (batch, num_assets, num_factors)
        returns: (batch, num_assets) - generated return samples
        """
        batch_size = factors.shape[0]
        x = torch.randn(batch_size, num_assets, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, factors)

        return x

    @torch.no_grad()
    def sample_uncond(self, model, num_assets, device, batch_size):
        """
        Full reverse process for unconditional model (no factors).
        returns: (batch_size, num_assets) - generated return samples
        """
        x = torch.randn(batch_size, num_assets, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, factors=None)

        return x

