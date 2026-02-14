"""
Synthetic data generation for factor-based return model.
Simulates: R_{t+1} = f(X_t) + u_{t+1}
"""
import numpy as np
import torch


def generate_synthetic_data(
    num_months=500,
    num_assets=20,
    num_factors=10,
    factor_loading_scale=0.1,
    noise_std=0.01,
    seed=42,
):
    """
    Generate synthetic factor and return data.
    Returns: factors (T, D, K), returns (T, D)
    """
    np.random.seed(seed)
    T, D, K = num_months, num_assets, num_factors

    # Factor loadings: each asset has random exposure to factors
    # f(X) ≈ X @ beta (linear factor model)
    beta = np.random.randn(D, K) * factor_loading_scale  # (D, K)

    # Generate factor values over time (persistent, cross-sectionally correlated)
    factors = np.zeros((T + 1, D, K))
    factors[0] = np.random.randn(D, K) * 0.5
    phi = 0.95  # persistence
    for t in range(1, T + 1):
        factors[t] = phi * factors[t - 1] + np.random.randn(D, K) * 0.2

    # Returns: R_t = sum over k of beta_ik * X_{i,t-1,k} + noise
    # Simplified: R_t = diag(X_{t-1} @ beta') + noise  (asset i return depends on asset i factors)
    returns = np.zeros((T, D))
    for t in range(T):
        # Conditional mean: for each asset i, sum over k: beta[i,k] * factors[t,i,k]
        cond_mean = np.sum(factors[t] * beta, axis=1)  # (D,)
        # Add correlated noise (simplified: independent)
        noise = np.random.randn(D) * noise_std
        returns[t] = cond_mean + noise

    # factors[t] is known at time t, we predict returns from t to t+1
    # So at time t we have factors[t] and want to predict returns[t] (which is R_{t+1} in paper notation)
    # Our data: (factors[t], returns[t]) for t=0..T-1
    factors = factors[:-1]  # (T, D, K) - factors at start of period
    # Standardize factors across stocks each day
    for t in range(T):
        f = factors[t]
        factors[t] = (f - np.mean(f)) / (np.std(f) + 1e-8)
    # Winsorize at 3 sigma
    factors = np.clip(factors, -3, 3)
    returns = np.clip(returns, -0.1, 0.1)  # winsorize returns

    return factors.astype(np.float32), returns.astype(np.float32)


class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset of (factors_t, returns_t) pairs."""

    def __init__(self, factors, returns):
        """
        factors: (T, D, K)
        returns: (T, D)
        """
        self.factors = torch.from_numpy(factors)
        self.returns = torch.from_numpy(returns)
        self.T = factors.shape[0]

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return self.factors[idx], self.returns[idx]


class ReturnsOnlyDataset(torch.utils.data.Dataset):
    """Dataset of returns only (for implicit-factor / unconditional diffusion)."""

    def __init__(self, returns):
        """
        returns: (T, D)
        """
        self.returns = torch.from_numpy(returns)
        self.T = returns.shape[0]

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return self.returns[idx]
