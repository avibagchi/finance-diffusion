"""
Mean-variance portfolio optimization with optional transaction costs.
"""
import torch
import numpy as np
from scipy.optimize import minimize


def mean_variance_weights(mu, Sigma, gamma=100.0):
    """
    Solve: max ω'μ - (γ/2) ω'Σω  s.t. ω'1=1, ω_i ≥ 0
    mu: (D,) mean returns
    Sigma: (D, D) covariance matrix
    gamma: risk aversion
    returns: (D,) optimal weights
    """
    D = len(mu)
    mu = np.asarray(mu).flatten()
    Sigma = np.asarray(Sigma)

    def neg_objective(omega):
        return -(omega @ mu - 0.5 * gamma * omega @ Sigma @ omega)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(D)]
    x0 = np.ones(D) / D

    res = minimize(neg_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def mean_variance_with_transaction_costs(
    mu, Sigma, omega_prev, gamma=100.0, buy_cost=0.00075, sell_cost=0.00125
):
    """
    Solve problem (2) from paper with transaction costs.
    max ω'μ - (γ/2) ω'Σω - (0.075% * buy + 0.125% * sell)
    s.t. ω'1=1, 0≤ω_i≤1, ω - ω_prev = b - s, b≥0, s≥0
    """
    D = len(mu)
    mu = np.asarray(mu).flatten()
    Sigma = np.asarray(Sigma)
    omega_prev = np.asarray(omega_prev).flatten()

    # Variable: [omega (D), buy (D), sell (D)] = 3D
    n = 3 * D

    def neg_objective(x):
        omega = x[:D]
        b, s = x[D : 2 * D], x[2 * D :]
        mean_term = omega @ mu - 0.5 * gamma * omega @ Sigma @ omega
        cost_term = buy_cost * np.sum(b) + sell_cost * np.sum(s)
        return -(mean_term - cost_term)

    def weight_sum(x):
        return np.sum(x[:D]) - 1

    def rebalance_eq(x):
        return x[:D] - omega_prev - x[D : 2 * D] + x[2 * D :]

    constraints = [
        {"type": "eq", "fun": weight_sum},
        {"type": "eq", "fun": rebalance_eq},
    ]
    bounds = [(0, 1)] * D + [(0, 2)] * D + [(0, 2)] * D  # b,s can be large for rebalancing
    x0 = np.zeros(n)
    x0[:D] = omega_prev
    # Ensure feasibility: if omega_prev sums to 1, b=s=0 is feasible
    if np.abs(np.sum(omega_prev) - 1) > 1e-6:
        x0[:D] = np.ones(D) / D

    res = minimize(
        neg_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return res.x[:D]


def sample_mean_cov(samples):
    """Estimate mean and covariance from samples. samples: (n_samples, D)."""
    samples = np.asarray(samples)
    mu = np.mean(samples, axis=0)
    Sigma = np.cov(samples, rowvar=False)
    # Ensure Sigma is positive semi-definite
    if np.any(np.linalg.eigvals(Sigma) < -1e-8):
        Sigma = Sigma + 1e-6 * np.eye(Sigma.shape[0])
    return mu, Sigma


def james_stein_shrinkage(mu, Sigma, shrinkage_target="mean"):
    """
    James-Stein shrinkage estimator for covariance.
    Shrink towards scaled identity or constant correlation.
    """
    D = Sigma.shape[0]
    if shrinkage_target == "mean":
        # Shrink towards diagonal with mean variance
        target = np.mean(np.diag(Sigma)) * np.eye(D)
    else:
        target = np.diag(np.diag(Sigma))

    # Simple constant shrinkage factor (can be improved with Ledoit-Wolf)
    lam = 0.1  # shrinkage intensity
    Sigma_shrunk = (1 - lam) * Sigma + lam * target
    return mu, Sigma_shrunk


def portfolio_metrics(returns, weights_sequence):
    """
    Compute portfolio metrics from daily returns and weight sequence.
    returns: (T, D) - realized returns each day
    weights_sequence: (T, D) - portfolio weights each day (omega at start of day)
    """
    # Portfolio return at t = omega_{t-1}' * r_t (we use weights from previous close)
    # Simplified: assume weights at t are applied to returns at t
    port_returns = np.sum(weights_sequence * returns, axis=1)
    mean_ret = np.mean(port_returns) * 100  # in %
    std_ret = np.std(port_returns) * 100
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0  # annualized
    sortino = mean_ret / np.std(port_returns[port_returns < 0]) * np.sqrt(252) if np.any(port_returns < 0) else sharpe
    return {"mean": mean_ret, "std": std_ret, "sharpe": sharpe, "sortino": sortino}
