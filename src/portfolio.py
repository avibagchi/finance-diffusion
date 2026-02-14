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


def _max_drawdown(cumulative_returns):
    """Compute max drawdown from cumulative returns (1 + r_1)(1 + r_2)..."""
    cumprod = np.cumprod(1 + cumulative_returns)
    running_max = np.maximum.accumulate(cumprod)
    drawdowns = (cumprod - running_max) / (running_max + 1e-10)
    return np.min(drawdowns) * 100  # as %


def drawdown_series(returns):
    """Return drawdown series (T,) in percent for plotting."""
    port_returns = np.asarray(returns).flatten()
    cumprod = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cumprod)
    drawdowns = (cumprod - running_max) / (running_max + 1e-10) * 100
    return drawdowns


def _cvar(returns, alpha=0.05):
    """Conditional VaR (Expected Shortfall) at alpha level."""
    var_idx = int(np.ceil(alpha * len(returns))) - 1
    var_idx = max(0, var_idx)
    sorted_ret = np.sort(returns)
    return np.mean(sorted_ret[: var_idx + 1]) * 100 if var_idx >= 0 else 0


def portfolio_metrics(returns, weights_sequence, benchmark_returns=None):
    """
    Compute portfolio metrics from monthly returns and weight sequence.
    returns: (T, D) - realized returns each month
    weights_sequence: (T, D) - portfolio weights each month (omega at start of month)
    benchmark_returns: (T,) optional - for RtC (Return to Capture) calculation
    """
    port_returns = np.sum(weights_sequence * returns, axis=1)
    mean_ret = np.mean(port_returns) * 100  # in %
    std_ret = np.std(port_returns) * 100
    sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0  # annualized

    neg_returns = port_returns[port_returns < 0]
    neg_std = np.std(neg_returns) if len(neg_returns) >= 2 else 0
    sortino = mean_ret / neg_std * np.sqrt(12) if neg_std > 0 else sharpe

    max_dd = _max_drawdown(port_returns)
    ann_ret_pct = mean_ret * 12  # approx annualized return in %
    calmar = ann_ret_pct / (abs(max_dd) + 1e-10) if max_dd < 0 else 0  # annualized ret / |max_dd|%

    cvar_05 = _cvar(port_returns, alpha=0.05)
    rtc = mean_ret / (abs(cvar_05) + 1e-10) if cvar_05 < 0 else 0  # Return to CVaR (tail risk)
    if benchmark_returns is not None:
        up_bench = benchmark_returns > 0
        dn_bench = benchmark_returns < 0
        upside_capture = (
            np.mean(port_returns[up_bench]) / (np.mean(benchmark_returns[up_bench]) + 1e-10) * 100
            if np.any(up_bench) else 0
        )
        downside_capture = (
            np.mean(port_returns[dn_bench]) / (np.mean(benchmark_returns[dn_bench]) + 1e-10) * 100
            if np.any(dn_bench) else 0
        )
        rtc = upside_capture - downside_capture  # Return to Capture spread (upside - downside)

    return {"mean": mean_ret, "std": std_ret, "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "rtc": rtc}


def portfolio_metrics_from_returns(port_returns, benchmark_returns=None):
    """
    Compute portfolio metrics from raw portfolio return series.
    port_returns: (T,) - portfolio returns each period (e.g. after transaction costs)
    benchmark_returns: (T,) optional - for RtC
    """
    port_returns = np.asarray(port_returns).flatten()
    mean_ret = np.mean(port_returns) * 100
    std_ret = np.std(port_returns) * 100
    sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0
    neg_returns = port_returns[port_returns < 0]
    neg_std = np.std(neg_returns) * 100 if len(neg_returns) >= 2 else 0
    sortino = mean_ret / neg_std * np.sqrt(12) if neg_std > 0 else sharpe
    max_dd = _max_drawdown(port_returns)
    ann_ret_pct = mean_ret * 12
    calmar = ann_ret_pct / (abs(max_dd) + 1e-10) if max_dd < 0 else 0
    cvar_05 = _cvar(port_returns, alpha=0.05)
    rtc = mean_ret / (abs(cvar_05) + 1e-10) if cvar_05 < 0 else 0
    if benchmark_returns is not None:
        up_bench = benchmark_returns > 0
        dn_bench = benchmark_returns < 0
        upside_capture = (
            np.mean(port_returns[up_bench]) / (np.mean(benchmark_returns[up_bench]) + 1e-10) * 100
            if np.any(up_bench) else 0
        )
        downside_capture = (
            np.mean(port_returns[dn_bench]) / (np.mean(benchmark_returns[dn_bench]) + 1e-10) * 100
            if np.any(dn_bench) else 0
        )
        rtc = upside_capture - downside_capture
    return {"mean": mean_ret, "std": std_ret, "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "rtc": rtc}
