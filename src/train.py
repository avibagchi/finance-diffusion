#!/usr/bin/env python3
"""
Train Factor-based Conditional Diffusion Model for portfolio optimization.
Uses synthetic data for demonstration.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from model import FactorDiT
from diffusion import GaussianDiffusion
from data import generate_synthetic_data, SyntheticDataset
from portfolio import (
    mean_variance_weights,
    sample_mean_cov,
    james_stein_shrinkage,
    portfolio_metrics,
    mean_variance_with_transaction_costs,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data: .pt file or synthetic
    if args.data_pt:
        print(f"Loading data from {args.data_pt}...")
        data = torch.load(args.data_pt, map_location="cpu")
        factors = data["factors"]
        returns = data["returns"]
        if torch.is_tensor(factors):
            factors = factors.numpy()
        if torch.is_tensor(returns):
            returns = returns.numpy()
        factors = factors.astype(np.float32)
        returns = returns.astype(np.float32)
        T, D, K = factors.shape
        print(f"  Shape: {T} days, {D} assets, {K} factors")
    else:
        print("Generating synthetic data...")
        factors, returns = generate_synthetic_data(
            num_days=args.num_days,
            num_assets=args.num_assets,
            num_factors=args.num_factors,
            seed=args.seed,
        )
        T, D, K = factors.shape
        print(f"  Shape: {T} days, {D} assets, {K} factors")

    # Train/val split (chronological)
    split = int(T * 0.8)
    train_factors, train_returns = factors[:split], returns[:split]
    test_factors, test_returns = factors[split:], returns[split:]

    dataset = SyntheticDataset(train_factors, train_returns)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Model and diffusion
    model = FactorDiT(
        num_assets=D,
        num_factors=K,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_factors, batch_returns in loader:
            batch_factors = batch_factors.to(device)
            batch_returns = batch_returns.to(device)
            loss = diffusion.p_loss(model, batch_returns, batch_factors)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}  Loss: {avg_loss:.6f}")

    # Evaluation: backtest on test set
    model.eval()
    n_test = len(test_returns)
    n_samples = args.n_gen_samples

    # Strategy 1: Factordiff (no transaction costs)
    factordiff_weights = []
    for t in range(n_test):
        f = torch.from_numpy(test_factors[t : t + 1]).float().to(device)
        f_batch = f.repeat(n_samples, 1, 1)
        with torch.no_grad():
            samples = diffusion.sample(model, f_batch, D, device)
        mu, Sigma = sample_mean_cov(samples.cpu().numpy())
        w = mean_variance_weights(mu, Sigma, gamma=args.gamma)
        factordiff_weights.append(w)
    factordiff_weights = np.array(factordiff_weights)

    # Strategy 2: Empirical
    emp_weights = []
    for t in range(n_test):
        hist_returns = train_returns if t == 0 else np.vstack([train_returns, test_returns[:t]])
        mu = np.mean(hist_returns, axis=0)
        Sigma = np.cov(hist_returns, rowvar=False) + 1e-6 * np.eye(D)
        w = mean_variance_weights(mu, Sigma, gamma=args.gamma)
        emp_weights.append(w)
    emp_weights = np.array(emp_weights)

    # Strategy 3: Shrinkage
    shr_weights = []
    for t in range(n_test):
        hist_returns = train_returns if t == 0 else np.vstack([train_returns, test_returns[:t]])
        mu = np.mean(hist_returns, axis=0)
        Sigma = np.cov(hist_returns, rowvar=False) + 1e-6 * np.eye(D)
        mu, Sigma = james_stein_shrinkage(mu, Sigma)
        w = mean_variance_weights(mu, Sigma, gamma=args.gamma)
        shr_weights.append(w)
    shr_weights = np.array(shr_weights)

    # Strategy 4: Equal weight
    ew_weights = np.ones((n_test, D)) / D

    # Compute metrics (no transaction costs)
    def eval_weights(weights):
        return portfolio_metrics(test_returns, weights)

    print("\n" + "=" * 60)
    print("Portfolio Performance (no transaction costs)")
    print("=" * 60)
    for name, w in [
        ("EW", ew_weights),
        ("Factordiff", factordiff_weights),
        ("Emp", emp_weights),
        ("ShrEmp", shr_weights),
    ]:
        m = eval_weights(w)
        print(f"{name:12}  Mean: {m['mean']:.4f}%  Std: {m['std']:.4f}%  Sharpe: {m['sharpe']:.4f}")

    # With transaction costs
    buy_cost, sell_cost = 0.00075, 0.00125
    factordiff_weights_tc = []
    omega_prev = np.ones(D) / D
    for t in range(n_test):
        f = torch.from_numpy(test_factors[t : t + 1]).float().to(device)
        f_batch = f.repeat(n_samples, 1, 1)
        with torch.no_grad():
            samples = diffusion.sample(model, f_batch, D, device)
        mu, Sigma = sample_mean_cov(samples.cpu().numpy())
        w = mean_variance_with_transaction_costs(
            mu, Sigma, omega_prev, gamma=args.gamma,
            buy_cost=buy_cost, sell_cost=sell_cost,
        )
        factordiff_weights_tc.append(w)
        omega_prev = w
    factordiff_weights_tc = np.array(factordiff_weights_tc)

    # Deduct transaction costs from returns for Factordiff
    port_ret = np.sum(factordiff_weights_tc * test_returns, axis=1)
    prev_w = np.vstack([np.ones(D) / D, factordiff_weights_tc[:-1]])
    turnover = np.sum(np.abs(factordiff_weights_tc - prev_w), axis=1)
    tc_per_day = buy_cost * np.sum(np.maximum(factordiff_weights_tc - prev_w, 0), axis=1)
    tc_per_day += sell_cost * np.sum(np.maximum(prev_w - factordiff_weights_tc, 0), axis=1)
    port_ret_net = port_ret - tc_per_day
    m_factordiff_tc = {
        "mean": np.mean(port_ret_net) * 100,
        "std": np.std(port_ret_net) * 100,
        "sharpe": np.mean(port_ret_net) / (np.std(port_ret_net) + 1e-8) * np.sqrt(252),
    }

    print("\nWith transaction costs (Factordiff):")
    print(f"  Mean: {m_factordiff_tc['mean']:.4f}%  Std: {m_factordiff_tc['std']:.4f}%  Sharpe: {m_factordiff_tc['sharpe']:.4f}")

    # Save model
    if args.save:
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
        }, args.save)
        print(f"\nModel saved to {args.save}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", type=int, default=500)
    parser.add_argument("--num_assets", type=int, default=20)
    parser.add_argument("--num_factors", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=100.0)
    parser.add_argument("--n_gen_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoint.pt")
    # Data source: .pt file or synthetic
    parser.add_argument("--data_pt", type=str, default=None, help="Path to .pt file with keys 'factors' (T,D,K) and 'returns' (T,D)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
