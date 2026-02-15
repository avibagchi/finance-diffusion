#!/usr/bin/env python3
"""
Train Factor-based Conditional Diffusion Model for portfolio optimization.
Uses synthetic data for demonstration.
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import FactorDiT, UncondDiT
from diffac_model import DiffacImplicitDiT
from diffusion import GaussianDiffusion
from data import generate_synthetic_data, SyntheticDataset, ReturnsOnlyDataset
from portfolio import (
    mean_variance_weights,
    sample_mean_cov,
    james_stein_shrinkage,
    portfolio_metrics,
    portfolio_metrics_from_returns,
    mean_variance_with_transaction_costs,
    drawdown_series,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    implicit = getattr(args, "implicit", False)

    # --- Data loading ---
    if implicit:
        # Implicit-factor (diffac-style): use returns only; no observed factors
        if args.data_pt:
            print(f"Loading returns from {args.data_pt} (implicit-factor mode)...")
            data = torch.load(args.data_pt, map_location="cpu")
            returns = data["returns"]
            if torch.is_tensor(returns):
                returns = returns.numpy()
            returns = returns.astype(np.float32)
            T, D = returns.shape
            print(f"  Shape: {T} months, {D} assets (no factors)")
        else:
            print("Generating synthetic data (implicit mode: using returns only)...")
            factors, returns = generate_synthetic_data(
                num_months=args.num_months,
                num_assets=args.num_assets,
                num_factors=args.num_factors,
                seed=args.seed,
            )
            T, D = returns.shape[0], returns.shape[1]
            print(f"  Shape: {T} months, {D} assets")
        split = int(T * 0.8)
        train_returns = returns[:split]
        test_returns = returns[split:]
        train_factors = None
        test_factors = None
    else:
        # Conditional: factors + returns
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
            T, D, K_full = factors.shape
            # Use first args.num_factors columns so sweep actually changes the model (0 or >= K_full = use all)
            if 1 <= args.num_factors < K_full:
                factors = factors[:, :, : args.num_factors]
                K = factors.shape[2]
                print(f"  Shape: {T} months, {D} assets, {K} factors (subset of {K_full})")
            else:
                K = K_full
                print(f"  Shape: {T} months, {D} assets, {K} factors")
        else:
            print("Generating synthetic data...")
            factors, returns = generate_synthetic_data(
                num_months=args.num_months,
                num_assets=args.num_assets,
                num_factors=args.num_factors,
                seed=args.seed,
            )
            T, D, K = factors.shape
            print(f"  Shape: {T} months, {D} assets, {K} factors")
        split = int(T * 0.8)
        train_factors, train_returns = factors[:split], returns[:split]
        test_factors, test_returns = factors[split:], returns[split:]

    # --- Dataset and loader ---
    if implicit:
        dataset = ReturnsOnlyDataset(train_returns)
    else:
        dataset = SyntheticDataset(train_factors, train_returns)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # --- Model and diffusion ---
    diffusion = GaussianDiffusion(timesteps=args.timesteps).to(device)
    use_score_decomp = getattr(args, "use_score_decomp", False)
    if implicit:
        if use_score_decomp:
            # Diffac-style: score decomposition (Chen et al. 2025)
            K_implicit = min(max(1, args.num_factors), D - 1)  # latent factor dim k
            model = DiffacImplicitDiT(
                num_assets=D,
                num_factors=K_implicit,
                hidden_size=args.hidden_size,
                timesteps=args.timesteps,
            ).to(device)
            strategy_name = "Diffac"
        else:
            model = UncondDiT(
                num_assets=D,
                hidden_size=args.hidden_size,
                depth=args.depth,
                num_heads=args.num_heads,
            ).to(device)
            strategy_name = "Diffusion"
    else:
        K = train_factors.shape[2]
        model = FactorDiT(
            num_assets=D,
            num_factors=K,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
        ).to(device)
        strategy_name = "Factordiff"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ---
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in loader:
            if implicit:
                batch_returns = batch.to(device)
                loss = diffusion.p_loss(model, batch_returns, factors=None)
            else:
                batch_factors, batch_returns = batch[0].to(device), batch[1].to(device)
                loss = diffusion.p_loss(model, batch_returns, batch_factors)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}  Loss: {avg_loss:.6f}")

    # --- Evaluation: backtest on test set ---
    model.eval()
    n_test = len(test_returns)
    n_samples = args.n_gen_samples
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
    all_samples_list = []

    if implicit:
        factordiff_weights = []
        for t in range(n_test):
            with torch.no_grad():
                samples = diffusion.sample_uncond(model, D, device, n_samples)
            samples_np = samples.cpu().numpy()
            if output_dir:
                all_samples_list.append(samples_np)
                np.save(os.path.join(samples_dir, f"samples_t{t:05d}.npy"), samples_np)
            mu, Sigma = sample_mean_cov(samples_np)
            w = mean_variance_weights(mu, Sigma, gamma=args.gamma)
            factordiff_weights.append(w)
        factordiff_weights = np.array(factordiff_weights)
    else:
        factordiff_weights = []
        for t in range(n_test):
            f = torch.from_numpy(test_factors[t : t + 1]).float().to(device)
            f_batch = f.repeat(n_samples, 1, 1)
            with torch.no_grad():
                samples = diffusion.sample(model, f_batch, D, device)
            samples_np = samples.cpu().numpy()
            if output_dir:
                all_samples_list.append(samples_np)
                np.save(os.path.join(samples_dir, f"samples_t{t:05d}.npy"), samples_np)
            mu, Sigma = sample_mean_cov(samples_np)
            w = mean_variance_weights(mu, Sigma, gamma=args.gamma)
            factordiff_weights.append(w)
        factordiff_weights = np.array(factordiff_weights)

    if output_dir and all_samples_list:
        np.savez(os.path.join(output_dir, "samples_all.npz"), *all_samples_list)
        print(f"\nSamples saved to {samples_dir} and {os.path.join(output_dir, 'samples_all.npz')}")

    # Empirical
    emp_weights = []
    for t in range(n_test):
        hist_returns = train_returns if t == 0 else np.vstack([train_returns, test_returns[:t]])
        mu = np.mean(hist_returns, axis=0)
        Sigma = np.cov(hist_returns, rowvar=False) + 1e-6 * np.eye(D)
        w = mean_variance_weights(mu, Sigma, gamma=args.gamma)
        emp_weights.append(w)
    emp_weights = np.array(emp_weights)

    # Shrinkage
    shr_weights = []
    for t in range(n_test):
        hist_returns = train_returns if t == 0 else np.vstack([train_returns, test_returns[:t]])
        mu = np.mean(hist_returns, axis=0)
        Sigma = np.cov(hist_returns, rowvar=False) + 1e-6 * np.eye(D)
        mu, Sigma = james_stein_shrinkage(mu, Sigma)
        w = mean_variance_weights(mu, Sigma, gamma=args.gamma)
        shr_weights.append(w)
    shr_weights = np.array(shr_weights)

    ew_weights = np.ones((n_test, D)) / D
    ew_returns = np.sum(ew_weights * test_returns, axis=1)

    def eval_weights(weights, benchmark=None):
        return portfolio_metrics(test_returns, weights, benchmark_returns=benchmark)

    print("\n" + "=" * 60)
    print("Portfolio Performance (no transaction costs)")
    print("=" * 60)
    metrics_list = []
    for name, w in [
        ("EW", ew_weights),
        (strategy_name, factordiff_weights),
        ("Emp", emp_weights),
        ("ShrEmp", shr_weights),
    ]:
        m = eval_weights(w, benchmark=ew_returns)
        metrics_list.append((name, m))
        print(f"{name:12}  Mean: {m['mean']:.4f}%  Std: {m['std']:.4f}%  Sharpe: {m['sharpe']:.4f}  Sortino: {m['sortino']:.4f}  Calmar: {m['calmar']:.4f}  RtC: {m['rtc']:.4f}")

    # With transaction costs (diffusion strategy)
    buy_cost, sell_cost = 0.00075, 0.00125
    factordiff_weights_tc = []
    omega_prev = np.ones(D) / D
    for t in range(n_test):
        if implicit:
            with torch.no_grad():
                samples = diffusion.sample_uncond(model, D, device, n_samples)
        else:
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

    port_ret = np.sum(factordiff_weights_tc * test_returns, axis=1)
    prev_w = np.vstack([np.ones(D) / D, factordiff_weights_tc[:-1]])
    tc_per_month = buy_cost * np.sum(np.maximum(factordiff_weights_tc - prev_w, 0), axis=1)
    tc_per_month += sell_cost * np.sum(np.maximum(prev_w - factordiff_weights_tc, 0), axis=1)
    port_ret_net = port_ret - tc_per_month
    m_factordiff_tc = portfolio_metrics_from_returns(port_ret_net, benchmark_returns=ew_returns)

    print(f"\nWith transaction costs ({strategy_name}):")
    print(f"  Mean: {m_factordiff_tc['mean']:.4f}%  Std: {m_factordiff_tc['std']:.4f}%  Sharpe: {m_factordiff_tc['sharpe']:.4f}  Sortino: {m_factordiff_tc['sortino']:.4f}  Calmar: {m_factordiff_tc['calmar']:.4f}  RtC: {m_factordiff_tc['rtc']:.4f}")

    if args.save:
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
        }, args.save)
        print(f"\nModel saved to {args.save}")

    # Plots (if output_dir specified)
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Suffix from key hyperparameters to avoid overwriting
        implicit = getattr(args, "implicit", False)
        use_sd = getattr(args, "use_score_decomp", False)
        lr_str = str(args.lr).replace(".", "p").replace("-", "m")
        hp_suffix = f"_im{int(implicit)}_sd{int(use_sd)}_k{args.num_factors}_h{args.hidden_size}_d{args.depth}_e{args.epochs}_lr{lr_str}"

        strategies = [
            ("EW", ew_weights),
            (strategy_name, factordiff_weights),
            ("Emp", emp_weights),
            ("ShrEmp", shr_weights),
        ]
        port_returns_dict = {
            name: np.sum(w * test_returns, axis=1) for name, w in strategies
        }

        # 1. Cumulative returns
        cum_data = {}
        for name, rets in port_returns_dict.items():
            cum = np.cumprod(1 + rets) - 1
            cum_data[name] = cum * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, cum in cum_data.items():
            ax.plot(cum, label=name)
        ax.set_xlabel("Month")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_title("Cumulative Returns by Strategy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        base = os.path.join(plots_dir, f"cumulative_returns{hp_suffix}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close()
        cols = ["Month"] + list(cum_data.keys())
        arr = np.column_stack([np.arange(len(cum_data[list(cum_data)[0]])), *[cum_data[n] for n in cum_data]])
        np.savetxt(base + ".csv", arr, delimiter=",", header=",".join(cols), comments="")
        dd_data = {name: drawdown_series(rets) for name, rets in port_returns_dict.items()}
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, dd in dd_data.items():
            ax.plot(dd, label=name)
        ax.set_xlabel("Month")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown by Strategy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        base = os.path.join(plots_dir, f"drawdown{hp_suffix}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close()
        cols = ["Month"] + list(dd_data.keys())
        arr = np.column_stack([np.arange(len(dd_data[list(dd_data)[0]])), *[dd_data[n] for n in dd_data]])
        np.savetxt(base + ".csv", arr, delimiter=",", header=",".join(cols), comments="")

        # 3. Metrics bar chart
        metric_names = ["sharpe", "sortino", "calmar", "rtc"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            names = [m[0] for m in metrics_list]
            values = [m[1][metric] for m in metrics_list]
            bars = ax.bar(names, values, color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"][: len(names)])
            ax.set_ylabel(metric.capitalize())
            ax.set_title(metric.capitalize())
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle("Portfolio Metrics Comparison", fontsize=14)
        fig.tight_layout()
        base = os.path.join(plots_dir, f"metrics_comparison{hp_suffix}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close()
        metrics_arr = np.array([[m[1][mn] for mn in metric_names] for m in metrics_list])
        metrics_names_col = np.array([m[0] for m in metrics_list], dtype=object).reshape(-1, 1)
        np.savetxt(base + ".csv", np.hstack([metrics_names_col, metrics_arr]), delimiter=",", fmt="%s", header="Strategy," + ",".join(metric_names), comments="")

        # TC strategy cumulative and drawdown
        cum_tc = np.cumprod(1 + port_ret_net) - 1
        dd_tc = drawdown_series(port_ret_net)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(cum_tc * 100, label=f"{strategy_name} (with TC)", color="darkgreen")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.set_title(f"{strategy_name} with Transaction Costs")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(dd_tc, color="darkgreen")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown (with TC)")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        base = os.path.join(plots_dir, f"factordiff_tc{hp_suffix}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close()
        tc_arr = np.column_stack([np.arange(len(cum_tc)), cum_tc * 100, dd_tc])
        np.savetxt(base + ".csv", tc_arr, delimiter=",", header="Month,cumulative_return_pct,drawdown_pct", comments="")

        # 5. Weight evolution over time (heatmaps and concentration)
        n_test, D = factordiff_weights.shape
        max_assets_heatmap = min(25, D)

        # 5a. Heatmap: top assets by mean weight for each strategy
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for idx, (name, w) in enumerate(strategies):
            ax = axes[idx]
            mean_w = np.mean(w, axis=0)
            top_idx = np.argsort(mean_w)[::-1][:max_assets_heatmap]
            w_top = w[:, top_idx].T
            im = ax.imshow(w_top * 100, aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(w * 100, 99))
            ax.set_xlabel("Month")
            ax.set_ylabel("Asset (top by mean weight)")
            ax.set_title(f"Weights: {name}")
            ax.set_yticks(np.arange(max_assets_heatmap))
            ax.set_yticklabels([str(i) for i in range(max_assets_heatmap)])
            plt.colorbar(im, ax=ax, label="Weight (%)")
        fig.suptitle("Portfolio Weight Evolution by Strategy (top assets)", fontsize=12)
        fig.tight_layout()
        base = os.path.join(plots_dir, f"weights_heatmap{hp_suffix}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close()

        # 5b. Concentration (HHI = sum of squared weights) over time
        hhi_data = {}
        for name, w in strategies:
            hhi = np.sum(w ** 2, axis=1) * 10000
            hhi_data[name] = hhi
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, hhi in hhi_data.items():
            ax.plot(hhi, label=name)
        ax.set_xlabel("Month")
        ax.set_ylabel("Concentration (HHI × 10⁴)")
        ax.set_title("Portfolio Concentration Over Time (HHI)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        base = os.path.join(plots_dir, f"weights_concentration{hp_suffix}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close()
        cols = ["Month"] + list(hhi_data.keys())
        arr = np.column_stack([np.arange(n_test), *[hhi_data[n] for n in hhi_data]])
        np.savetxt(base + ".csv", arr, delimiter=",", header=",".join(cols), comments="")

        # 5c. Full weight matrices as CSV (one per strategy)
        for name, w in strategies:
            cols_w = ["Month"] + [f"Asset_{j}" for j in range(D)]
            arr_w = np.column_stack([np.arange(n_test), w])
            base_w = os.path.join(plots_dir, f"weights_{name.lower().replace(' ', '_')}{hp_suffix}")
            np.savetxt(base_w + ".csv", arr_w, delimiter=",", header=",".join(cols_w), comments="")

        print(f"\nPlots and CSV data saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_months", type=int, default=500)
    parser.add_argument("--num_assets", type=int, default=20)
    parser.add_argument("--num_factors", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=100.0)
    parser.add_argument("--n_gen_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for samples, plots; if set, saves samples and PDFs")
    # Data source: .pt file or synthetic
    parser.add_argument("--data_pt", type=str, default=None, help="Path to .pt file with keys 'factors' (T,D,K) and 'returns' (T,D)")
    parser.add_argument("--implicit", action="store_true", help="Train on returns only, no factors; model learns distribution implicitly")
    parser.add_argument("--use_score_decomp", action="store_true", help="Use score decomposition (diffac) for implicit mode; requires --implicit")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
