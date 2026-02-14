# Factor-Based Conditional Diffusion Model for Portfolio Optimization

Implementation of the NeurIPS 2025 workshop paper "Factor-Based Conditional Diffusion Model for Portfolio Optimization" (Gao et al.). The model learns the cross-sectional distribution of next-month stock returns conditioned on asset-specific factors, using a Diffusion Transformer (DiT) with token-wise conditioning.

## Overview

- **Model**: FactorDiT — DiT architecture adapted for finance with token-wise conditioning on factors
- **Diffusion**: DDPM with linear variance schedule
- **Portfolio**: Mean-variance optimization (with optional transaction costs)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train on synthetic data (default):

```bash
python train.py
```

Options:

```
--num_months 500      # Months of synthetic data
--num_assets 20       # Number of assets (stocks)
--num_factors 10      # Number of factors per asset
--epochs 50           # Training epochs
--batch_size 16       # Batch size
--lr 0.003            # Learning rate
--n_gen_samples 200   # Samples for moment estimation
--gamma 100           # Risk aversion in mean-variance
--save checkpoint.pt  # Save model path
```

## Architecture

- **Input**: Noisy returns `R^(n)` (batch × num_assets), diffusion step `t`, factors `X_t` (batch × num_assets × num_factors)
- **Token-wise conditioning**: Each asset gets `c_i = MLP(x_i) + embed(t)`
- **AdaLN-Zero**: Modulation parameters from `MLP_Ada(c_i)` per token; zero-initialized for stable training
- **Output**: Predicted noise `ε` (batch × num_assets)

## Data

Synthetic data simulates `R_{t+1} = f(X_t) + u_{t+1}` with a linear factor structure. For real data, replace `generate_synthetic_data` in `data.py` with your factor and return series.

## Files

- `model.py` — FactorDiT model
- `diffusion.py` — DDPM forward/reverse process
- `portfolio.py` — Mean-variance optimization (with/without transaction costs)
- `data.py` — Synthetic data generation
- `train.py` — Training and backtest evaluation
