"""
Compute financial factors from price/return data.

This script takes price and return data and computes standard financial factors:
- Momentum (returns over various windows)
- Volatility (rolling standard deviation)
- Volume-based factors (if volume data available)
- Cross-sectional factors (relative performance)
- Technical indicators

Factors are normalized cross-sectionally each day and saved for use in the
Factor-Based Conditional Diffusion Model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Tuple


def compute_momentum_factors(returns: np.ndarray, windows: List[int]) -> np.ndarray:
    """
    Compute momentum factors (cumulative returns over different windows).

    Args:
        returns: (T, D) array of daily returns
        windows: List of lookback windows (e.g., [5, 10, 21, 63, 126])

    Returns:
        factors: (T, D, len(windows)) array of momentum factors
    """
    T, D = returns.shape
    factors = np.zeros((T, D, len(windows)))

    for i, window in enumerate(windows):
        for t in range(T):
            if t < window:
                # Not enough history - use available data
                factors[t, :, i] = np.sum(returns[:t+1], axis=0)
            else:
                # Compute returns over the window
                factors[t, :, i] = np.sum(returns[t-window+1:t+1], axis=0)

    return factors


def compute_volatility_factors(returns: np.ndarray, windows: List[int]) -> np.ndarray:
    """
    Compute volatility factors (rolling standard deviation).

    Args:
        returns: (T, D) array of daily returns
        windows: List of lookback windows (e.g., [5, 10, 21, 63])

    Returns:
        factors: (T, D, len(windows)) array of volatility factors
    """
    T, D = returns.shape
    factors = np.zeros((T, D, len(windows)))

    for i, window in enumerate(windows):
        for t in range(T):
            if t < window:
                # Not enough history - use available data
                factors[t, :, i] = np.std(returns[:t+1], axis=0)
            else:
                # Compute volatility over the window
                factors[t, :, i] = np.std(returns[t-window+1:t+1], axis=0)

    return factors


def compute_relative_strength(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Compute relative strength (asset return vs cross-sectional mean).

    Args:
        returns: (T, D) array of daily returns
        window: Lookback window

    Returns:
        factors: (T, D, 1) array of relative strength
    """
    T, D = returns.shape
    factors = np.zeros((T, D, 1))

    for t in range(T):
        if t < window:
            cum_returns = np.sum(returns[:t+1], axis=0)
        else:
            cum_returns = np.sum(returns[t-window+1:t+1], axis=0)

        # Relative to cross-sectional mean
        factors[t, :, 0] = cum_returns - np.mean(cum_returns)

    return factors


def compute_price_reversal(returns: np.ndarray, short_window: int = 1, long_window: int = 5) -> np.ndarray:
    """
    Compute short-term reversal (recent return vs longer-term return).

    Args:
        returns: (T, D) array of daily returns
        short_window: Short-term window
        long_window: Long-term window

    Returns:
        factors: (T, D, 1) array of reversal factors
    """
    T, D = returns.shape
    factors = np.zeros((T, D, 1))

    for t in range(T):
        # Short-term return
        if t < short_window:
            short_ret = np.sum(returns[:t+1], axis=0)
        else:
            short_ret = np.sum(returns[t-short_window+1:t+1], axis=0)

        # Long-term return
        if t < long_window:
            long_ret = np.sum(returns[:t+1], axis=0)
        else:
            long_ret = np.sum(returns[t-long_window+1:t+1], axis=0)

        # Reversal = long_term - short_term (assets with strong recent drop but weak overall)
        factors[t, :, 0] = long_ret - short_ret

    return factors


def compute_max_drawdown(returns: np.ndarray, window: int = 63) -> np.ndarray:
    """
    Compute maximum drawdown over rolling window.

    Args:
        returns: (T, D) array of daily returns
        window: Lookback window

    Returns:
        factors: (T, D, 1) array of max drawdown factors
    """
    T, D = returns.shape
    factors = np.zeros((T, D, 1))

    for t in range(T):
        if t < window:
            window_returns = returns[:t+1]
        else:
            window_returns = returns[t-window+1:t+1]

        # Compute cumulative returns
        cum_returns = np.cumprod(1 + window_returns, axis=0) - 1

        # Max drawdown for each asset
        running_max = np.maximum.accumulate(cum_returns, axis=0)
        drawdown = (cum_returns - running_max) / (1 + running_max)

        factors[t, :, 0] = np.min(drawdown, axis=0)

    return factors


def normalize_factors_cross_sectionally(factors: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
    """
    Normalize factors cross-sectionally (across assets) at each time step.

    Args:
        factors: (T, D, K) array of factors
        clip_std: Winsorize at this many standard deviations

    Returns:
        normalized_factors: (T, D, K) array
    """
    T, D, K = factors.shape
    normalized = np.zeros_like(factors)

    for t in range(T):
        for k in range(K):
            factor_t_k = factors[t, :, k]

            # Normalize
            mean = np.mean(factor_t_k)
            std = np.std(factor_t_k)

            if std > 1e-8:
                normalized[t, :, k] = (factor_t_k - mean) / std
            else:
                normalized[t, :, k] = 0.0

            # Winsorize
            normalized[t, :, k] = np.clip(normalized[t, :, k], -clip_std, clip_std)

    return normalized


def compute_all_factors(returns: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all financial factors.

    Args:
        returns: (T, D) array of daily returns

    Returns:
        factors: (T, D, K) array of all factors
        factor_names: List of factor names
    """
    print("Computing factors...")

    # Define factor configurations
    momentum_windows = [5, 10, 21, 63]  # 1w, 2w, 1m, 3m
    volatility_windows = [5, 21, 63]    # 1w, 1m, 3m

    factor_names = []
    all_factors = []

    # Momentum factors
    print("  - Momentum factors...")
    momentum = compute_momentum_factors(returns, momentum_windows)
    all_factors.append(momentum)
    factor_names.extend([f'momentum_{w}d' for w in momentum_windows])

    # Volatility factors
    print("  - Volatility factors...")
    volatility = compute_volatility_factors(returns, volatility_windows)
    all_factors.append(volatility)
    factor_names.extend([f'volatility_{w}d' for w in volatility_windows])

    # Relative strength
    print("  - Relative strength...")
    rel_strength = compute_relative_strength(returns, window=21)
    all_factors.append(rel_strength)
    factor_names.append('relative_strength_21d')

    # Price reversal
    print("  - Price reversal...")
    reversal = compute_price_reversal(returns, short_window=1, long_window=5)
    all_factors.append(reversal)
    factor_names.append('reversal_1d_5d')

    # Max drawdown
    print("  - Max drawdown...")
    max_dd = compute_max_drawdown(returns, window=63)
    all_factors.append(max_dd)
    factor_names.append('max_drawdown_63d')

    # Concatenate all factors
    factors = np.concatenate(all_factors, axis=2)

    print(f"  Total factors: {factors.shape[2]}")

    return factors, factor_names


def process_dataset(dataset_name: str):
    """
    Process a single dataset: load returns, compute factors, save results.

    Args:
        dataset_name: Name of dataset directory in data/processed/
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}\n")

    # Load existing data
    data_dir = Path(f'data/processed/{dataset_name}')

    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        return False

    # Load returns
    returns = np.load(data_dir / 'returns.npy')  # (T, D)
    print(f"Loaded returns: {returns.shape}")

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Compute factors
    factors, factor_names = compute_all_factors(returns)
    print(f"\nComputed factors: {factors.shape}")

    # Normalize cross-sectionally
    print("Normalizing factors cross-sectionally...")
    factors = normalize_factors_cross_sectionally(factors)

    # Save factors
    np.save(data_dir / 'factors.npy', factors.astype(np.float32))

    # Update metadata
    metadata['num_factors'] = int(factors.shape[2])
    metadata['factor_names'] = factor_names

    # Compute factor statistics
    factor_stats = {
        'mean': float(factors.mean()),
        'std': float(factors.std()),
        'min': float(factors.min()),
        'max': float(factors.max()),
        'per_factor_stats': {}
    }

    for i, name in enumerate(factor_names):
        factor_stats['per_factor_stats'][name] = {
            'mean': float(factors[:, :, i].mean()),
            'std': float(factors[:, :, i].std()),
            'min': float(factors[:, :, i].min()),
            'max': float(factors[:, :, i].max())
        }

    metadata['factor_stats'] = factor_stats

    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved factors to {data_dir / 'factors.npy'}")
    print(f"  Shape: {factors.shape}")
    print(f"  Factor names: {factor_names}")
    print(f"  Stats: mean={factors.mean():.4f}, std={factors.std():.4f}")

    return True


def main():
    """Process all datasets in data/processed/"""
    processed_dir = Path('data/processed')

    if not processed_dir.exists():
        print("ERROR: data/processed directory does not exist")
        print("Please run create_datasets.py first")
        return

    # Find all dataset directories
    datasets = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    if not datasets:
        print("ERROR: No datasets found in data/processed/")
        return

    print("Found datasets:", datasets)
    print(f"Processing {len(datasets)} datasets...\n")

    successful = []
    failed = []

    for dataset_name in datasets:
        try:
            success = process_dataset(dataset_name)
            if success:
                successful.append(dataset_name)
            else:
                failed.append(dataset_name)
        except Exception as e:
            print(f"✗ ERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(dataset_name)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(successful)} - {successful}")
    if failed:
        print(f"✗ Failed: {len(failed)} - {failed}")
    print(f"\nFactors saved to data/processed/<dataset>/factors.npy")
    print("="*60)


if __name__ == '__main__':
    main()
