"""
Convert parquet factor data to our dataset format.

This script takes the rich factor parquet file and converts it to the format
used by the finance-diffusion model:
- prices.npy: (T, D) array of prices
- returns.npy: (T, D) array of returns
- factors.npy: (T, D, K) array of selected factors
- metadata.json: dataset information
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


def load_parquet_data(parquet_path):
    """Load and inspect parquet data."""
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Loaded shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")

    # Check for stock identifier
    stock_id_col = None
    for col in ['permno', 'gvkey', 'id', 'permco']:
        if col in df.columns and df[col].notna().sum() > 0:
            stock_id_col = col
            print(f"Using {col} as stock identifier")
            print(f"Unique stocks: {df[col].nunique()}")
            break

    if stock_id_col is None:
        raise ValueError("No valid stock identifier found")

    return df, stock_id_col


def select_factors(df, max_factors=20):
    """
    Select a subset of high-quality factors.

    Prioritizes:
    - Momentum factors (ret_*)
    - Volatility/Risk (ivol_*, beta_*)
    - Valuation (be_me, *_me ratios)
    - Quality (qmj_*, *_score)
    - Growth (*_gr1, *_gr3)
    """
    # Define factor categories with priority
    factor_groups = {
        'momentum': [
            'ret_1_0', 'ret_3_1', 'ret_6_1', 'ret_12_1', 'ret_60_12'
        ],
        'volatility': [
            'ivol_capm_21d', 'ivol_ff3_21d', 'beta_60m', 'rvol_21d', 'beta_252d'
        ],
        'valuation': [
            'be_me', 'at_me', 'sale_me', 'ni_me', 'ocf_me', 'ebitda_mev'
        ],
        'quality': [
            'qmj', 'f_score', 'o_score', 'roe_be_std', 'profit_cl'
        ],
        'growth': [
            'sale_gr1', 'at_gr1', 'ni_gr1a', 'ebit_gr1a', 'capx_gr1'
        ],
        'size_liquidity': [
            'me', 'dolvol_126d', 'turnover_126d', 'ami_126d'
        ]
    }

    selected_factors = []
    selected_names = []

    print("\n=== SELECTING FACTORS ===")
    for group_name, factor_list in factor_groups.items():
        print(f"\n{group_name.upper()}:")
        for factor in factor_list:
            if factor in df.columns:
                # Check data quality
                non_null_pct = df[factor].notna().mean()
                if non_null_pct > 0.5:  # At least 50% non-null
                    selected_factors.append(factor)
                    selected_names.append(factor)
                    print(f"  ✓ {factor} ({non_null_pct*100:.1f}% coverage)")
                else:
                    print(f"  ✗ {factor} (only {non_null_pct*100:.1f}% coverage)")

            if len(selected_factors) >= max_factors:
                break

        if len(selected_factors) >= max_factors:
            break

    print(f"\nTotal factors selected: {len(selected_factors)}")
    return selected_factors


def convert_to_panel_format(df, stock_id_col, selected_factors):
    """
    Convert long-format parquet to panel format (T, D, K).

    Args:
        df: DataFrame with columns [date, stock_id, factors...]
        stock_id_col: Name of stock identifier column
        selected_factors: List of factor column names

    Returns:
        dates: List of dates (T,)
        stock_ids: List of stock IDs (D,)
        prices: (T, D) array
        returns: (T, D) array
        factors: (T, D, K) array
    """
    print("\n=== CONVERTING TO PANEL FORMAT ===")

    # Remove rows with missing stock ID
    df = df[df[stock_id_col].notna()].copy()

    # Sort by date and stock
    df = df.sort_values(['date', stock_id_col])

    # Get unique dates and stocks
    dates = sorted(df['date'].unique())
    stock_ids = sorted(df[stock_id_col].unique())

    T = len(dates)
    D = len(stock_ids)
    K = len(selected_factors)

    print(f"Panel dimensions: T={T} dates, D={D} stocks, K={K} factors")

    # Initialize arrays
    prices = np.full((T, D), np.nan)
    returns = np.full((T, D), np.nan)
    factors = np.full((T, D, K), np.nan)

    # Create date and stock index mappings
    date_idx = {date: i for i, date in enumerate(dates)}
    stock_idx = {sid: i for i, sid in enumerate(stock_ids)}

    # Fill arrays
    print("Filling arrays...")
    for _, row in df.iterrows():
        t = date_idx[row['date']]
        d = stock_idx[row[stock_id_col]]

        # Price (use absolute value if negative)
        if pd.notna(row['prc']):
            prices[t, d] = abs(row['prc'])

        # Return
        if pd.notna(row['ret']):
            returns[t, d] = row['ret']

        # Factors
        for k, factor in enumerate(selected_factors):
            if pd.notna(row[factor]):
                factors[t, d, k] = row[factor]

    # Fill missing values
    print("\nHandling missing values...")

    # For returns, fill with 0
    returns = np.nan_to_num(returns, nan=0.0)

    # For prices, forward fill within each stock
    for d in range(D):
        mask = ~np.isnan(prices[:, d])
        if mask.any():
            # Forward fill
            last_valid = np.nan
            for t in range(T):
                if not np.isnan(prices[t, d]):
                    last_valid = prices[t, d]
                elif not np.isnan(last_valid):
                    prices[t, d] = last_valid

    # For factors, fill with cross-sectional median
    for k in range(K):
        for t in range(T):
            mask = ~np.isnan(factors[t, :, k])
            if mask.any():
                median = np.median(factors[t, mask, k])
                factors[t, ~mask, k] = median
            else:
                factors[t, :, k] = 0.0

    # Final fillna for any remaining
    prices = np.nan_to_num(prices, nan=1.0)
    factors = np.nan_to_num(factors, nan=0.0)

    print(f"✓ Conversion complete")
    print(f"  Prices shape: {prices.shape}")
    print(f"  Returns shape: {returns.shape}")
    print(f"  Factors shape: {factors.shape}")

    return dates, stock_ids, prices, returns, factors


def save_dataset(output_dir, dates, stock_ids, prices, returns, factors,
                 factor_names, dataset_name="real_factors"):
    """Save dataset in standard format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== SAVING DATASET TO {output_dir} ===")

    # Save arrays
    np.save(output_dir / 'prices.npy', prices.astype(np.float32))
    np.save(output_dir / 'returns.npy', returns.astype(np.float32))
    np.save(output_dir / 'factors.npy', factors.astype(np.float32))

    print(f"✓ Saved prices.npy: {prices.shape}")
    print(f"✓ Saved returns.npy: {returns.shape}")
    print(f"✓ Saved factors.npy: {factors.shape}")

    # Create metadata
    metadata = {
        'dataset_name': dataset_name,
        'source': 'Parquet factor data',
        'start_date': str(dates[0]),
        'end_date': str(dates[-1]),
        'num_days': len(dates),
        'num_assets': len(stock_ids),
        'num_factors': len(factor_names),
        'asset_names': [str(sid) for sid in stock_ids],
        'factor_names': factor_names,
        'created_at': datetime.now().isoformat(),
        'return_stats': {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'min': float(returns.min()),
            'max': float(returns.max())
        },
        'factor_stats': {
            'mean': float(factors.mean()),
            'std': float(factors.std()),
            'min': float(factors.min()),
            'max': float(factors.max())
        }
    }

    # Per-factor statistics
    metadata['per_factor_stats'] = {}
    for i, name in enumerate(factor_names):
        metadata['per_factor_stats'][name] = {
            'mean': float(factors[:, :, i].mean()),
            'std': float(factors[:, :, i].std()),
            'min': float(factors[:, :, i].min()),
            'max': float(factors[:, :, i].max())
        }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata.json")
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Name: {dataset_name}")
    print(f"Period: {dates[0]} to {dates[-1]}")
    print(f"Dimensions: {len(dates)} days × {len(stock_ids)} stocks × {len(factor_names)} factors")
    print(f"Factors: {', '.join(factor_names)}")
    print(f"{'='*60}")


def main():
    """Main conversion pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert parquet factor data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input parquet file')
    parser.add_argument('--output', type=str, default='data/processed/real_factors',
                       help='Output directory')
    parser.add_argument('--name', type=str, default='real_factors',
                       help='Dataset name')
    parser.add_argument('--max_factors', type=int, default=20,
                       help='Maximum number of factors to include')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PARQUET TO PANEL CONVERSION")
    print(f"{'='*60}\n")

    # Load data
    df, stock_id_col = load_parquet_data(args.input)

    # Select factors
    selected_factors = select_factors(df, max_factors=args.max_factors)

    # Convert to panel format
    dates, stock_ids, prices, returns, factors = convert_to_panel_format(
        df, stock_id_col, selected_factors
    )

    # Save dataset
    save_dataset(
        args.output,
        dates,
        stock_ids,
        prices,
        returns,
        factors,
        selected_factors,
        dataset_name=args.name
    )

    print(f"\n✅ Conversion complete!")
    print(f"Dataset saved to: {args.output}")
    print(f"\nYou can now use this dataset with:")
    print(f"  python src/data_loader.py --data_dir {args.output}")


if __name__ == '__main__':
    main()
