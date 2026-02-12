"""
Convert parquet with ALL factors - no filtering or selection.

This script converts all available factor columns from the parquet file,
keeping everything in the (T, D, K) format.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


def load_and_convert_all_factors(parquet_path, output_dir, dataset_name="all_factors"):
    """
    Load parquet and convert ALL factors to panel format.

    No filtering - keeps every numeric factor column.
    """
    print(f"\n{'='*60}")
    print("CONVERTING ALL FACTORS FROM PARQUET")
    print(f"{'='*60}\n")

    # Load parquet
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Shape: {df.shape}")

    # Identify column types
    metadata_cols = ['obs_main', 'exch_main', 'common', 'primary_sec', 'permno', 'date',
                     'permco', 'gvkey', 'iid', 'id', 'excntry', 'eom']
    price_ret_cols = ['prc', 'ret']

    # Find stock ID column
    stock_id_col = None
    for col in ['gvkey', 'permno', 'permco', 'id']:
        if col in df.columns and df[col].notna().sum() > 0:
            stock_id_col = col
            break

    if stock_id_col is None:
        raise ValueError("No valid stock identifier found")

    print(f"Using {stock_id_col} as stock identifier")

    # Get ALL factor columns (anything that's not metadata or price/return)
    factor_cols = []
    for col in df.columns:
        if col not in metadata_cols + price_ret_cols + [stock_id_col]:
            # Only include if it's numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                factor_cols.append(col)

    print(f"\nFound {len(factor_cols)} factor columns")
    print(f"Metadata columns: {len(metadata_cols)}")
    print(f"Price/return columns: {len(price_ret_cols)}")
    print(f"Stock ID: {stock_id_col}")

    # Remove rows with missing stock ID
    df = df[df[stock_id_col].notna()].copy()

    # Sort by date and stock
    df = df.sort_values(['date', stock_id_col])

    # Get dimensions
    dates = sorted(df['date'].unique())
    stock_ids = sorted(df[stock_id_col].unique())

    T = len(dates)
    D = len(stock_ids)
    K = len(factor_cols)

    print(f"\nPanel dimensions:")
    print(f"  T = {T} days ({dates[0]} to {dates[-1]})")
    print(f"  D = {D} stocks")
    print(f"  K = {K} factors")

    # Initialize arrays
    prices = np.full((T, D), np.nan)
    returns = np.full((T, D), np.nan)
    factors = np.full((T, D, K), np.nan)

    # Create index mappings
    date_idx = {date: i for i, date in enumerate(dates)}
    stock_idx = {sid: i for i, sid in enumerate(stock_ids)}

    # Fill arrays
    print("\nFilling arrays...")
    for _, row in df.iterrows():
        t = date_idx[row['date']]
        d = stock_idx[row[stock_id_col]]

        # Price (use absolute value if negative)
        if pd.notna(row['prc']):
            prices[t, d] = abs(row['prc'])

        # Return
        if pd.notna(row['ret']):
            returns[t, d] = row['ret']

        # All factors
        for k, factor in enumerate(factor_cols):
            if pd.notna(row[factor]):
                # Handle any type conversions
                try:
                    val = float(row[factor])
                    if not np.isinf(val):
                        factors[t, d, k] = val
                except (ValueError, TypeError):
                    pass

    # Handle missing values
    print("Handling missing values...")

    # Returns: fill with 0
    returns = np.nan_to_num(returns, nan=0.0)

    # Prices: forward fill within each stock
    for d in range(D):
        mask = ~np.isnan(prices[:, d])
        if mask.any():
            last_valid = np.nan
            for t in range(T):
                if not np.isnan(prices[t, d]):
                    last_valid = prices[t, d]
                elif not np.isnan(last_valid):
                    prices[t, d] = last_valid

    # Factors: fill with cross-sectional median
    for k in range(K):
        for t in range(T):
            mask = ~np.isnan(factors[t, :, k])
            if mask.any():
                median = np.median(factors[t, mask, k])
                factors[t, ~mask, k] = median
            else:
                factors[t, :, k] = 0.0

    # Final cleanup
    prices = np.nan_to_num(prices, nan=1.0, posinf=1.0, neginf=1.0)
    factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n✓ Conversion complete")
    print(f"  Prices: {prices.shape}")
    print(f"  Returns: {returns.shape}")
    print(f"  Factors: {factors.shape}")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_dir}...")

    np.save(output_dir / 'prices.npy', prices.astype(np.float32))
    np.save(output_dir / 'returns.npy', returns.astype(np.float32))
    np.save(output_dir / 'factors.npy', factors.astype(np.float32))

    # Create metadata
    metadata = {
        'dataset_name': dataset_name,
        'source': f'All factors from {Path(parquet_path).name}',
        'start_date': str(dates[0]),
        'end_date': str(dates[-1]),
        'num_days': T,
        'num_assets': D,
        'num_factors': K,
        'asset_names': [str(sid) for sid in stock_ids],
        'factor_names': factor_cols,
        'created_at': datetime.now().isoformat(),
        'conversion_note': 'All available factors included - no filtering applied',
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

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved all files")
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Location: {output_dir}")
    print(f"Dimensions: ({T}, {D}, {K})")
    print(f"All {K} factors preserved!")
    print(f"{'='*60}\n")

    return factors, returns, prices, factor_cols


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert parquet with ALL factors')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input parquet file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--name', type=str, default='all_factors',
                       help='Dataset name')
    args = parser.parse_args()

    load_and_convert_all_factors(args.input, args.output, args.name)
