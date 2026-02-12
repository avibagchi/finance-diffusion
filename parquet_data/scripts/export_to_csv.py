"""
Export numpy arrays to CSV format for easier viewing and analysis.

This script converts the binary .npy files to human-readable CSV files.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse


def export_dataset_to_csv(data_dir, output_dir=None):
    """
    Export a dataset from numpy to CSV format.

    Args:
        data_dir: Path to processed data directory (with .npy files)
        output_dir: Output directory for CSV files (default: data_dir/csv/)
    """
    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = data_dir / 'csv'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("EXPORTING TO CSV")
    print(f"{'='*60}\n")
    print(f"Input:  {data_dir}")
    print(f"Output: {output_dir}\n")

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load arrays
    factors = np.load(data_dir / 'factors.npy')  # (T, D, K)
    returns = np.load(data_dir / 'returns.npy')  # (T, D)
    prices = np.load(data_dir / 'prices.npy')    # (T, D)

    T, D, K = factors.shape

    # Get names
    asset_names = metadata.get('asset_names', [f'asset_{i}' for i in range(D)])
    factor_names = metadata.get('factor_names', [f'factor_{i}' for i in range(K)])

    # Create date range
    start_date = pd.to_datetime(metadata['start_date'])
    dates = pd.date_range(start=start_date, periods=T, freq='D')

    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Dimensions: T={T}, D={D}, K={K}\n")

    # Export returns
    print("Exporting returns.csv...")
    returns_df = pd.DataFrame(
        returns,
        index=dates,
        columns=asset_names
    )
    returns_df.index.name = 'date'
    returns_df.to_csv(output_dir / 'returns.csv')
    print(f"  ✓ Shape: {returns_df.shape}")

    # Export prices
    print("Exporting prices.csv...")
    prices_df = pd.DataFrame(
        prices,
        index=dates,
        columns=asset_names
    )
    prices_df.index.name = 'date'
    prices_df.to_csv(output_dir / 'prices.csv')
    print(f"  ✓ Shape: {prices_df.shape}")

    # Export factors (one CSV per factor for readability)
    print(f"Exporting {K} factor CSVs...")
    for k, factor_name in enumerate(factor_names):
        factor_df = pd.DataFrame(
            factors[:, :, k],
            index=dates,
            columns=asset_names
        )
        factor_df.index.name = 'date'

        # Clean factor name for filename
        clean_name = factor_name.replace('/', '_').replace(' ', '_')
        factor_df.to_csv(output_dir / f'factor_{clean_name}.csv')
    print(f"  ✓ Exported {K} factors")

    # Export all factors in long format (more compact)
    print("Exporting factors_long.csv (stacked format)...")

    # Reshape to long format
    factor_data = []
    for t in range(T):
        for d in range(D):
            row = {
                'date': dates[t],
                'asset': asset_names[d]
            }
            for k, factor_name in enumerate(factor_names):
                row[factor_name] = factors[t, d, k]
            factor_data.append(row)

    factors_long_df = pd.DataFrame(factor_data)
    factors_long_df.to_csv(output_dir / 'factors_long.csv', index=False)
    print(f"  ✓ Shape: {factors_long_df.shape}")

    # Export summary statistics
    print("Exporting summary_stats.csv...")

    stats = []

    # Returns stats
    stats.append({
        'variable': 'returns',
        'mean': float(returns.mean()),
        'std': float(returns.std()),
        'min': float(returns.min()),
        'max': float(returns.max()),
        'median': float(np.median(returns))
    })

    # Prices stats
    stats.append({
        'variable': 'prices',
        'mean': float(prices.mean()),
        'std': float(prices.std()),
        'min': float(prices.min()),
        'max': float(prices.max()),
        'median': float(np.median(prices))
    })

    # Factor stats
    for k, factor_name in enumerate(factor_names):
        factor_data = factors[:, :, k]
        stats.append({
            'variable': f'factor_{factor_name}',
            'mean': float(factor_data.mean()),
            'std': float(factor_data.std()),
            'min': float(factor_data.min()),
            'max': float(factor_data.max()),
            'median': float(np.median(factor_data))
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / 'summary_stats.csv', index=False)
    print(f"  ✓ Exported summary statistics")

    # Create README
    print("Creating CSV README...")
    readme_content = f"""# CSV Data Files

This directory contains CSV exports of the {metadata['dataset_name']} dataset.

## Files

### Main Data Files

1. **returns.csv** ({T} × {D})
   - Daily returns for each asset
   - Index: date
   - Columns: asset names
   - Values: decimal returns (0.01 = 1%)

2. **prices.csv** ({T} × {D})
   - Daily prices for each asset
   - Index: date
   - Columns: asset names
   - Values: price levels

3. **factors_long.csv** ({T*D} × {K+2})
   - All factors in long/stacked format
   - Columns: date, asset, {', '.join(factor_names)}
   - More compact than wide format

### Factor Files (Wide Format)

Individual CSV files for each factor ({T} × {D}):
"""

    for factor_name in factor_names:
        clean_name = factor_name.replace('/', '_').replace(' ', '_')
        readme_content += f"- **factor_{clean_name}.csv** - {factor_name} values\n"

    readme_content += f"""
### Summary Statistics

4. **summary_stats.csv**
   - Mean, std, min, max, median for all variables
   - Useful for quick data inspection

## Data Period

- Start: {metadata['start_date']}
- End: {metadata['end_date']}
- Days: {T}
- Assets: {D}
- Factors: {K}

## Loading CSVs

### Python (pandas)

```python
import pandas as pd

# Load returns
returns = pd.read_csv('returns.csv', index_col='date', parse_dates=True)

# Load prices
prices = pd.read_csv('prices.csv', index_col='date', parse_dates=True)

# Load factors (long format)
factors = pd.read_csv('factors_long.csv', parse_dates=['date'])

# Load a specific factor (wide format)
factor_ret_1_0 = pd.read_csv('factor_ret_1_0.csv', index_col='date', parse_dates=True)

# Load summary stats
stats = pd.read_csv('summary_stats.csv')
```

### R

```r
# Load returns
returns <- read.csv('returns.csv', row.names=1)

# Load prices
prices <- read.csv('prices.csv', row.names=1)

# Load factors (long format)
factors <- read.csv('factors_long.csv')

# Load summary stats
stats <- read.csv('summary_stats.csv')
```

### Excel / Google Sheets

All CSV files can be directly opened in Excel or Google Sheets for viewing and analysis.

## Notes

- All CSV files use comma (`,`) as delimiter
- Dates are in YYYY-MM-DD format
- Missing values (if any) are represented as empty cells
- Files are UTF-8 encoded
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    print(f"  ✓ Created README.md")

    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Files saved to: {output_dir}")
    print(f"\nExported files:")
    print(f"  - returns.csv")
    print(f"  - prices.csv")
    print(f"  - factors_long.csv")
    print(f"  - {K} factor_*.csv files")
    print(f"  - summary_stats.csv")
    print(f"  - README.md")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Export numpy arrays to CSV')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with .npy files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for CSV files (default: input/csv/)')
    args = parser.parse_args()

    export_dataset_to_csv(args.input, args.output)


if __name__ == '__main__':
    main()
