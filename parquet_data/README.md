# Parquet Financial Factor Data

This directory contains financial factor data converted from parquet format to the panel format required by the finance-diffusion model.

## Directory Structure

```
parquet_data/
├── raw/                          # Raw parquet files
│   └── factors_first_500.parquet # Original factor data
├── processed/                    # Converted panel data
│   └── real_factors/
│       ├── factors.npy          # (T, D, K) factor array
│       ├── returns.npy          # (T, D) returns array
│       ├── prices.npy           # (T, D) price array
│       ├── metadata.json        # Dataset metadata
│       └── csv/                 # CSV exports for easy viewing
│           ├── returns.csv
│           ├── prices.csv
│           ├── factors_long.csv
│           ├── factor_*.csv     # Individual factor files
│           ├── summary_stats.csv
│           └── README.md
├── scripts/                      # Conversion utilities
│   ├── convert_parquet_data.py  # Parquet to numpy conversion
│   └── export_to_csv.py         # Numpy to CSV export
└── README.md                     # This file
```

## Data Format

All processed data follows the standard panel format:

### Dimensions
- **T** = num_days (11 days: 2010-01-04 to 2010-01-15)
- **D** = num_assets (500 stocks)
- **K** = num_factors (10 factors)

### Arrays

#### factors.npy
- Shape: `(T, D, K)` = `(11, 500, 10)`
- Type: `float32`
- Contains 10 financial factors for each stock on each day

#### returns.npy
- Shape: `(T, D)` = `(11, 500)`
- Type: `float32`
- Daily returns for each stock

#### prices.npy
- Shape: `(T, D)` = `(11, 500)`
- Type: `float32`
- Daily prices for each stock

## Factors Included

The dataset includes 10 carefully selected factors:

### Momentum Factors (4)
1. **ret_1_0** - 1-day return
2. **ret_3_1** - 3-month momentum
3. **ret_6_1** - 6-month momentum
4. **ret_12_1** - 12-month momentum

### Valuation Factors (6)
5. **be_me** - Book equity to market equity ratio
6. **at_me** - Total assets to market equity ratio
7. **sale_me** - Sales to market equity ratio
8. **ni_me** - Net income to market equity ratio
9. **ocf_me** - Operating cash flow to market equity ratio
10. **ebitda_mev** - EBITDA to market enterprise value ratio

## Dataset Statistics

```
Period: 2010-01-04 to 2010-01-15
Number of stocks: 500 (identified by gvkey)
Number of trading days: 11
Number of factors: 10

Returns:
  Mean: varies by stock and day
  Range: [-0.802, 4.432]

Factors:
  All factors are normalized/standardized from original parquet data
  Range: [-9.261, 58.270]
```

## Using the Data

### Load with PyTorch

```python
import numpy as np
import torch

# Load data
factors = np.load('parquet_data/processed/real_factors/factors.npy')
returns = np.load('parquet_data/processed/real_factors/returns.npy')
prices = np.load('parquet_data/processed/real_factors/prices.npy')

# Convert to tensors
factors_tensor = torch.from_numpy(factors).float()  # (11, 500, 10)
returns_tensor = torch.from_numpy(returns).float()  # (11, 500)
```

### Load with Custom DataLoader

```python
from src.data_loader import load_real_dataset

# Load dataset with DataLoader
dataset, dataloader = load_real_dataset(
    data_dir='parquet_data/processed/real_factors',
    split='train',
    batch_size=32
)

# Iterate through batches
for factors_batch, returns_batch in dataloader:
    # factors_batch: (batch_size, D, K)
    # returns_batch: (batch_size, D)
    pass
```

### Load CSV Files (for Excel, R, or pandas)

```python
import pandas as pd

# Load returns (wide format)
returns = pd.read_csv('parquet_data/processed/real_factors/csv/returns.csv',
                      index_col='date', parse_dates=True)

# Load prices (wide format)
prices = pd.read_csv('parquet_data/processed/real_factors/csv/prices.csv',
                     index_col='date', parse_dates=True)

# Load all factors (long/stacked format)
factors = pd.read_csv('parquet_data/processed/real_factors/csv/factors_long.csv',
                      parse_dates=['date'])

# Load individual factor (wide format)
ret_1_0 = pd.read_csv('parquet_data/processed/real_factors/csv/factor_ret_1_0.csv',
                      index_col='date', parse_dates=True)

# Load summary statistics
stats = pd.read_csv('parquet_data/processed/real_factors/csv/summary_stats.csv')
```

### Inspect Metadata

```python
import json

with open('parquet_data/processed/real_factors/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Dataset: {metadata['dataset_name']}")
print(f"Period: {metadata['start_date']} to {metadata['end_date']}")
print(f"Stocks: {metadata['num_assets']}")
print(f"Factors: {metadata['factor_names']}")
```

## Exporting to CSV

To export any dataset to CSV format:

```bash
cd parquet_data
python scripts/export_to_csv.py --input processed/your_dataset
```

This creates a `csv/` subdirectory with:
- `returns.csv` - Daily returns (wide format)
- `prices.csv` - Daily prices (wide format)
- `factors_long.csv` - All factors (long/stacked format)
- `factor_*.csv` - Individual factor files (wide format)
- `summary_stats.csv` - Statistical summary
- `README.md` - CSV-specific documentation

## Converting New Data

To convert additional parquet files:

```bash
cd parquet_data
python scripts/convert_parquet_data.py \
  --input raw/your_file.parquet \
  --output processed/your_dataset \
  --name your_dataset \
  --max_factors 10
```

### Conversion Script Options

- `--input`: Path to input parquet file (required)
- `--output`: Output directory for processed data (default: processed/real_factors)
- `--name`: Dataset name for metadata (default: real_factors)
- `--max_factors`: Maximum number of factors to extract (default: 20)

### Factor Selection Criteria

The conversion script automatically selects factors based on:
1. **Data coverage**: Factors must have >50% non-null values
2. **Priority categories**: Momentum > Volatility > Valuation > Quality > Growth
3. **Quality checks**: Missing values are filled using cross-sectional medians

## Data Quality

### Missing Value Handling

- **Returns**: Missing values filled with 0.0
- **Prices**: Forward-filled within each stock
- **Factors**: Cross-sectional median imputation per day

### Normalization

- Factors are used as-is from the parquet file (already normalized in most cases)
- Returns are raw daily returns
- Prices are absolute values (negative prices converted to positive)

## Source Data

The original parquet file (`factors_first_500.parquet`) contains:
- 500 rows (stock observations)
- 418 columns (various financial factors and identifiers)
- Rich set of factors including:
  - Momentum and reversal factors
  - Growth rates
  - Valuation ratios
  - Quality scores
  - Volatility and beta measures
  - Liquidity metrics

## Notes

- The dataset is relatively small (11 days) - suitable for testing/prototyping
- Stock identifiers (gvkey) are stored in metadata as asset_names
- All arrays are saved as float32 for memory efficiency
- Data is stored in chronological order (earliest to latest)
