# Data Format Specification

This document describes the exact format requirements for data used in the finance-diffusion model.

## Required Format

All datasets must follow this panel data structure:

### Array Dimensions

```
Factors: (T, D, K)
  T = num_days (number of time steps)
  D = num_assets (number of stocks/securities)
  K = num_factors (number of factor features)

Returns: (T, D)
  T = same num_days
  D = same num_assets

Prices: (T, D)
  T = same num_days
  D = same num_assets
```

### Example Dimensions

For the real_factors dataset:
```
T = 11 days
D = 500 stocks
K = 10 factors

factors.npy: (11, 500, 10)
returns.npy: (11, 500)
prices.npy:  (11, 500)
```

## File Structure

Each dataset directory must contain:

```
dataset_name/
├── factors.npy      # NumPy array (T, D, K)
├── returns.npy      # NumPy array (T, D)
├── prices.npy       # NumPy array (T, D)
└── metadata.json    # Dataset information
```

## File Specifications

### factors.npy

**Shape**: `(T, D, K)`
**Data Type**: `float32`
**Description**: Factor exposures for each asset on each day

**Indexing**:
- `factors[t, d, k]` = value of factor k for asset d on day t
- t: time index (0 to T-1)
- d: asset index (0 to D-1)
- k: factor index (0 to K-1)

**Values**:
- Can be any real number
- Should be normalized/standardized for best model performance
- Missing values should be imputed (not NaN or Inf)

**Example**:
```python
import numpy as np
factors = np.load('factors.npy')
print(factors.shape)  # (11, 500, 10)

# Get factor 0 for asset 5 on day 3
value = factors[3, 5, 0]
```

### returns.npy

**Shape**: `(T, D)`
**Data Type**: `float32`
**Description**: Daily returns for each asset

**Indexing**:
- `returns[t, d]` = return of asset d on day t
- t: time index (0 to T-1)
- d: asset index (0 to D-1)

**Values**:
- Decimal format (0.01 = 1% return)
- Typical range: [-0.2, 0.2] for daily returns
- Can include extreme values during market events

**Example**:
```python
import numpy as np
returns = np.load('returns.npy')
print(returns.shape)  # (11, 500)

# Get return for asset 10 on day 5
ret = returns[5, 10]
print(f"Return: {ret*100:.2f}%")
```

### prices.npy

**Shape**: `(T, D)`
**Data Type**: `float32`
**Description**: Price levels for each asset

**Indexing**:
- `prices[t, d]` = price of asset d on day t
- t: time index (0 to T-1)
- d: asset index (0 to D-1)

**Values**:
- Positive numbers only
- Absolute price levels (can be adjusted or raw)
- Missing days should be forward-filled

**Example**:
```python
import numpy as np
prices = np.load('prices.npy')
print(prices.shape)  # (11, 500)

# Get price for asset 20 on day 7
price = prices[7, 20]
```

### metadata.json

**Format**: JSON
**Description**: Dataset information and statistics

**Required Fields**:
```json
{
  "dataset_name": "string",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "num_days": T,
  "num_assets": D,
  "num_factors": K,
  "asset_names": ["asset1", "asset2", ...],
  "factor_names": ["factor1", "factor2", ...]
}
```

**Optional Fields**:
```json
{
  "source": "description of data source",
  "created_at": "ISO timestamp",
  "return_stats": {
    "mean": float,
    "std": float,
    "min": float,
    "max": float
  },
  "factor_stats": {
    "mean": float,
    "std": float,
    "min": float,
    "max": float
  },
  "per_factor_stats": {
    "factor_name": {
      "mean": float,
      "std": float,
      "min": float,
      "max": float
    }
  }
}
```

## Time Ordering

All arrays must be in **chronological order**:
- Index 0 = earliest date
- Index T-1 = latest date

This is critical for:
- Train/validation/test splits (chronological)
- Backtesting and forward validation
- Time-series model assumptions

## Asset Ordering

Assets can be in any order, but must be **consistent** across all arrays:
- `factors[:, d, :]` must correspond to same asset as `returns[:, d]` and `prices[:, d]`
- Asset order should match `asset_names` in metadata.json

## Data Quality Requirements

### 1. No Missing Values
All arrays must not contain NaN or Inf values. Handle missing data by:
- Forward filling (for prices)
- Zero filling (for returns)
- Cross-sectional median (for factors)

### 2. Consistent Dimensions
Verify dimensions match:
```python
T, D, K = factors.shape
assert returns.shape == (T, D)
assert prices.shape == (T, D)
assert len(metadata['factor_names']) == K
assert len(metadata['asset_names']) == D
```

### 3. Data Types
All arrays should be `float32` for memory efficiency:
```python
factors = factors.astype(np.float32)
returns = returns.astype(np.float32)
prices = prices.astype(np.float32)
```

## Loading Example

```python
import numpy as np
import json
from pathlib import Path

def load_dataset(data_dir):
    """Load dataset in standard format."""
    data_dir = Path(data_dir)

    # Load arrays
    factors = np.load(data_dir / 'factors.npy')  # (T, D, K)
    returns = np.load(data_dir / 'returns.npy')  # (T, D)
    prices = np.load(data_dir / 'prices.npy')    # (T, D)

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Verify dimensions
    T, D, K = factors.shape
    assert returns.shape == (T, D), "Returns shape mismatch"
    assert prices.shape == (T, D), "Prices shape mismatch"
    assert metadata['num_days'] == T
    assert metadata['num_assets'] == D
    assert metadata['num_factors'] == K

    return factors, returns, prices, metadata

# Usage
factors, returns, prices, metadata = load_dataset('parquet_data/processed/real_factors')
print(f"Loaded {metadata['dataset_name']}")
print(f"Shape: ({factors.shape[0]} days, {factors.shape[1]} assets, {factors.shape[2]} factors)")
```

## Creating New Datasets

When creating a new dataset, follow these steps:

1. **Prepare data in panel format**
   - Ensure time-series continuity
   - Align dates across all assets
   - Fill missing values appropriately

2. **Create arrays**
   ```python
   factors = np.zeros((T, D, K), dtype=np.float32)
   returns = np.zeros((T, D), dtype=np.float32)
   prices = np.zeros((T, D), dtype=np.float32)
   ```

3. **Populate arrays**
   - Fill data in chronological order
   - Maintain asset consistency
   - Verify no NaN/Inf values

4. **Save arrays**
   ```python
   np.save('dataset_dir/factors.npy', factors)
   np.save('dataset_dir/returns.npy', returns)
   np.save('dataset_dir/prices.npy', prices)
   ```

5. **Create metadata**
   ```python
   metadata = {
       'dataset_name': 'my_dataset',
       'start_date': '2020-01-01',
       'end_date': '2020-12-31',
       'num_days': T,
       'num_assets': D,
       'num_factors': K,
       'asset_names': asset_list,
       'factor_names': factor_list
   }

   with open('dataset_dir/metadata.json', 'w') as f:
       json.dump(metadata, f, indent=2)
   ```

6. **Validate**
   ```python
   # Test loading
   f, r, p, m = load_dataset('dataset_dir')

   # Check for issues
   assert not np.any(np.isnan(f))
   assert not np.any(np.isnan(r))
   assert not np.any(np.isnan(p))
   ```
