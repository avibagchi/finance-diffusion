# CSV Data Files

This directory contains CSV exports of the real_factors dataset.

## Files

### Main Data Files

1. **returns.csv** (11 × 500)
   - Daily returns for each asset
   - Index: date
   - Columns: asset names
   - Values: decimal returns (0.01 = 1%)

2. **prices.csv** (11 × 500)
   - Daily prices for each asset
   - Index: date
   - Columns: asset names
   - Values: price levels

3. **factors_long.csv** (5500 × 12)
   - All factors in long/stacked format
   - Columns: date, asset, ret_1_0, ret_3_1, ret_6_1, ret_12_1, be_me, at_me, sale_me, ni_me, ocf_me, ebitda_mev
   - More compact than wide format

### Factor Files (Wide Format)

Individual CSV files for each factor (11 × 500):
- **factor_ret_1_0.csv** - ret_1_0 values
- **factor_ret_3_1.csv** - ret_3_1 values
- **factor_ret_6_1.csv** - ret_6_1 values
- **factor_ret_12_1.csv** - ret_12_1 values
- **factor_be_me.csv** - be_me values
- **factor_at_me.csv** - at_me values
- **factor_sale_me.csv** - sale_me values
- **factor_ni_me.csv** - ni_me values
- **factor_ocf_me.csv** - ocf_me values
- **factor_ebitda_mev.csv** - ebitda_mev values

### Summary Statistics

4. **summary_stats.csv**
   - Mean, std, min, max, median for all variables
   - Useful for quick data inspection

## Data Period

- Start: 2010-01-04
- End: 2010-01-15
- Days: 11
- Assets: 500
- Factors: 10

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
