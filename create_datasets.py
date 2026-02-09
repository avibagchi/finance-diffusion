"""
Simple script to create price time series datasets for different market segments.
Each dataset contains daily closing prices organized as (time, assets) arrays.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Dataset definitions
DATASETS = {
    'sp500': {
        'name': 'S&P 500 Large Cap',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
                   'JPM', 'V', 'UNH', 'XOM', 'WMT', 'MA', 'PG', 'JNJ', 'HD', 'COST',
                   'ABBV', 'NFLX', 'CRM', 'BAC', 'CVX', 'MRK', 'KO', 'ADBE', 'PEP', 'AMD',
                   'LIN', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'WFC', 'ORCL', 'DHR', 'PM',
                   'IBM', 'GE', 'TXN', 'INTU', 'CAT', 'CMCSA', 'QCOM', 'VZ', 'AMGN', 'NEE',
                   'AMAT', 'MS', 'RTX', 'HON', 'UNP', 'LOW', 'SPGI', 'PFE', 'T', 'UPS',
                   'BLK', 'AXP', 'NKE', 'SYK', 'BA', 'PLD', 'BSX', 'BKNG', 'DE', 'C',
                   'ADP', 'TJX', 'INTC', 'MDT', 'VRTX', 'BMY', 'AMT', 'ISRG', 'GILD', 'ADI'],
        'period': '7y'
    },
    'tech': {
        'name': 'Technology Stocks',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
                   'CRM', 'CSCO', 'AMD', 'INTC', 'QCOM', 'TXN', 'AMAT', 'ADI', 'INTU', 'NXPI',
                   'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'PANW', 'CRWD', 'ZS',
                   'DDOG', 'NET', 'SNOW', 'MDB', 'TEAM', 'NOW', 'WDAY', 'ZM', 'DOCU', 'TWLO'],
        'period': '7y'
    },
    'industrials': {
        'name': 'Industrial & Materials',
        'tickers': ['CAT', 'BA', 'GE', 'HON', 'UNP', 'RTX', 'LMT', 'DE', 'MMM', 'UPS',
                   'EMR', 'ETN', 'ITW', 'WM', 'GD', 'NOC', 'PH', 'CMI', 'FDX', 'NSC',
                   'CSX', 'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'VMC', 'MLM', 'DOW',
                   'DD', 'PPG', 'NUE', 'STLD', 'CLF', 'X', 'AA', 'ALB', 'FMC', 'EMN'],
        'period': '7y'
    },
    'small_cap': {
        'name': 'Small Cap Stocks',
        'tickers': ['IONQ', 'RGTI', 'QUBT', 'QBTS', 'RDDT', 'HIMS', 'RKLB', 'PLTR', 'HOOD', 'SOFI',
                   'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'CRSR', 'BLNK', 'CHPT', 'EVGO', 'STEM',
                   'FSLR', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'BE', 'PLUG', 'FCEL', 'BLDP', 'WKHS',
                   'NKLA', 'HYLN', 'GOEV', 'FSR', 'RIDE', 'LEV', 'ARVL', 'MULN', 'ELMS', 'AYRO'],
        'period': '5y'
    }
}

def download_and_process(dataset_name, config):
    """Download and process a single dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {config['name']}")
    print(f"{'='*60}\n")

    tickers = config['tickers']
    period = config['period']

    # Download data
    print(f"Downloading {len(tickers)} tickers for {period}...")
    data = yf.download(tickers, period=period, progress=False)['Close']

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Clean data
    print(f"Raw shape: {data.shape}")
    data = data.dropna(axis=1, how='any')  # Remove tickers with any missing data
    data = data.dropna(axis=0)  # Remove days with missing data

    print(f"Clean shape: {data.shape} ({data.shape[1]} assets, {data.shape[0]} days)")

    if data.shape[1] < 10:
        print(f"WARNING: Only {data.shape[1]} assets remaining - skipping dataset")
        return False

    # Convert to numpy
    prices = data.values.astype(np.float32)
    dates = data.index
    assets = data.columns.tolist()

    # Calculate returns
    returns = np.diff(prices, axis=0) / prices[:-1]
    returns = np.clip(returns, -0.15, 0.15)  # Winsorize at ±15%

    # Align prices and returns
    prices = prices[1:]  # Remove first day so shapes match
    dates = dates[1:]

    # Save processed data
    output_dir = Path(f'data/processed/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'prices.npy', prices)
    np.save(output_dir / 'returns.npy', returns)

    # Save metadata
    metadata = {
        'dataset_name': config['name'],
        'num_days': int(prices.shape[0]),
        'num_assets': int(prices.shape[1]),
        'start_date': str(dates[0].date()),
        'end_date': str(dates[-1].date()),
        'asset_names': assets,
        'returns_stats': {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'min': float(returns.min()),
            'max': float(returns.max())
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved to {output_dir}")
    print(f"  Prices: {prices.shape}")
    print(f"  Returns: {returns.shape}")
    print(f"  Return stats: mean={returns.mean():.4f}, std={returns.std():.4f}")

    return True

def main():
    print("Creating financial datasets...")
    print(f"Total datasets to create: {len(DATASETS)}\n")

    successful = []
    failed = []

    for dataset_name, config in DATASETS.items():
        try:
            success = download_and_process(dataset_name, config)
            if success:
                successful.append(dataset_name)
            else:
                failed.append(dataset_name)
        except Exception as e:
            print(f"✗ ERROR processing {dataset_name}: {e}")
            failed.append(dataset_name)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(successful)} - {successful}")
    if failed:
        print(f"✗ Failed: {len(failed)} - {failed}")
    print(f"\nData saved to data/processed/")
    print("="*60)

if __name__ == '__main__':
    main()
