"""
List all available datasets with their properties.
"""

from pathlib import Path
import json


def list_datasets():
    """List all available datasets."""
    processed_dir = Path('data/processed')

    if not processed_dir.exists():
        print("ERROR: data/processed directory does not exist")
        return

    datasets = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])

    if not datasets:
        print("No datasets found in data/processed/")
        return

    print("="*80)
    print(f"AVAILABLE DATASETS ({len(datasets)} total)")
    print("="*80)

    for dataset_name in datasets:
        data_dir = processed_dir / dataset_name
        metadata_file = data_dir / 'metadata.json'

        if not metadata_file.exists():
            print(f"\n{dataset_name}: [No metadata file]")
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"\n{'─'*80}")
        print(f"📊 {dataset_name.upper()}")
        print(f"{'─'*80}")
        print(f"Name:           {metadata.get('dataset_name', 'N/A')}")
        print(f"Period:         {metadata.get('start_date', 'N/A')} → {metadata.get('end_date', 'N/A')}")
        print(f"Trading days:   {metadata.get('num_days', 'N/A')}")
        print(f"Assets:         {metadata.get('num_assets', 'N/A')}")
        print(f"Factors:        {metadata.get('num_factors', 'N/A')}")

        # Asset names (show first 10)
        asset_names = metadata.get('asset_names', [])
        if asset_names:
            if len(asset_names) <= 10:
                print(f"Tickers:        {', '.join(asset_names)}")
            else:
                print(f"Tickers:        {', '.join(asset_names[:10])}... (+{len(asset_names)-10} more)")

        # Factor names
        factor_names = metadata.get('factor_names', [])
        if factor_names:
            print(f"Factor names:   {', '.join(factor_names)}")

        # Returns statistics
        returns_stats = metadata.get('returns_stats', {})
        if returns_stats:
            print(f"\nReturns stats:")
            print(f"  Mean:   {returns_stats.get('mean', 0):.6f}")
            print(f"  Std:    {returns_stats.get('std', 0):.6f}")
            print(f"  Range:  [{returns_stats.get('min', 0):.4f}, {returns_stats.get('max', 0):.4f}]")

        # Factor statistics
        factor_stats = metadata.get('factor_stats', {})
        if factor_stats:
            print(f"\nFactor stats:")
            print(f"  Mean:   {factor_stats.get('mean', 0):.6f}")
            print(f"  Std:    {factor_stats.get('std', 0):.6f}")
            print(f"  Range:  [{factor_stats.get('min', 0):.4f}, {factor_stats.get('max', 0):.4f}]")

        # Check if all required files exist
        files_status = []
        required_files = ['factors.npy', 'returns.npy', 'prices.npy', 'metadata.json']
        for file in required_files:
            if (data_dir / file).exists():
                files_status.append(f"✓ {file}")
            else:
                files_status.append(f"✗ {file}")

        print(f"\nFiles:          {' '.join(files_status)}")

    print(f"\n{'='*80}")
    print("\nUsage:")
    print("  python src/data_loader.py --data_dir data/processed/sp500 --split train")
    print("="*80)


if __name__ == '__main__':
    list_datasets()
