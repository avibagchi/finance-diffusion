"""
Data loader for real financial datasets.

Loads preprocessed factor and return data for training the Factor-Based
Conditional Diffusion Model.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class RealFinancialDataset(Dataset):
    """
    PyTorch Dataset for real financial data.

    Loads preprocessed factors (T, D, K) and returns (T, D) from disk.
    Supports train/val/test splits with chronological ordering.
    """

    def __init__(self, data_dir, split='train', train_ratio=0.8, val_ratio=0.0):
        """
        Initialize dataset.

        Args:
            data_dir (str): Path to processed data directory
            split (str): One of 'train', 'val', 'test'
            train_ratio (float): Fraction of data for training
            val_ratio (float): Fraction of data for validation
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Load data
        self.factors = np.load(self.data_dir / 'factors.npy')  # (T, D, K)
        self.returns = np.load(self.data_dir / 'returns.npy')  # (T, D)

        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Apply chronological split
        T = len(self.returns)
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))

        if split == 'train':
            self.factors = self.factors[:train_end]
            self.returns = self.returns[:train_end]
        elif split == 'val':
            if val_ratio > 0:
                self.factors = self.factors[train_end:val_end]
                self.returns = self.returns[train_end:val_end]
            else:
                raise ValueError("Validation split requested but val_ratio=0")
        elif split == 'test':
            self.factors = self.factors[val_end:]
            self.returns = self.returns[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Convert to PyTorch tensors
        self.factors = torch.from_numpy(self.factors).float()
        self.returns = torch.from_numpy(self.returns).float()

        print(f"Loaded {split} dataset:")
        print(f"  Factors shape: {self.factors.shape}")
        print(f"  Returns shape: {self.returns.shape}")

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx):
        """
        Get a single data point.

        Returns:
            factors: (D, K) tensor of factor exposures
            returns: (D,) tensor of returns
        """
        return self.factors[idx], self.returns[idx]

    def get_num_assets(self):
        """Get number of assets (D)."""
        return self.factors.shape[1]

    def get_num_factors(self):
        """Get number of factors (K)."""
        return self.factors.shape[2]

    def get_asset_names(self):
        """Get list of asset names."""
        return self.metadata.get('asset_names', [])

    def get_factor_names(self):
        """Get list of factor names."""
        return self.metadata.get('factor_names', [])

    def get_date_range(self):
        """Get date range of dataset."""
        return {
            'start_date': self.metadata.get('start_date'),
            'end_date': self.metadata.get('end_date'),
            'num_days': self.metadata.get('num_days')
        }


def load_real_dataset(data_dir, split='train', batch_size=32, shuffle=None):
    """
    Convenience function to load dataset and create DataLoader.

    Args:
        data_dir (str): Path to processed data directory
        split (str): One of 'train', 'val', 'test'
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle data (default: True for train, False otherwise)

    Returns:
        dataset: RealFinancialDataset instance
        dataloader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader

    # Load dataset
    dataset = RealFinancialDataset(data_dir, split=split)

    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )

    return dataset, dataloader


def get_dataset_info(data_dir):
    """
    Get information about a dataset without loading the full data.

    Args:
        data_dir (str): Path to processed data directory

    Returns:
        dict: Dataset information from metadata.json
    """
    data_dir = Path(data_dir)

    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load statistics if available
    stats_file = data_dir / 'statistics.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            statistics = json.load(f)
        metadata['statistics'] = statistics

    return metadata


# Example usage and testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test data loader')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print(f"\nLoading dataset from: {args.data_dir}")
    print(f"Split: {args.split}")

    # Get dataset info
    print("\n" + "="*60)
    print("DATASET INFO")
    print("="*60)
    info = get_dataset_info(args.data_dir)
    print(f"Dataset: {info['dataset_name']}")
    print(f"Period: {info['start_date']} to {info['end_date']}")
    print(f"Shape: {info['num_days']} days × {info['num_assets']} assets × {info['num_factors']} factors")
    print(f"Assets: {info['asset_names'][:5]}..." if len(info['asset_names']) > 5 else f"Assets: {info['asset_names']}")
    print(f"Factors: {', '.join(info['factor_names'])}")

    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    dataset, dataloader = load_real_dataset(args.data_dir, split=args.split, batch_size=args.batch_size)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    # Test iteration
    print("\n" + "="*60)
    print("TESTING ITERATION")
    print("="*60)
    for i, (factors, returns) in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Factors shape: {factors.shape}")
        print(f"  Returns shape: {returns.shape}")
        print(f"  Factors range: [{factors.min():.3f}, {factors.max():.3f}]")
        print(f"  Returns range: [{returns.min():.3f}, {returns.max():.3f}]")

        if i >= 2:  # Only show first 3 batches
            break

    print("\n✅ Data loader test successful!")
