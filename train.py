from typing import Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model.stpred import SpatioTemporalPredictor

# Training constants
PATCH_SIZE = 3  # Size of NDVI patches (must be odd)
BATCH_SIZE = 1  # Batch size for training
SEQUENCE_LENGTH = 5  # Number of timesteps to use for prediction


class NDVIDataset(Dataset):
    """Dataset for NDVI time series data"""

    def __init__(self, data: np.ndarray, sequence_length: int = SEQUENCE_LENGTH, patch_size: int = PATCH_SIZE) -> None:
        """
        Args:
            data: Array of shape (timesteps, height, width, features)
            sequence_length: Number of timesteps to use for prediction
            patch_size: Size of NDVI patches (must be odd)
        """
        self.data = torch.from_numpy(data).float()
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.pad_size = patch_size // 2

        # Add padding to height and width dimensions for patch extraction
        self.padded_data = torch.nn.functional.pad(self.data, (0, 0, self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode="replicate")

        # Calculate valid indices (need sequence_length + 1 consecutive timestamps)
        self.valid_indices = []
        for t in range(len(data) - sequence_length):
            for h in range(data.shape[1]):
                for w in range(data.shape[2]):
                    self.valid_indices.append((t, h, w))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        import ipdb

        ipdb.set_trace()
        t, h, w = self.valid_indices[idx]
        h_pad = h + self.pad_size
        w_pad = w + self.pad_size

        # Extract patches for the sequence
        patches = []
        for i in range(self.sequence_length):
            patch = self.padded_data[t + i, h_pad - self.pad_size : h_pad + self.pad_size + 1, w_pad - self.pad_size : w_pad + self.pad_size + 1, 0].flatten()
            patches.append(patch)

        sequence_patches = torch.stack(patches)  # (seq_len, patch_size*patch_size)

        # Get other features for the center pixel
        sequence = self.data[t : t + self.sequence_length, h, w]  # (seq_len, features)
        target = self.data[t + self.sequence_length, h, w, 0]  # Next NDVI value

        month = sequence[..., 1]
        year = sequence[..., 2]
        lat = sequence[..., 3]
        lon = sequence[..., 4]

        return sequence_patches, month, year, lat, lon, target


def load_data() -> np.ndarray:
    """Load the NDVI data"""
    data = np.load("data/Landsat_NDVI_time_series_1984_to_2024.npy")
    return data


def create_train_val_split(
    data: np.ndarray,
    train_years: tuple[int, int],
    val_years: tuple[int, int],
    sequence_length: int = SEQUENCE_LENGTH,
    patch_size: int = PATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation splits based on years"""

    # Find indices for train/val years
    train_mask = (data[..., 2] >= train_years[0]) & (data[..., 2] <= train_years[1])
    val_mask = (data[..., 2] >= val_years[0]) & (data[..., 2] <= val_years[1])

    train_data = data[train_mask.any(axis=(1, 2))]
    val_data = data[val_mask.any(axis=(1, 2))]

    # Create datasets
    train_dataset = NDVIDataset(train_data, sequence_length, patch_size)
    val_dataset = NDVIDataset(val_data, sequence_length, patch_size)

    # print simple ndvi value
    print(train_dataset[0][0])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def train() -> None:
    """Main training function"""
    # Load data
    data = load_data()
    print(f"Loaded data with shape: {data.shape}")

    # Create train/val split (80% train, 20% val)
    # Using years 1984-2014 for training, 2015-2024 for validation
    train_loader, val_loader = create_train_val_split(data, train_years=(1984, 2014), val_years=(2015, 2024), sequence_length=SEQUENCE_LENGTH, patch_size=PATCH_SIZE)

    # Initialize model
    model = SpatioTemporalPredictor(d_model=256, patch_size=PATCH_SIZE, ndvi_embed_dim=32, year_embed_dim=8, latlon_embed_dim=8, num_heads=8, num_layers=3, dropout=0.1)

    # TODO: Add training loop
    print("Model initialized, ready for training")


if __name__ == "__main__":
    train()
