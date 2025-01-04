from typing import Tuple, Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model.stpred import SpatioTemporalPredictor
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm

# Training constants
PATCH_SIZE = 3  # Size of NDVI patches (must be odd)
BATCH_SIZE = 32  # Batch size for training
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

        # Calculate valid indices (need sequence_length + 1 consecutive timestamps)
        # Only include pixels that have enough surrounding pixels for the patch
        self.valid_indices = []
        for t in range(len(data) - sequence_length):
            for h in range(self.pad_size, data.shape[1] - self.pad_size):
                for w in range(self.pad_size, data.shape[2] - self.pad_size):
                    self.valid_indices.append((t, h, w))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t, h, w = self.valid_indices[idx]

        # Extract patches for the sequence
        patches = []
        for i in range(self.sequence_length):
            patch = self.data[t + i, h - self.pad_size : h + self.pad_size + 1, w - self.pad_size : w + self.pad_size + 1, 0].flatten()
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

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def setup_model(device: torch.device) -> Tuple[SpatioTemporalPredictor, torch.optim.Optimizer, torch.nn.Module]:
    """Initialize model, optimizer and loss function"""
    model = SpatioTemporalPredictor(d_model=256, patch_size=PATCH_SIZE, ndvi_embed_dim=32, year_embed_dim=8, latlon_embed_dim=8, num_heads=8, num_layers=3, dropout=0.1).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = MSELoss()

    return model, optimizer, criterion


def train_epoch(model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, epoch: int, num_epochs: int) -> float:
    """Train for one epoch and return average loss"""
    model.train()
    train_loss = 0.0
    train_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar:
        # Unpack batch and move to device
        ndvi_patches, month, year, lat, lon, target = [x.to(device) for x in batch]

        # Forward pass
        predictions, _ = model(ndvi_patches, month, year, lat, lon)

        # predictions should in in shape (batch_size), last dim is unneeded
        predictions = predictions.squeeze(-1)

        loss = criterion(predictions, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        train_loss += loss.item()
        train_batches += 1
        progress_bar.set_postfix({"train_loss": f"{train_loss/train_batches:.6f}"})

    return train_loss / train_batches


def validate(model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> float:
    """Run validation and return average loss"""
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Unpack batch and move to device
            ndvi_patches, month, year, lat, lon, target = [x.to(device) for x in batch]

            # Forward pass
            predictions, _ = model(ndvi_patches, month, year, lat, lon)
            loss = criterion(predictions.squeeze(), target)

            # Update metrics
            val_loss += loss.item()
            val_batches += 1

    return val_loss / val_batches


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, train_loss: float, val_loss: float, filename: str = "best_model.pt") -> None:
    """Save model checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        filename,
    )
    print("Saved new best model!")


def train() -> None:
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = load_data()
    print(f"Loaded data with shape: {data.shape}")

    # Create train/val split
    train_loader, val_loader = create_train_val_split(data, train_years=(1984, 2014), val_years=(2015, 2024), sequence_length=SEQUENCE_LENGTH, patch_size=PATCH_SIZE)

    # Setup model, optimizer and criterion
    model, optimizer, criterion = setup_model(device)

    # Training parameters
    num_epochs = 100
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)

        # Validation phase
        avg_val_loss = validate(model, val_loader, criterion, device)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_val_loss)

        print("-" * 50)


if __name__ == "__main__":
    train()
