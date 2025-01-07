import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.stpred import SpatioTemporalPredictor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from datetime import datetime
import gc
import time
from tqdm import tqdm  # Add tqdm import

# Training constants
PATCH_SIZE = 3  # Size of NDVI patches (must be odd)
BATCH_SIZE = 4096  # Increased batch size for better GPU utilization
SEQUENCE_LENGTH = 10  # Number of timesteps to use for prediction
EPOCHS = 30
WARMUP_EPOCHS = 5  # Number of epochs to keep learning rate constant
DATA_SAMPLE_PERCENTAGE = 0.1  # Percentage of data to use for training and validation
LEARNING_RATE = 2e-3  # Increased learning rate to compensate for larger batch size
DECAY_GAMMA = 0.95  # Decay rate for learning rate scheduler
WORKERS = 8  # Reduced number of workers to prevent CPU bottleneck
PREFETCH_FACTOR = 2  # Reduced prefetch factor to prevent memory issues
LOG_INTERVAL = 10  # Reduced logging interval for better progress tracking
GRADIENT_ACCUMULATION_STEPS = 1  # Removed gradient accumulation for direct updates

class NDVIDataset(Dataset):
    """Dataset for NDVI time series data with on-demand patch computation"""

    def __init__(self, data: np.ndarray, sequence_length: int = SEQUENCE_LENGTH, patch_size: int = PATCH_SIZE) -> None:
        """
        Args:
            data: Array of shape (timesteps, height, width, features)
            sequence_length: Number of timesteps to use for prediction
            patch_size: Size of NDVI patches (must be odd)
        """
        logging.info(f"Initializing NDVIDataset with data shape: {data.shape}")

        # Store data as numpy array
        self.data = data
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.pad_size = patch_size // 2

        # Calculate valid indices directly
        logging.info("Calculating valid indices...")
        valid_indices = []
        for t in range(len(data) - sequence_length):
            for h in range(self.pad_size, data.shape[1] - self.pad_size):
                for w in range(self.pad_size, data.shape[2] - self.pad_size):
                    valid_indices.append((t, h, w))

        # Convert to numpy array and sample
        self.valid_indices = np.array(valid_indices)
        num_samples = int(len(self.valid_indices) * DATA_SAMPLE_PERCENTAGE)
        indices = np.random.choice(len(self.valid_indices), size=num_samples, replace=False)
        self.valid_indices = self.valid_indices[indices]

        logging.info(f"Dataset initialized with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Compute patches and metadata on-demand"""
        t, h, w = self.valid_indices[idx]

        # Extract sequence patches
        sequence_patches = np.empty((self.sequence_length, self.patch_size * self.patch_size), dtype=np.float32)
        for i in range(self.sequence_length):
            sequence_patches[i] = self.data[t + i, h - self.pad_size : h + self.pad_size + 1, w - self.pad_size : w + self.pad_size + 1, 0].flatten()

        # Convert to tensor efficiently
        patches = torch.from_numpy(sequence_patches)

        # Extract metadata for the sequence
        sequence = self.data[t : t + self.sequence_length, h, w]  # (seq_len, features)
        target = self.data[t + self.sequence_length, h, w, 0]  # Next NDVI value

        return {
            "ndvi": patches,
            "month": torch.tensor(sequence[..., 1], dtype=torch.float32),
            "year": torch.tensor(sequence[..., 2], dtype=torch.float32),
            "lat": torch.tensor(sequence[..., 3], dtype=torch.float32),
            "lon": torch.tensor(sequence[..., 4], dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


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
) -> tuple[DataLoader, DataLoader]:
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, persistent_workers=True, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, persistent_workers=True, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)

    return train_loader, val_loader


def setup_model(device: torch.device) -> tuple[SpatioTemporalPredictor, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, torch.nn.Module]:
    """Initialize model, optimizer, scheduler and loss function"""
    model = SpatioTemporalPredictor(d_model=256, patch_size=PATCH_SIZE, ndvi_embed_dim=32, year_embed_dim=8, latlon_embed_dim=8, num_heads=8, num_layers=3, dropout=0.1).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=DECAY_GAMMA)
    criterion = MSELoss()

    return model, optimizer, scheduler, criterion


def train_epoch(
    model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, epoch: int, num_epochs: int, writer: SummaryWriter
) -> tuple[float, float]:
    """Train for one epoch and return average loss and MAE"""
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    train_batches = 0
    mae_criterion = L1Loss()
    total_batches = len(train_loader)
    start_time = time.time()
    batch_start_time = time.time()

    logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
    optimizer.zero_grad(set_to_none=True)

    # Enable automatic mixed precision for faster training
    scaler = torch.amp.GradScaler()

    # Enable automatic memory management
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i, batch in enumerate(train_loader):
        try:
            # move batch to device efficiently in advance
            batch = {k: v.to("cuda") for k, v in batch.items()}

            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                predictions, _ = model(batch["ndvi"], batch["month"], batch["year"], batch["lat"], batch["lon"])
                predictions = predictions.squeeze(-1)
                loss = criterion(predictions, batch["target"]) / GRADIENT_ACCUMULATION_STEPS
                mae = mae_criterion(predictions, batch["target"])

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Update metrics
            train_loss += float(loss.item() * GRADIENT_ACCUMULATION_STEPS)
            train_mae += float(mae.item())
            train_batches += 1

            # Only optimize after accumulation steps
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Log progress periodically
            if (i + 1) % LOG_INTERVAL == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                batch_time = current_time - batch_start_time
                samples_per_second = (BATCH_SIZE * LOG_INTERVAL) / batch_time
                current_loss = train_loss / train_batches
                current_mae = train_mae / train_batches
                progress = (i + 1) / total_batches * 100
                logging.info(
                    f"Epoch {epoch+1}/{num_epochs} - Batch [{i+1}/{total_batches}] ({progress:.1f}%) - "
                    f"Loss: {current_loss:.6f}, MAE: {current_mae:.6f} - "
                    f"Speed: {samples_per_second:.1f} samples/sec"
                )
                batch_start_time = current_time

        finally:
            # Explicit cleanup
            del loss, mae, predictions
            for k in list(batch.keys()):
                del batch[k]
            del batch

            if (i + 1) % (GRADIENT_ACCUMULATION_STEPS * 2) == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    avg_train_loss = train_loss / train_batches
    avg_train_mae = train_mae / train_batches

    # Log to TensorBoard
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("MAE/train", avg_train_mae, epoch)

    return avg_train_loss, avg_train_mae


def validate(model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module, device: torch.device, epoch: int, writer: SummaryWriter) -> tuple[float, float]:
    """Run validation and return average loss and MAE"""
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_batches = 0
    mae_criterion = L1Loss()
    total_batches = len(val_loader)

    logging.info("Starting validation...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            predictions, _ = model(batch["ndvi"], batch["month"], batch["year"], batch["lat"], batch["lon"])
            predictions = predictions.squeeze(-1)

            loss = criterion(predictions, batch["target"])
            mae = mae_criterion(predictions, batch["target"])

            # Update metrics
            val_loss += loss.item()
            val_mae += mae.item()
            val_batches += 1

            # Log progress periodically
            if (i + 1) % LOG_INTERVAL == 0:
                progress = (i + 1) / total_batches * 100
                current_loss = val_loss / val_batches
                current_mae = val_mae / val_batches
                logging.info(f"Validation - {progress:.1f}% - Loss: {current_loss:.6f}, MAE: {current_mae:.6f}")

            # Cleanup
            del loss, mae, predictions, batch

    avg_val_loss = val_loss / val_batches
    avg_val_mae = val_mae / val_batches

    # Log to TensorBoard
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("MAE/val", avg_val_mae, epoch)

    return avg_val_loss, avg_val_mae


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, train_loss: float, val_loss: float, run_dir: str) -> None:
    """Save model checkpoint in the run directory"""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Generate filename with just epoch and validation loss
    filename = f"model_epoch{epoch:03d}_valloss{val_loss:.6f}.pt"
    filepath = os.path.join(models_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        filepath,
    )
    print(f"Saved checkpoint: {filepath}")


def setup_logging(run_dir: str) -> None:
    """Setup logging to both file and console"""
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Setup file handler
    log_file = os.path.join(run_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def train() -> None:
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create run directory with timestamp under checkpoints
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    run_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    setup_logging(run_dir)

    logging.info(f"Using device: {device}")

    # Initialize TensorBoard writer in the run directory
    writer = SummaryWriter(log_dir=run_dir)

    # Load data
    data = load_data()
    logging.info(f"Loaded data with shape: {data.shape}")

    # Create train/val split
    train_loader, val_loader = create_train_val_split(data, train_years=(1984, 2014), val_years=(2015, 2024), sequence_length=SEQUENCE_LENGTH, patch_size=PATCH_SIZE)

    # Setup model, optimizer and criterion
    model, optimizer, scheduler, criterion = setup_model(device)

    # Training parameters
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        avg_train_loss, avg_train_mae = train_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS, writer)

        # Validation phase
        avg_val_loss, avg_val_mae = validate(model, val_loader, criterion, device, epoch, writer)

        # Step the scheduler after warm-up period
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        # Log epoch summary
        logging.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        logging.info(f"Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f}")
        logging.info(f"Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}")
        logging.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        if epoch < WARMUP_EPOCHS:
            logging.info("(Warm-up period - Learning rate constant)")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_val_loss, run_dir)

        logging.info("-" * 50)

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train()
