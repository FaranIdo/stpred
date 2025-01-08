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
from config import *  # Import all constants from config.py

class NDVIDataset(Dataset):
    """Dataset for NDVI time series data with on-demand patch computation and random patch size masking"""

    def __init__(self, data: np.ndarray, sequence_length: int = TRAIN_SEQUENCE_LENGTH, patch_size: int = PATCH_SIZE, sequence_masking: bool = False) -> None:
        """
        Args:
            data: Array of shape (timesteps, height, width, features)
            sequence_length: Maximum number of timesteps to use for prediction
            patch_size: Maximum size of NDVI patches (must be odd)
            sequence_masking: Whether to use variable sequence lengths between batches
        """
        logging.info(f"Initializing NDVIDataset with data shape: {data.shape}")

        # Store data as numpy array
        self.data = data
        self.max_sequence_length = sequence_length  # Renamed to indicate it's the maximum
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.sequence_masking = sequence_masking
        self.current_sequence_length = sequence_length  # Add current sequence length
        # Pre-compute possible patch sizes (odd numbers from 3 to patch_size)
        self.possible_patch_sizes = np.arange(3, patch_size + 1, 2)

        # Pre-compute padding configurations for each possible patch size
        self.padding_configs = {}
        for size in self.possible_patch_sizes:
            pad_amount = (patch_size - size) // 2
            # Store pre-computed indices for efficient padding
            idx = np.zeros(patch_size * patch_size, dtype=np.int32)
            src_idx = np.arange(size * size)
            start_idx = pad_amount * patch_size + pad_amount
            for i in range(size):
                idx[start_idx + i * patch_size : start_idx + i * patch_size + size] = src_idx[i * size : (i + 1) * size]
            self.padding_configs[size] = idx

        # Calculate valid indices directly
        logging.info("Calculating valid indices...")
        # Vectorized valid indices calculation
        t_range = np.arange(len(data) - self.max_sequence_length)
        h_range = np.arange(self.pad_size, data.shape[1] - self.pad_size)
        w_range = np.arange(self.pad_size, data.shape[2] - self.pad_size)

        t, h, w = np.meshgrid(t_range, h_range, w_range, indexing="ij")
        valid_indices = np.stack([t.ravel(), h.ravel(), w.ravel()], axis=1)

        # Sample indices
        num_samples = int(len(valid_indices) * DATA_SAMPLE_PERCENTAGE)
        indices = np.random.choice(len(valid_indices), size=num_samples, replace=False)
        self.valid_indices = valid_indices[indices]

        logging.info(f"Dataset initialized with {len(self)} samples")

    def set_sequence_length(self) -> None:
        """Update the current sequence length if sequence masking is enabled"""
        if self.sequence_masking:
            self.current_sequence_length = np.random.randint(1, self.max_sequence_length + 1)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_random_patch_size(self) -> int:
        """Generate random odd patch size less than or equal to PATCH_SIZE"""
        return np.random.choice(self.possible_patch_sizes)

    def _extract_and_pad_patch(self, t: int, h: int, w: int, random_patch_size: int) -> np.ndarray:
        """Efficiently extract and pad patch to patch_size*patch_size"""
        random_pad_size = random_patch_size // 2

        # Extract patch
        patch = self.data[t, h - random_pad_size : h + random_pad_size + 1, w - random_pad_size : w + random_pad_size + 1, 0]

        # Calculate padding needed on each side
        pad_amount = (self.patch_size - random_patch_size) // 2

        # Pad the patch to match patch_size
        padded = np.pad(patch, pad_amount, mode="constant", constant_values=0)
        return padded.ravel()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Compute patches and metadata on-demand with random patch size"""
        t, h, w = self.valid_indices[idx]

        # Use current_sequence_length instead of generating random length per sample
        sequence_length = self.current_sequence_length
        random_patch_size = self._get_random_patch_size()

        # Pre-allocate array for sequence with actual sequence length
        padded_patches = np.empty((sequence_length, self.patch_size * self.patch_size), dtype=np.float32)

        # Extract and pad patches for the sequence
        for i in range(sequence_length):
            padded_patches[i] = self._extract_and_pad_patch(t + i, h, w, random_patch_size)

        # Extract metadata for the sequence
        sequence = self.data[t : t + sequence_length, h, w]  # (seq_len, features)

        # Create all tensors at once
        return {
            "ndvi": torch.from_numpy(padded_patches),
            "month": torch.from_numpy(sequence[..., 1].astype(np.float32)),
            "year": torch.from_numpy(sequence[..., 2].astype(np.float32)),
            "lat": torch.from_numpy(sequence[..., 3].astype(np.float32)),
            "lon": torch.from_numpy(sequence[..., 4].astype(np.float32)),
            "target": torch.tensor(self.data[t + sequence_length, h, w, 0], dtype=torch.float32),
            "patch_size": torch.tensor(random_patch_size, dtype=torch.int32),
            "sequence_length": torch.tensor(sequence_length, dtype=torch.int32),
        }


def load_data() -> np.ndarray:
    """Load the NDVI data"""
    data = np.load("data/Landsat_NDVI_time_series_1984_to_2024.npy")
    return data


def create_train_val_split(
    data: np.ndarray,
    train_years: tuple[int, int] = TRAIN_YEARS,
    val_years: tuple[int, int] = VAL_YEARS,
    train_sequence_length: int = TRAIN_SEQUENCE_LENGTH,
    val_sequence_length: int = VAL_SEQUENCE_LENGTH,
    patch_size: int = PATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation splits based on years"""

    # Find indices for train/val years
    train_mask = (data[..., 2] >= train_years[0]) & (data[..., 2] <= train_years[1])
    val_mask = (data[..., 2] >= val_years[0]) & (data[..., 2] <= val_years[1])

    train_data = data[train_mask.any(axis=(1, 2))]
    val_data = data[val_mask.any(axis=(1, 2))]

    # Create datasets with sequence masking enabled
    train_dataset = NDVIDataset(train_data, train_sequence_length, patch_size, sequence_masking=True)
    val_dataset = NDVIDataset(val_data, val_sequence_length, patch_size, sequence_masking=True)

    # Custom batch sampler that updates sequence length before each batch
    class SequenceLengthBatchSampler:
        def __init__(self, dataset: NDVIDataset, batch_size: int, shuffle: bool):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_samples = len(dataset)

        def __iter__(self):
            if self.shuffle:
                indices = torch.randperm(self.num_samples).tolist()
            else:
                indices = list(range(self.num_samples))

            for i in range(0, len(indices), self.batch_size):
                # Update sequence length before each batch if sequence masking is enabled
                if self.dataset.sequence_masking:
                    self.dataset.set_sequence_length()
                yield indices[i : i + self.batch_size]

        def __len__(self):
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    # Create dataloaders with custom batch sampler
    train_batch_sampler = SequenceLengthBatchSampler(train_dataset, BATCH_SIZE, shuffle=True)
    val_batch_sampler = SequenceLengthBatchSampler(val_dataset, BATCH_SIZE, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=WORKERS, persistent_workers=True, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=WORKERS, persistent_workers=True, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)

    return train_loader, val_loader


def setup_model(device: torch.device) -> tuple[SpatioTemporalPredictor, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, torch.nn.Module]:
    """Initialize model, optimizer, scheduler and loss function"""
    model = SpatioTemporalPredictor(
        d_model=MODEL_D_MODEL,
        patch_size=PATCH_SIZE,
        ndvi_embed_dim=MODEL_NDVI_EMBED_DIM,
        year_embed_dim=MODEL_YEAR_EMBED_DIM,
        latlon_embed_dim=MODEL_LATLON_EMBED_DIM,
        num_heads=MODEL_NUM_HEADS,
        num_layers=MODEL_NUM_LAYERS,
        d_ff=MODEL_D_FF,
        dropout=MODEL_DROPOUT,
        max_seq_len=MODEL_MAX_SEQ_LEN,
        first_year=MODEL_FIRST_YEAR,
    ).to(device)

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
    scaler = torch.cuda.amp.GradScaler()

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

    logging.info(f"Starting validation with total batches: {total_batches}")
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
    train_loader, val_loader = create_train_val_split(data)

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
