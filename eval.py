import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import logging
import argparse
from sklearn.metrics import f1_score, r2_score
from tqdm import tqdm
import random
from train import NDVIDataset, load_data, setup_model
from config import *
import pandas as pd


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint"""
    # First create a new model instance
    model, _, _, _ = setup_model(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.6f}")
    return model


def calculate_f1_scores(predictions: np.ndarray, targets: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    """Calculate F1 scores at different NDVI thresholds"""
    f1_scores = {}

    for threshold in thresholds:
        # Convert continuous NDVI values to binary classes based on threshold
        pred_binary = (predictions >= threshold).astype(int)
        target_binary = (targets >= threshold).astype(int)

        # Calculate F1 score
        f1 = f1_score(target_binary, pred_binary)
        f1_scores[f"f1_score_{threshold}"] = f1

    return f1_scores


def calculate_r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate R² score using PyTorch tensors"""
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def calculate_and_log_statistics(predictions: np.ndarray, targets: np.ndarray, num_samples: int = 10) -> None:
    """Calculate and log comprehensive statistics for predictions and targets.

    Args:
        predictions: Array of model predictions
        targets: Array of ground truth values
        num_samples: Number of random samples to display (default: 10)
    """
    # Ground Truth Statistics
    logging.info("\nGround Truth Statistics:")
    logging.info(f"Min: {np.min(targets):.4f}")
    logging.info(f"Max: {np.max(targets):.4f}")
    logging.info(f"Mean: {np.mean(targets):.4f}")
    logging.info(f"Median: {np.median(targets):.4f}")
    logging.info(f"Std: {np.std(targets):.4f}")
    logging.info(f"25th Percentile: {np.percentile(targets, 25):.4f}")
    logging.info(f"75th Percentile: {np.percentile(targets, 75):.4f}")

    # Prediction Statistics
    logging.info("\nPrediction Statistics:")
    logging.info(f"Min: {np.min(predictions):.4f}")
    logging.info(f"Max: {np.max(predictions):.4f}")
    logging.info(f"Mean: {np.mean(predictions):.4f}")
    logging.info(f"Median: {np.median(predictions):.4f}")
    logging.info(f"Std: {np.std(predictions):.4f}")
    logging.info(f"25th Percentile: {np.percentile(predictions, 25):.4f}")
    logging.info(f"75th Percentile: {np.percentile(predictions, 75):.4f}")

    # Error Statistics
    errors = np.abs(predictions - targets)  # Using absolute differences
    logging.info("\nError Statistics (Absolute Differences):")
    logging.info(f"Min Error: {np.min(errors):.4f}")
    logging.info(f"Max Error: {np.max(errors):.4f}")
    logging.info(f"Mean Error: {np.mean(errors):.4f}")  # This is now MAE
    logging.info(f"Median Error: {np.median(errors):.4f}")
    logging.info(f"Std Error: {np.std(errors):.4f}")
    logging.info(f"25th Percentile Error: {np.percentile(errors, 25):.4f}")
    logging.info(f"75th Percentile Error: {np.percentile(errors, 75):.4f}")

    # Also show raw error statistics for bias analysis
    raw_errors = predictions - targets
    logging.info("\nRaw Error Statistics (for bias analysis):")
    logging.info(f"Mean Raw Error: {np.mean(raw_errors):.4f}")  # Bias
    logging.info(f"Std Raw Error: {np.std(raw_errors):.4f}")

    # Display random samples
    logging.info("\nRandom Sample Comparisons:")
    logging.info("Index\tGround Truth\tPrediction\tAbsolute Error")
    logging.info("-" * 50)

    sample_indices = random.sample(range(len(predictions)), num_samples)
    for idx in sample_indices:
        ground_truth = targets[idx]
        prediction = predictions[idx]
        abs_error = abs(ground_truth - prediction)
        logging.info(f"{idx:5d}\t{ground_truth:.4f}\t{prediction:.4f}\t{abs_error:.4f}")


def evaluate(checkpoint_path: str, subset_size: float = 0.1) -> None:
    """Evaluate model on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    data = load_data()
    logging.info(f"Loaded data with shape: {data.shape}")

    # Create test dataset using a subset of the validation years
    test_mask = (data[..., 2] >= VAL_YEARS[0]) & (data[..., 2] <= VAL_YEARS[1])
    test_data = data[test_mask.any(axis=(1, 2))]

    # Create test dataset with fixed sequence length
    test_dataset = NDVIDataset(test_data, sequence_length=VAL_SEQUENCE_LENGTH, patch_size=PATCH_SIZE, sequence_masking=False)

    # Calculate subset size and convert numpy array to list for indices
    num_samples = int(len(test_dataset) * subset_size)
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False).tolist()

    # Create subset dataset
    test_loader = DataLoader(torch.utils.data.Subset(test_dataset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    # Load the full model from checkpoint
    model = load_model(checkpoint_path, device)
    model.eval()

    # Evaluation metrics
    mse_loss = 0.0
    mae_loss = 0.0
    all_predictions = []
    all_targets = []

    # Thresholds for F1 score calculation
    ndvi_thresholds = [0.2, 0.4, 0.6, 0.8]  # Common NDVI thresholds

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            predictions, _ = model(batch["ndvi"], batch["month"], batch["year"], batch["lat"], batch["lon"])
            predictions = predictions.squeeze(-1)

            # Calculate losses
            mse = torch.nn.functional.mse_loss(predictions, batch["target"])
            mae = torch.nn.functional.l1_loss(predictions, batch["target"])

            mse_loss += mse.item()
            mae_loss += mae.item()

            # Store predictions and targets for F1 score calculation
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch["target"].cpu().numpy())

            # Cleanup
            del predictions, mse, mae, batch

    # Calculate average losses
    avg_mse = mse_loss / len(test_loader)
    avg_mae = mae_loss / len(test_loader)

    # Convert lists to numpy arrays and calculate R² score
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    r2 = r2_score(all_targets, all_predictions)

    # Calculate F1 scores at different thresholds
    f1_scores = calculate_f1_scores(all_predictions, all_targets, ndvi_thresholds)

    # Log results
    logging.info(f"\nEvaluation Results:")
    logging.info(f"Average MSE: {avg_mse:.6f}")
    logging.info(f"Average MAE: {avg_mae:.6f}")
    logging.info(f"R² Score: {r2:.6f}")
    for threshold, f1 in f1_scores.items():
        logging.info(f"{threshold}: {f1:.4f}")

    # Calculate and log comprehensive statistics
    calculate_and_log_statistics(all_predictions, all_targets)


def evaluate_configuration(model: torch.nn.Module, data: np.ndarray, patch_size: int, sequence_length: int, device: torch.device, subset_size: float = 0.1) -> tuple[float, float, float, float]:
    """
    Evaluate model for a specific patch size and sequence length configuration.
    Returns RME, L1 (MAE), L2 (MSE), and F1 score.
    """
    logging.info(f"Evaluating configuration: Patch {patch_size}x{patch_size}, Sequence Length {sequence_length}")

    # Create test dataset with specific configuration
    test_mask = (data[..., 2] >= VAL_YEARS[0]) & (data[..., 2] <= VAL_YEARS[1])
    test_data = data[test_mask.any(axis=(1, 2))]

    test_dataset = NDVIDataset(test_data, sequence_length=sequence_length, patch_size=PATCH_SIZE, sequence_masking=False, specific_patch_size=patch_size)

    # Calculate subset size and create dataloader
    num_samples = int(len(test_dataset) * subset_size)
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False).tolist()
    test_loader = DataLoader(torch.utils.data.Subset(test_dataset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    mse_loss = 0.0
    mae_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Patch {patch_size}x{patch_size}, Seq {sequence_length}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions, _ = model(batch["ndvi"], batch["month"], batch["year"], batch["lat"], batch["lon"])
            predictions = predictions.squeeze(-1)

            # Calculate MSE and MAE for this batch
            mse = torch.nn.functional.mse_loss(predictions, batch["target"])
            mae = torch.nn.functional.l1_loss(predictions, batch["target"])

            mse_loss += mse.item()
            mae_loss += mae.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch["target"].cpu().numpy())

    # Calculate final metrics
    l2_score = float(mse_loss / len(test_loader))  # MSE
    l1_score = float(mae_loss / len(test_loader))  # MAE
    rme = float(np.sqrt(l2_score))  # RMSE

    # Calculate F1 score using threshold 0.5
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    pred_binary = (all_predictions >= 0.5).astype(int)
    target_binary = (all_targets >= 0.5).astype(int)
    f1 = float(f1_score(target_binary, pred_binary))

    logging.info(f"Results - RME: {rme:.4f}, L1: {l1_score:.4f}, L2: {l2_score:.4f}, F1: {f1:.4f}")
    return rme, l1_score, l2_score, f1


def generate_performance_table(checkpoint_path: str, patch_sizes: list[int] = [3, 5, 7, 9], sequence_lengths: list[int] = [1, 5, 10, 15, 20], subset_size: float = 0.1) -> pd.DataFrame:
    """
    Generate a performance metrics table for different spatial and sequence configurations.

    Args:
        checkpoint_path: Path to the model checkpoint
        patch_sizes: List of patch sizes to evaluate
        sequence_lengths: List of sequence lengths to evaluate
        subset_size: Fraction of validation set to use

    Returns:
        DataFrame containing the performance metrics table
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data and model
    logging.info("Loading data and model...")
    data = load_data()
    model = load_model(checkpoint_path, device)
    model.eval()

    # Initialize results lists
    results = {"sequence_length": [], "patch_size": [], "rme": [], "l1_mae": [], "l2_mse": [], "f1_score": []}  # int  # int  # float  # float  # float  # float

    # Calculate total configurations
    total_configs = len(patch_sizes) * len(sequence_lengths)
    current_config = 0

    # Evaluate each configuration
    for patch_size in patch_sizes:
        for seq_length in sequence_lengths:
            current_config += 1
            logging.info(f"\nProgress: {current_config}/{total_configs} configurations")

            rme, l1, l2, f1 = evaluate_configuration(model, data, patch_size, seq_length, device, subset_size)

            # Store results
            results["sequence_length"].append(seq_length)
            results["patch_size"].append(patch_size)
            results["rme"].append(rme)
            results["l1_mae"].append(l1)
            results["l2_mse"].append(l2)
            results["f1_score"].append(f1)

    # Create DataFrame with proper column types
    df = pd.DataFrame(results)
    df = df.astype({"sequence_length": "int32", "patch_size": "int32", "rme": "float32", "l1_mae": "float32", "l2_mse": "float32", "f1_score": "float32"})

    return df


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Evaluate the SpatioTemporal Predictor model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--subset", type=float, default=0.1, help="Fraction of validation set to use (default: 0.1)")
    parser.add_argument("--generate-table", action="store_true", help="Generate performance metrics table")

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    if args.generate_table:
        # Generate and display the performance table
        df = generate_performance_table(args.checkpoint, subset_size=args.subset)

        # Display formatted table
        print("\nPerformance Metrics Table:")
        print(df.to_string(float_format=lambda x: "{:.3f}".format(x)))

        # Save to CSV with headers indicating types
        output_path = "performance_metrics.csv"
        header = "sequence_length[int],patch_size[int],rme[float],l1_mae[float],l2_mse[float],f1_score[float]"
        df.to_csv(output_path, index=False, float_format="%.3f")

        logging.info(f"Saved performance metrics table to {output_path}")
    else:
        # Run regular evaluation
        evaluate(args.checkpoint, args.subset)
