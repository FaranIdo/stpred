import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import logging
import argparse
import os
from tqdm import tqdm

from eval import load_model, load_data, NDVIDataset
from config import *


def setup_visualization_logging(output_dir: str) -> None:
    """Setup logging for visualization"""
    os.makedirs(output_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Setup file handler
    log_file = os.path.join(output_dir, "visualization.log")
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


def get_predictions(model: torch.nn.Module, data: np.ndarray, start_x: int, start_y: int, area_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions for a specific area by processing all pixels at once"""
    model.eval()

    # Create dataset for the specific area
    test_mask = (data[..., 2] >= VAL_YEARS[0]) & (data[..., 2] <= VAL_YEARS[1])
    test_data = data[test_mask.any(axis=(1, 2))]

    # Extract just the area we want - note the time dimension comes first
    area_data = test_data[:, start_y : start_y + area_size, start_x : start_x + area_size, :]

    # Get dimensions
    num_timesteps, height, width, num_features = area_data.shape
    sequence_length = 6  # 5 input timesteps + 1 target timestep

    # Create patches for each pixel
    patches = np.zeros((height, width, sequence_length, PATCH_SIZE, PATCH_SIZE))  # Only for NDVI
    months = np.zeros((height, width, sequence_length))
    years = np.zeros((height, width, sequence_length))
    lats = np.zeros((height, width, sequence_length))
    lons = np.zeros((height, width, sequence_length))

    # For each pixel in the area
    for i in range(height):
        for j in range(width):
            # Calculate patch boundaries with padding for edges
            patch_start_y = max(0, i - PATCH_SIZE // 2)
            patch_end_y = min(height, i + PATCH_SIZE // 2 + 1)
            patch_start_x = max(0, j - PATCH_SIZE // 2)
            patch_end_x = min(width, j + PATCH_SIZE // 2 + 1)

            # Extract and pad patch if necessary for NDVI
            patch = area_data[:sequence_length, patch_start_y:patch_end_y, patch_start_x:patch_end_x, 0]  # Only NDVI channel
            if patch.shape[1] < PATCH_SIZE or patch.shape[2] < PATCH_SIZE:
                padded_patch = np.zeros((sequence_length, PATCH_SIZE, PATCH_SIZE))
                y_offset = (PATCH_SIZE - patch.shape[1]) // 2
                x_offset = (PATCH_SIZE - patch.shape[2]) // 2
                padded_patch[:, y_offset : y_offset + patch.shape[1], x_offset : x_offset + patch.shape[2]] = patch
                patch = padded_patch

            patches[i, j] = patch
            # Store other features for all timesteps (including target)
            months[i, j] = area_data[:sequence_length, i, j, 1]  # Month
            years[i, j] = area_data[:sequence_length, i, j, 2]  # Year
            lats[i, j] = area_data[:sequence_length, i, j, 3]  # Latitude
            lons[i, j] = area_data[:sequence_length, i, j, 4]  # Longitude

    # Prepare inputs for the model
    batch_size = height * width
    patches = patches.reshape(batch_size, sequence_length, PATCH_SIZE * PATCH_SIZE)
    months = months.reshape(batch_size, sequence_length)
    years = years.reshape(batch_size, sequence_length)
    lats = lats.reshape(batch_size, sequence_length)
    lons = lons.reshape(batch_size, sequence_length)

    # Extract targets (last timestep NDVI values)
    targets = patches[:, -1, PATCH_SIZE * PATCH_SIZE // 2].reshape(height, width)  # Center pixel of last timestep

    # Convert to torch tensors and move to device
    ndvi_input = torch.FloatTensor(patches).to(device)
    month_input = torch.FloatTensor(months).to(device)
    year_input = torch.FloatTensor(years).to(device)
    lat_input = torch.FloatTensor(lats).to(device)
    lon_input = torch.FloatTensor(lons).to(device)

    # Get predictions
    with torch.no_grad():
        predictions, _ = model(ndvi_input, month_input, year_input, lat_input, lon_input)
        predictions = predictions.squeeze(-1).cpu().numpy()

    # Reshape predictions to match targets shape (height, width)
    predictions = predictions.reshape(height, width)

    return predictions, targets


def plot_comparison(predictions: np.ndarray, targets: np.ndarray, start_x: int, start_y: int, area_size: int, output_dir: str) -> None:
    """Plot ground truth vs predictions comparison for single timestep"""
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NDVI Comparison: Ground Truth vs Predictions", fontsize=16)

    # Plot ground truth
    im1 = ax1.imshow(targets, cmap="YlGn", vmin=0, vmax=1)
    ax1.set_title("Ground Truth")
    ax1.axis("off")

    # Plot prediction
    im2 = ax2.imshow(predictions, cmap="YlGn", vmin=0, vmax=1)
    ax2.set_title("Prediction")
    ax2.axis("off")

    # Add colorbar
    plt.colorbar(im1, ax=[ax1, ax2], label="NDVI Value")

    # Add error plot
    error = np.abs(predictions - targets)
    error_fig = plt.figure(figsize=(10, 8))
    plt.imshow(error, cmap="Reds", vmin=0, vmax=1)
    plt.colorbar(label="Absolute Error")
    plt.title("Prediction Error")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"error_map_{start_x}_{start_y}_{area_size}.png"))
    plt.close(error_fig)

    # Save comparison plot
    plt.savefig(os.path.join(output_dir, f"ndvi_comparison_{start_x}_{start_y}_{area_size}.png"), dpi=300)
    plt.close()


def visualize_predictions(checkpoint_path: str, start_x: int, start_y: int, area_size: int, output_dir: str, batch_size: int = 32) -> None:
    """Visualize model predictions for a specific area"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_visualization_logging(output_dir)
    logging.info(f"Using device: {device}")

    # Load data and model
    data = load_data()
    model = load_model(checkpoint_path, device)

    # Get predictions
    predictions, targets = get_predictions(model, data, start_x, start_y, area_size, device)

    # Plot results
    plot_comparison(predictions, targets, start_x, start_y, area_size, output_dir)
    logging.info(f"Saved visualization to {output_dir}")

    # Calculate and log error metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    logging.info(f"\nError Metrics for area ({start_x}, {start_y}, {area_size}x{area_size}):")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NDVI predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--start-x", type=int, default=200, help="Starting X coordinate")
    parser.add_argument("--start-y", type=int, default=200, help="Starting Y coordinate")
    parser.add_argument("--area-size", type=int, default=100, help="Size of area to visualize")
    parser.add_argument("--output-dir", type=str, default="visualization_results", help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")

    args = parser.parse_args()

    visualize_predictions(args.checkpoint, args.start_x, args.start_y, args.area_size, args.output_dir, args.batch_size)
