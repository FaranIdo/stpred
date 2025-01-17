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


def get_predictions(model: torch.nn.Module, data: np.ndarray, start_x: int, start_y: int, area_size: int, start_year: int, start_month: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions for a specific area and time by processing all pixels at once"""
    model.eval()

    # Find the target timestep that matches our desired year and month
    time_mask = (data[..., 2] == start_year) & (data[..., 1] == start_month)
    target_timestep = np.where(time_mask.any(axis=(1, 2)))[0][0]

    # Get 6 consecutive timesteps ending at our target time
    sequence_length = 20  # 5 input timesteps + 1 target timestep
    start_timestep = target_timestep - sequence_length + 1
    selected_data = data[start_timestep : target_timestep + 1]

    # Extract just the area we want
    area_data = selected_data[:, start_y : start_y + area_size, start_x : start_x + area_size, :]

    # Get dimensions
    num_timesteps, height, width, num_features = area_data.shape

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


def plot_comparison(predictions: np.ndarray, targets: np.ndarray, start_x: int, start_y: int, area_size: int, time_points: list[tuple[int, int]], output_dir: str) -> None:
    """Plot ground truth vs predictions comparison for multiple timesteps"""
    num_timepoints = len(time_points)
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    # Create figure with adjusted spacing and left margin
    fig = plt.figure(figsize=(5 * num_timepoints - 4, 8))  # Increased multiplier from 4 to 5 and height from 6 to 8
    # Add left margin for labels
    gs = fig.add_gridspec(2, num_timepoints, hspace=0.1, wspace=0.0, left=0.02)
    axes = gs.subplots()

    # Handle the case when there's only one timepoint
    if num_timepoints == 1:
        axes = axes.reshape(2, 1)

    # Create a common colorbar for all plots
    vmin, vmax = 0, 1

    # Move row labels much closer to the plots
    fig.text(0.005, 0.7, "Ground Truth", fontsize=22, rotation=90, va="center")
    fig.text(0.005, 0.3, "Prediction", fontsize=22, rotation=90, va="center")

    for idx, ((year, month), target, pred) in enumerate(zip(time_points, targets, predictions)):
        # Plot ground truth
        im1 = axes[0, idx].imshow(target, cmap="YlGn", vmin=vmin, vmax=vmax)
        axes[0, idx].set_title(f"{month_names[month-1]} {year}", fontsize=24, pad=10)
        axes[0, idx].axis("off")

        # Plot prediction
        im2 = axes[1, idx].imshow(pred, cmap="YlGn", vmin=vmin, vmax=vmax)
        axes[1, idx].axis("off")

    # Add vertical colorbar with adjusted positioning
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Moved slightly left from 0.95 to 0.92
    plt.colorbar(im1, cax=cax, orientation="vertical", label="NDVI Value")

    # Save with maximum quality
    plt.savefig(
        os.path.join(output_dir, f"ndvi_comparison_{start_x}_{start_y}_{area_size}.png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,  # Increased DPI from 300 to 600
    )  # Added maximum quality parameter
    plt.close()

    # Create error plot with adjusted spacing
    error_fig = plt.figure(figsize=(5 * num_timepoints - 4, 4))  # Increased size here too
    gs_error = error_fig.add_gridspec(1, num_timepoints, wspace=0.0)
    error_axes = gs_error.subplots()

    # Handle the case when there's only one timepoint
    if num_timepoints == 1:
        error_axes = np.array([error_axes])

    for idx, ((year, month), target, pred) in enumerate(zip(time_points, targets, predictions)):
        error = np.abs(pred - target)
        im = error_axes[idx].imshow(error, cmap="Reds", vmin=0, vmax=1)
        error_axes[idx].set_title(f"Error - {month_names[month-1]} {year}", fontsize=24, pad=10)
        error_axes[idx].axis("off")

    # Add vertical colorbar with adjusted positioning for error plot
    error_cax = error_fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Moved slightly left from 0.95 to 0.92
    plt.colorbar(im, cax=error_cax, orientation="vertical", label="Absolute Error")

    # Save error plot with maximum quality
    plt.savefig(os.path.join(output_dir, f"error_map_{start_x}_{start_y}_{area_size}.png"), dpi=600, bbox_inches="tight", pad_inches=0.05)  # Increased DPI  # Added maximum quality parameter
    plt.close()


def visualize_predictions(checkpoint_path: str, start_x: int, start_y: int, area_size: int, time_points: list[tuple[int, int]], output_dir: str, batch_size: int = 32) -> None:
    """Visualize model predictions for a specific area across multiple time points

    Args:
        checkpoint_path: Path to model checkpoint
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        area_size: Size of area to visualize
        time_points: List of (year, month) tuples
        output_dir: Output directory for plots
        batch_size: Batch size for processing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_visualization_logging(output_dir)
    logging.info(f"Using device: {device}")

    data = load_data()
    model = load_model(checkpoint_path, device)

    all_predictions = []
    all_targets = []
    all_metrics = []

    for year, month in time_points:
        predictions, targets = get_predictions(model, data, start_x, start_y, area_size, year, month, device)
        all_predictions.append(predictions)
        all_targets.append(targets)

        # Calculate error metrics
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        all_metrics.append((mae, mse, rmse))

        logging.info(f"\nError Metrics for {month}/{year}:")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"MSE: {mse:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")

    # Plot results
    plot_comparison(all_predictions, all_targets, start_x, start_y, area_size, time_points, output_dir)
    logging.info(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NDVI predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--start-x", type=int, default=200, help="Starting X coordinate")
    parser.add_argument("--start-y", type=int, default=200, help="Starting Y coordinate")
    parser.add_argument("--area-size", type=int, default=100, help="Size of area to visualize")
    parser.add_argument("--time-points", type=str, required=True, help="List of year,month pairs in format '2019,1;2019,2;2019,3'")
    parser.add_argument("--output-dir", type=str, default="visualization_results", help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")

    args = parser.parse_args()

    # Parse time points from string
    time_points = [tuple(map(int, tp.split(","))) for tp in args.time_points.split(";")]

    visualize_predictions(args.checkpoint, args.start_x, args.start_y, args.area_size, time_points, args.output_dir, args.batch_size)
