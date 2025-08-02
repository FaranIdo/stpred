# STpred: Spatio-Temporal NDVI Prediction

A deep learning framework for predicting vegetation indices (NDVI) using transformer-based models on satellite time series data.

## Overview

STpred implements a spatio-temporal prediction model that forecasts future Normalized Difference Vegetation Index (NDVI) values based on historical satellite data. The model uses a transformer architecture to capture both spatial relationships through image patches and temporal dependencies through time series sequences.

### Key Features

- **Transformer Architecture**: Uses attention mechanisms to model complex spatio-temporal relationships
- **Flexible Input**: Supports variable patch sizes (1x1 to 9x9) and sequence lengths 
- **Multi-modal Data**: Incorporates NDVI values, temporal metadata (month, year), and spatial coordinates
- **Large-scale Dataset**: Trained on Landsat NDVI data spanning 1984-2024
- **GPU Acceleration**: Optimized for CUDA with mixed precision training
- **Comprehensive Monitoring**: TensorBoard integration and detailed logging

## Installation

### Requirements

The project requires Python 3.8+ and the following key dependencies:

```bash
# Core ML framework
torch>=1.9.0
torchvision
tensorboard

# Data processing  
numpy
pandas
sklearn
rasterio

# Visualization and monitoring
matplotlib
tqdm
pynvml

# Geospatial data (optional, for data download)
earthengine-api
geemap
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/FaranIdo/stpred.git
cd stpred
```

2. Create a virtual environment (recommended):
```bash
python -m venv stpred_env
source stpred_env/bin/activate  # On Windows: stpred_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm tensorboard rasterio pynvml
```

4. For data download functionality (optional):
```bash
pip install earthengine-api geemap
```

## Usage

### Data Preparation

The model expects NDVI data in a specific format. If you have access to Google Earth Engine:

1. Use the data download script:
```bash
python dataset/download_ee_data.py
```

2. The expected data format is a NumPy array with shape `(timesteps, height, width, features)` where features include:
   - Index 0: NDVI values
   - Index 1: Month  
   - Index 2: Year
   - Index 3: Latitude
   - Index 4: Longitude

3. Save the processed data as: `data/Landsat_NDVI_time_series_1984_to_2024.npy`

### Training

Start training with default parameters:

```bash
python train.py
```

Add a custom run name:
```bash
python train.py --name my_experiment
```

Training will:
- Create a timestamped directory under `checkpoints/`
- Save model checkpoints, logs, and TensorBoard data
- Use years 1984-2013 for training, 2014-2024 for validation

### Evaluation

Evaluate a trained model:

```bash
python eval.py --checkpoint checkpoints/[run_directory]/models/model_epoch001_valloss0.123456.pt
```

This will generate detailed metrics and save results to the model directory.

### Monitoring

Monitor training progress:

```bash
# View real-time GPU/CPU usage
python monitor.py

# Launch TensorBoard
tensorboard --logdir checkpoints/[run_directory]
```

### Visualization

Generate result visualizations:

```bash
python visualize_results.py --checkpoint [checkpoint_path]
```

## Configuration

Model and training parameters are defined in `config.py`:

### Key Parameters

- **Data Parameters**:
  - `PATCH_SIZE`: Spatial patch size (default: 9x9)
  - `TRAIN_SEQUENCE_LENGTH`: Temporal sequence length for training (default: 20)
  - `DATA_SAMPLE_PERCENTAGE`: Fraction of data to use (default: 0.1)

- **Training Parameters**:
  - `BATCH_SIZE`: Training batch size (default: 8192)
  - `EPOCHS`: Number of training epochs (default: 30)
  - `LEARNING_RATE`: Initial learning rate (default: 2e-3)

- **Model Architecture**:
  - `MODEL_D_MODEL`: Model dimension (default: 256)
  - `MODEL_NUM_HEADS`: Attention heads (default: 8)
  - `MODEL_NUM_LAYERS`: Transformer layers (default: 3)

## Folder Structure

```
stpred/
├── README.md              # Project documentation
├── config.py              # Configuration parameters
├── train.py               # Main training script
├── eval.py                # Model evaluation
├── monitor.py             # System monitoring
├── sanity.py              # Sanity checks
├── visualize_results.py   # Results visualization
├── model/                 # Model architecture
│   ├── __init__.py
│   ├── stpred.py         # Main SpatioTemporalPredictor model
│   ├── transformer.py    # Transformer encoder implementation
│   └── embedding.py      # Feature embedding layers
├── dataset/               # Data processing utilities
│   ├── download_ee_data.py    # Google Earth Engine data download
│   ├── prepare_data.ipynb     # Data preparation notebook
│   └── visualize.ipynb        # Data visualization notebook
├── data/                  # Data directory (created during use)
│   └── Landsat_NDVI_time_series_1984_to_2024.npy
└── checkpoints/           # Training outputs (created during use)
    └── [timestamp]_[name]/
        ├── training.log   # Training logs
        ├── eval.log      # Evaluation logs
        ├── models/       # Model checkpoints
        └── events.out.tfevents.*  # TensorBoard data
```

## Model Architecture

The STpred model consists of three main components:

### 1. TimeSeriesEmbedder (`embedding.py`)
- Embeds NDVI patches using learnable embeddings
- Encodes temporal metadata (year, month) 
- Encodes spatial coordinates (latitude, longitude)
- Combines all features into unified representations

### 2. TimeSeriesTransformer (`transformer.py`)
- Multi-head self-attention for sequence modeling
- Positional encoding for temporal relationships
- Layer normalization and residual connections
- Configurable depth and attention heads

### 3. SpatioTemporalPredictor (`stpred.py`)
- Main model combining embedder and transformer
- Predicts future NDVI values from sequence representations
- Supports variable input dimensions and sequence lengths

### Training Strategy

- **Temporal Split**: Clean separation by years to prevent data leakage
- **Variable Sequences**: Random sequence lengths during training for robustness
- **Spatial Augmentation**: Random patch sizes to improve generalization
- **Mixed Precision**: Automatic mixed precision for faster training
- **Gradient Accumulation**: Support for large effective batch sizes

## Data Requirements

The model expects preprocessed satellite data with the following characteristics:

- **Temporal Coverage**: Multi-year time series (1984-2024 in the reference dataset)
- **Spatial Resolution**: Regular grid with geographic coordinates
- **Data Format**: 5-channel arrays (NDVI, month, year, latitude, longitude)
- **File Format**: NumPy array saved as `.npy` file
- **Missing Data**: Should be handled during preprocessing

## Performance Notes

- **GPU Requirements**: CUDA-capable GPU recommended (8GB+ VRAM)
- **Memory Usage**: Scales with batch size and sequence length
- **Training Time**: Varies by dataset size and hardware (hours to days)
- **Checkpointing**: Automatic saving of best models by validation loss

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is available under the terms specified in the repository license.