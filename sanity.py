import torch
import numpy as np
from model.stpred import SpatioTemporalPredictor
from train import NDVIDataset
from config import *  # Import all constants from config.py

def test_stpred():
    # Test parameters
    batch_size = 20
    seq_len = TRAIN_SEQUENCE_LENGTH
    patch_size = PATCH_SIZE

    # Create model
    model = SpatioTemporalPredictor(
        d_model=MODEL_D_MODEL,
        patch_size=patch_size,
        ndvi_embed_dim=MODEL_NDVI_EMBED_DIM,
        year_embed_dim=MODEL_YEAR_EMBED_DIM,
        latlon_embed_dim=MODEL_LATLON_EMBED_DIM,
        num_heads=MODEL_NUM_HEADS,
        num_layers=MODEL_NUM_LAYERS,
        d_ff=MODEL_D_FF,
        dropout=MODEL_DROPOUT,
        max_seq_len=MODEL_MAX_SEQ_LEN,
        first_year=MODEL_FIRST_YEAR,
    )

    # Create sample inputs
    ndvi_patch = torch.rand(batch_size, seq_len, patch_size * patch_size)
    month = torch.randint(1, 13, (batch_size, seq_len))
    year = MODEL_FIRST_YEAR + torch.randint(0, 35, (batch_size, seq_len))
    lat = torch.full((batch_size, seq_len), 31.76)
    lon = torch.full((batch_size, seq_len), 34.77)

    # Forward pass
    predictions, transformer_output = model(ndvi_patch, month, year, lat, lon)

    print(f"Input NDVI shape: {ndvi_patch.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")
    print(f"Predictions shape: {predictions.shape}")  # Should be (batch_size, 1)

    # Verify prediction shape is correct (one value per item in batch)
    assert predictions.shape == (batch_size, 1)
    print("\n✅ SpatioTemporalPredictor test passed successfully!")


def test_ndvidataset_getitem():
    # Create sample data
    timesteps = 20
    height = 50
    width = 40
    features = 5  # [ndvi, month, year, lat, lon]
    data = np.random.rand(timesteps, height, width, features)

    # Initialize dataset with default parameters
    sequence_length = 10
    patch_size = 21
    dataset = NDVIDataset(data, sequence_length=sequence_length, patch_size=patch_size, sequence_masking=False)

    # Get a sample
    sample = dataset[0]

    # Test that all expected keys are present
    expected_keys = {"ndvi", "month", "year", "lat", "lon", "target", "patch_size", "sequence_length"}
    assert set(sample.keys()) == expected_keys

    # Test shapes
    assert sample["ndvi"].shape == (sequence_length, patch_size * patch_size)
    assert sample["month"].shape == (sequence_length,)
    assert sample["year"].shape == (sequence_length,)
    assert sample["lat"].shape == (sequence_length,)
    assert sample["lon"].shape == (sequence_length,)
    assert sample["target"].shape == ()  # Scalar
    assert sample["patch_size"].shape == ()  # Scalar

    # Test types
    assert isinstance(sample["ndvi"], torch.Tensor)
    assert isinstance(sample["month"], torch.Tensor)
    assert isinstance(sample["year"], torch.Tensor)
    assert isinstance(sample["lat"], torch.Tensor)
    assert isinstance(sample["lon"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)
    assert isinstance(sample["patch_size"], torch.Tensor)

    # Test data ranges
    assert sample["patch_size"] >= 3
    assert sample["patch_size"] <= patch_size
    assert sample["patch_size"] % 2 == 1  # Should be odd number

    # Test that NDVI values are properly padded
    # The actual patch size may be smaller than patch_size due to random masking
    # but the output should always be padded to patch_size * patch_size
    assert not torch.any(torch.isnan(sample["ndvi"]))

    # Test that target is a single value
    assert sample["target"].numel() == 1
    print("\n✅ NDVIDataset test passed successfully!")


def test_ndvidataset_sequence_lengths():
    # Create sample data
    timesteps = 90
    height = 50
    width = 40
    features = 5  # [ndvi, month, year, lat, lon]
    data = np.random.rand(timesteps, height, width, features)

    # Initialize dataset with a large sequence length
    max_sequence_length = 80
    patch_size = 21
    dataset = NDVIDataset(data, sequence_length=max_sequence_length, patch_size=patch_size, sequence_masking=True)

    # Get multiple samples and check their sequence lengths
    num_samples = 50
    sequence_lengths = []
    for _ in range(num_samples):
        # Set a new sequence length before getting the sample
        dataset.set_sequence_length()
        curr_seq_len = dataset.current_sequence_length

        sample = dataset[0]  # Always get first item, but sequence length should vary
        sample_seq_len = sample["sequence_length"].item()

        # Verify the sample's sequence length matches the dataset's current sequence length
        assert curr_seq_len == sample_seq_len, f"Sample sequence length {sample_seq_len} doesn't match dataset's current sequence length {curr_seq_len}"
        sequence_lengths.append(curr_seq_len)

        # Basic assertions for each sample
        assert curr_seq_len >= 1, f"Sequence length should be greater than 1, got {curr_seq_len}"
        assert curr_seq_len <= max_sequence_length, f"Sequence length {curr_seq_len} exceeds maximum {max_sequence_length}"
        assert sample["ndvi"].shape[0] == curr_seq_len, "NDVI sequence length doesn't match sequence_length"
        assert sample["month"].shape[0] == curr_seq_len, "Month sequence length doesn't match sequence_length"
        assert sample["year"].shape[0] == curr_seq_len, "Year sequence length doesn't match sequence_length"
        assert sample["lat"].shape[0] == curr_seq_len, "Latitude sequence length doesn't match sequence_length"
        assert sample["lon"].shape[0] == curr_seq_len, "Longitude sequence length doesn't match sequence_length"

    # Verify we got at least some variation in sequence lengths
    unique_lengths = set(sequence_lengths)
    assert len(unique_lengths) > 1, "Should get varying sequence lengths across multiple gets"

    print(f"\nObserved sequence lengths: {sorted(unique_lengths)}")
    print("✅ NDVIDataset sequence length variation test passed successfully!")


if __name__ == "__main__":
    print("\nRunning sanity tests...")
    test_stpred()
    test_ndvidataset_getitem()
    test_ndvidataset_sequence_lengths()
    print("\n🎉 All sanity tests passed successfully!")
