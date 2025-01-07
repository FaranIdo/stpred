import torch
import numpy as np
from model.stpred import SpatioTemporalPredictor
from train import NDVIDataset

def test_stpred():
    # Test parameters
    batch_size = 20
    seq_len = 10
    patch_size = 3  # 3x3 patch with center pixel

    # Create model
    model = SpatioTemporalPredictor(
        d_model=256,
        patch_size=patch_size,
        ndvi_embed_dim=32,
        year_embed_dim=8,
        latlon_embed_dim=8,
        num_heads=8,
        num_layers=3,
    )

    # Create sample inputs
    ndvi_patch = torch.rand(batch_size, seq_len, patch_size * patch_size)
    month = torch.randint(1, 13, (batch_size, seq_len))
    year = 1984 + torch.randint(0, 35, (batch_size, seq_len))
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
    dataset = NDVIDataset(data, sequence_length=sequence_length, patch_size=patch_size)

    # Get a sample
    sample = dataset[0]

    # Test that all expected keys are present
    expected_keys = {"ndvi", "month", "year", "lat", "lon", "target", "patch_size"}
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


if __name__ == "__main__":
    print("\nRunning sanity tests...")
    test_stpred()
    test_ndvidataset_getitem()
    print("\n🎉 All sanity tests passed successfully!")
