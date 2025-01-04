import torch
from model.stpred import SpatioTemporalPredictor


def test_stpred():
    # Test parameters
    batch_size = 2
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


if __name__ == "__main__":
    test_stpred()
