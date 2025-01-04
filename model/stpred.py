from typing import Tuple
import torch
import torch.nn as nn
from .embedding import TimeSeriesEmbedder
from .transformer import TimeSeriesTransformer


class SpatioTemporalPredictor(nn.Module):
    """
    Combined model that:
    1. Embeds the input features using TimeSeriesEmbedder
    2. Processes the sequence using TimeSeriesTransformer
    3. Takes the last sequence element and predicts the next NDVI value
    """

    def __init__(
        self,
        d_model: int = 256,
        patch_size: int = 1,
        ndvi_embed_dim: int = 32,
        year_embed_dim: int = 8,
        latlon_embed_dim: int = 8,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        first_year: int = 1984,
    ) -> None:
        """
        Args:
            d_model: Dimension of the model
            patch_size: Size of NDVI patches (1 for 1x1, 3 for 3x3, etc.)
            ndvi_embed_dim: Dimension for NDVI embedding
            year_embed_dim: Dimension for year embedding
            latlon_embed_dim: Dimension for lat/lon embedding
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Transformer feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            first_year: Reference year for year encoding
        """
        super().__init__()

        # Validate patch size is odd (to have a center pixel)
        assert patch_size % 2 == 1, "Patch size must be odd to have a center pixel"

        # 1. Embedding layer
        self.embedder = TimeSeriesEmbedder(
            d_model=d_model,
            patch_size=patch_size,
            ndvi_embed_dim=ndvi_embed_dim,
            year_embed_dim=year_embed_dim,
            latlon_embed_dim=latlon_embed_dim,
            use_positional_encoding=True,
            max_seq_len=max_seq_len,
            first_year=first_year,
        )

        # 2. Transformer
        self.transformer = TimeSeriesTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 3. Final prediction head (d_model -> 1 NDVI value for center pixel)
        self.prediction_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))  # Predict single NDVI value for center pixel

    def forward(
        self,
        ndvi_patch: torch.Tensor,
        month: torch.Tensor,
        year: torch.Tensor,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ndvi_patch: (batch_size, seq_len, patch_size*patch_size) NDVI values
            month: (batch_size, seq_len) months [1..12]
            year: (batch_size, seq_len) years
            lat_deg: (batch_size, seq_len) latitudes
            lon_deg: (batch_size, seq_len) longitudes

        Returns:
            predictions: (batch_size, 1) Predicted NDVI for center pixel of next timestep
            transformer_output: (batch_size, seq_len, d_model) Full transformer output
        """
        # 1. Embed all inputs
        embedded = self.embedder(ndvi_patch, month, year, lat_deg, lon_deg)

        # 2. Process with transformer
        transformer_output = self.transformer(embedded)

        # 3. Take last sequence element and predict next central NDVI
        last_state = transformer_output[:, -1, :]  # (batch_size, d_model)
        predictions = self.prediction_head(last_state)  # (batch_size, 1)

        return predictions, transformer_output
