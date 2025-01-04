from typing import Optional
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int = 2048, dropout: float = 0.1, num_classes: int = 1, activation: str = "relu") -> None:
        """Transformer model for time series prediction.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            num_classes: Number of output classes (1 for regression)
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=encoder_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor from embedder (batch_size, seq_len, d_model)

        Returns:
            predictions: (batch_size, seq_len, num_classes)
        """
        # Pass through transformer
        x = self.transformer(x)
        return x


def test_transformer():
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 8
    num_layers = 3

    # Create model
    model = TimeSeriesTransformer(d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=1024, dropout=0.1, num_classes=1)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_transformer()
