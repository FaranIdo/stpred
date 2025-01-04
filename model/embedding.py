import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding from the "Attention is All You Need" paper.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a long enough P matrix of shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div term (1 / 10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Fill pe[:, 0::2] with sin, pe[:, 1::2] with cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as a buffer so it’s not a learnable parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model) with added positional encoding
        """
        seq_len = x.size(1)
        # Add the positional encoding up to seq_len
        x = x + self.pe[:, :seq_len, :]
        return x


class CyclicalMonthEncoder(nn.Module):
    """Encodes months (1-12) into cyclical sine/cosine features."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, month: torch.Tensor) -> torch.Tensor:
        """
        Args:
            month: Tensor of shape (batch_size, seq_len) with values in [1..12]
        Returns:
            month_features: Tensor of shape (batch_size, seq_len, 2) containing sin/cos features
        """
        month_sin = torch.sin(2 * math.pi * (month / 12.0))
        month_cos = torch.cos(2 * math.pi * (month / 12.0))
        return torch.stack([month_sin, month_cos], dim=-1)


class TimeSeriesEmbedder(nn.Module):
    """
    Embeds NDVI, cyclical month, and (year - first_year) into a d_model-dimensional vector.
    Optionally adds sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, ndvi_embed_dim: int = 8, year_embed_dim: int = 8, use_positional_encoding: bool = True, max_seq_len: int = 5000, first_year: int = 1984):
        """
        Args:
            d_model: Final dimension for Transformer input.
            ndvi_embed_dim: Dimension of the embedding we learn for NDVI.
            year_embed_dim: Dimension of the embedding we learn for Year.
            use_positional_encoding: Whether to add sinusoidal PE.
            max_seq_len: Maximum sequence length we might need (for positional encoding).
            first_year: Integer or float used to shift the year so it's smaller (e.g. year-2000). First year in the dataset.
        """
        super().__init__()
        self.first_year = first_year
        self.ndvi_embed = nn.Linear(1, ndvi_embed_dim)  # NDVI -> embedding
        self.year_embed = nn.Linear(1, year_embed_dim)  # Year -> embedding
        self.month_encoder = CyclicalMonthEncoder()

        # Combine dimension: ndvi_embed_dim + 2 (month sin/cos) + year_embed_dim
        combined_dim = ndvi_embed_dim + 2 + year_embed_dim
        self.final_proj = nn.Linear(combined_dim, d_model)

        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

    def forward(self, ndvi: torch.Tensor, month: torch.Tensor, year: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ndvi: Tensor of shape (batch_size, seq_len) containing NDVI values
            month: Tensor of shape (batch_size, seq_len) with values in [1..12]
            year: Tensor of shape (batch_size, seq_len) with float years
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, d_model)
        """
        ndvi_3d = ndvi.unsqueeze(-1)
        year_3d = (year - self.first_year).unsqueeze(-1)

        # Get embeddings for each component
        ndvi_e = self.ndvi_embed(ndvi_3d)
        year_e = self.year_embed(year_3d)
        month_e = self.month_encoder(month)  # Returns (batch, seq_len, 2)

        # Concatenate all features
        combined = torch.cat([ndvi_e, month_e, year_e], dim=-1)

        # Final projection
        x = self.final_proj(combined)

        # Optionally add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        return x


def test_time_series_embedder():
    batch_size = 2
    seq_len = 5
    d_model = 16

    # Dummy inputs
    ndvi = torch.rand(batch_size, seq_len)  # e.g., NDVI in [0..1]
    month = torch.randint(1, 13, (batch_size, seq_len))  # months in 1..12
    year = 2000 + torch.randint(0, 5, (batch_size, seq_len), dtype=torch.float)  # Convert to float
    first_year = 2000

    # print shape of inputs
    print("ndvi shape:", ndvi.shape)
    print("month shape:", month.shape)
    print("year shape:", year.shape)

    embedder = TimeSeriesEmbedder(d_model=d_model, ndvi_embed_dim=8, year_embed_dim=8, first_year=first_year)

    embeddings = embedder(ndvi, month, year)
    print("embeddings shape:", embeddings.shape)  # (batch_size, seq_len, d_model)


if __name__ == "__main__":
    test_time_series_embedder()
