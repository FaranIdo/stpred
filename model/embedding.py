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


class LatLonEncoder(nn.Module):
    """
    Converts latitude/longitude in degrees to (x, y, z) on a unit sphere,
    then applies a linear layer to get a lat-lon embedding of dimension embed_dim.
    """

    def __init__(self, embed_dim: int = 8):
        """
        Args:
            embed_dim: dimension of the embedding returned for each (lat, lon).
        """
        super().__init__()
        # We'll embed from 3 features (x, y, z) -> embed_dim
        self.linear = nn.Linear(3, embed_dim)

    def forward(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lat_deg: (batch_size, seq_len) of latitudes in degrees
            lon_deg: (batch_size, seq_len) of longitudes in degrees
        Returns:
            latlon_embed: (batch_size, seq_len, embed_dim)
        """
        # 1) Convert degrees -> radians
        lat_rad = lat_deg * math.pi / 180.0
        lon_rad = lon_deg * math.pi / 180.0

        # 2) Convert to (x, y, z) on the unit sphere
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)

        # 3) Stack into shape (batch_size, seq_len, 3)
        xyz = torch.stack([x, y, z], dim=-1)

        # 4) Linear embedding (3 -> embed_dim)
        latlon_embed = self.linear(xyz)
        return latlon_embed


class TimeSeriesEmbedder(nn.Module):
    """
    Embeds NDVI patches, cyclical month, (year - first_year), and lat/lon
    into a d_model-dimensional vector. Optionally adds sinusoidal positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int = 1,  # 1 means 1x1, 3 means 3x3, 5 means 5x5
        ndvi_embed_dim: int = 32,
        year_embed_dim: int = 8,
        latlon_embed_dim: int = 8,
        use_positional_encoding: bool = True,
        max_seq_len: int = 5000,
        first_year: int = 1984,
    ):
        """
        Args:
            d_model: Final dimension for Transformer input.
            patch_size: Size of the NDVI spatial patch (1, 3, or 5).
            ndvi_embed_dim: Dimension of the embedding we learn for NDVI patch.
            year_embed_dim: Dimension of the embedding we learn for Year.
            latlon_embed_dim: Dimension of the embedding for (lat, lon).
            use_positional_encoding: Whether to add sinusoidal PE.
            max_seq_len: Maximum sequence length for positional encoding.
            first_year: integer/float for shifting the year (e.g. year-2000).
        """
        super().__init__()
        self.first_year = first_year
        self.patch_size = patch_size
        self.n_pixels = patch_size * patch_size

        # NDVI and year embeddings
        self.ndvi_embed = nn.Linear(self.n_pixels, ndvi_embed_dim)  # NDVI patch -> embedding
        self.year_embed = nn.Linear(1, year_embed_dim)  # Year -> embedding

        # Month encoder -> 2 features (sin, cos)
        self.month_encoder = CyclicalMonthEncoder()

        # Lat/lon encoder (handles the lat-lon -> (x,y,z) -> linear)
        self.latlon_encoder = LatLonEncoder(embed_dim=latlon_embed_dim)

        # Combine dimension:
        #   NDVI + Month(2) + Year + LatLonEmbed
        combined_dim = ndvi_embed_dim + 2 + year_embed_dim + latlon_embed_dim

        # Final projection to d_model
        self.final_proj = nn.Linear(combined_dim, d_model)

        # (Optional) Positional Encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

    def forward(
        self,
        ndvi_patch: torch.Tensor,  # Shape: [batch_size, seq_len, patch_size*patch_size]
        month: torch.Tensor,  # Shape: [batch_size, seq_len]
        year: torch.Tensor,  # Shape: [batch_size, seq_len]
        lat_deg: torch.Tensor,  # Shape: [batch_size, seq_len]
        lon_deg: torch.Tensor,  # Shape: [batch_size, seq_len]
    ) -> torch.Tensor:
        """
        Args:
            ndvi_patch: Flattened NDVI patches
            month: months in [1..12] (e.g. 1, 2, 3, ..., 12)
            year: float years (e.g. 2000.0)
            lat_deg: latitude in degrees (e.g. 30.0)
            lon_deg: longitude in degrees (e.g. 30.0)
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        # --- 1) NDVI Patch Embedding
        ndvi_e = self.ndvi_embed(ndvi_patch)  # [batch, seq_len, ndvi_embed_dim]

        # --- 2) Year Embedding (shifted by first_year)
        year_shifted = (year.float() - self.first_year).unsqueeze(-1)
        year_e = self.year_embed(year_shifted)  # [batch, seq_len, year_embed_dim]

        # --- 3) Month -> cyclical (2D)
        month_e = self.month_encoder(month)  # [batch, seq_len, 2]

        # --- 4) Lat/Lon -> (x, y, z) -> linear embedding
        latlon_e = self.latlon_encoder(lat_deg, lon_deg)  # [batch, seq_len, latlon_embed_dim]

        # --- 5) Concatenate all features
        combined = torch.cat([ndvi_e, month_e, year_e, latlon_e], dim=-1)

        # --- 6) Final projection to d_model
        x_out = self.final_proj(combined)

        # --- 7) (Optional) Sinusoidal positional encoding
        if self.use_positional_encoding:
            x_out = self.pos_encoder(x_out)

        return x_out


def test_time_series_embedder():
    batch_size, seq_len = 2, 5
    d_model = 16
    patch_size = 3  # 3x3 patch

    # Create the embedder
    embedder = TimeSeriesEmbedder(
        d_model=d_model,
        patch_size=patch_size,
        ndvi_embed_dim=4,
        year_embed_dim=4,
        latlon_embed_dim=4,
        use_positional_encoding=True,
        max_seq_len=500,
        first_year=1984,
    )

    # Example input data
    ndvi_patch = torch.rand(batch_size, seq_len, patch_size * patch_size)  # Flattened patches
    month = torch.randint(1, 13, (batch_size, seq_len))  # months [1..12]
    year = 1984 + torch.randint(0, 35, (batch_size, seq_len))  # random years
    lat = torch.full((batch_size, seq_len), 31.76)  # ~31.76 deg
    lon = torch.full((batch_size, seq_len), 34.77)  # ~34.77 deg

    # Get embeddings
    embeddings = embedder(ndvi_patch, month, year, lat, lon)
    print("Output embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    test_time_series_embedder()
