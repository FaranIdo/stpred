import torch
import torch.nn as nn
import math
from .embedding import ContinuousTemporalEncoding


class SequencePositionalEncoding(nn.Module):
    # 100 - max sequence of years from example 1980 to 2080
    def __init__(self, d_model, max_len=100):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # [d_model/2,]

        # keep pe[0,:] to zeros
        pe[1:, 0::2] = torch.sin(position * div_term)  # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)  # broadcasting to [max_len, d_model/2]

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[x]


class TemporalPositionalNDVITransformer(nn.Module):

    def __init__(
        self, embedding_dim: int, attn_heads: int, num_encoder_layers: int, sequence_length: int, start_year: int, end_year: int, max_seq_length: int = 50, num_features: int = 1, dropout: float = 0.1
    ):
        """
        Initializes the TemporalPositionalNDVITransformer model.

        Args:
            embedding_dim (int): The dimension of the model.
            attn_heads (int): The number of attention heads.
            num_encoder_layers (int): The number of encoder layers.
            sequence_length (int): The length of the time series sequence - how many timesteps the network looks at to predict the next timestep
            start_year (int): The starting year for temporal encoding.
            end_year (int): The ending year for temporal encoding.
            max_seq_length (int, optional): The maximum sequence length for positional encoding. Defaults to 50.
            num_features (int, optional): The number of input features. Defaults to 1.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(TemporalPositionalNDVITransformer, self).__init__()

        dim_feedforward = embedding_dim * 4

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # Embeddings

        # Embed the NDVI values- only num_features and not sequence_length, because we don't want to "interact" the features yet
        # Interactions will be done in the transformer encoder
        self.ndvi_embed = nn.Linear(num_features, embedding_dim)

        # Temporal encoding
        self.temporal_encoding = ContinuousTemporalEncoding(embedding_dim, start_year, end_year)

        # Positional encoding - Captture the order of values in the input sequence
        self.positional_encoding = SequencePositionalEncoding(embedding_dim, max_seq_length)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, attn_heads, dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        # Pooling layer
        self.pooling = nn.MaxPool1d(self.sequence_length)

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TemporalPositionalNDVITransformer model.

        Args:
            ndvi (torch.Tensor): The NDVI values with shape [batch_size, seq_len, num_features].
            years (torch.Tensor): The years with shape [batch_size, seq_len].
            seasons (torch.Tensor): The seasons with shape [batch_size, seq_len] - 0 for winter, 1 for summer

        Returns:
            torch.Tensor: The output of the model.
        """
        batch_size, seq_len = ndvi.shape

        ndvi = ndvi.unsqueeze(2)

        # Embed NDVI values
        ndvi_embedding = self.ndvi_embed(ndvi)

        # Create temporal encoding
        temporal_encoding = self.temporal_encoding(years, seasons)

        # Create positional encoding
        positions = torch.arange(seq_len, device=ndvi.device).unsqueeze(0).expand(batch_size, seq_len)
        positional_encoding = self.positional_encoding(positions)

        # Combine NDVI embedding with temporal and positional encodings
        x = ndvi_embedding + temporal_encoding + positional_encoding

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer encoder
        output = self.transformer_encoder(x.transpose(0, 1))

        # Apply pooling over the sequence dimension (first dimension)
        # required shape before pooling : [batch_size, hidden_size, seq_len]
        # required shape after pooling : [batch_size, hidden_size]
        output = self.pooling(output.permute(1, 2, 0)).squeeze()

        # Pass through output layer
        output = self.output_layer(output)
        return output.squeeze(-1)
