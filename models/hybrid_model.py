import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class HybridTransformerAutoencoder(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        bottleneck: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(bottleneck, d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.input_proj(x)       
        h = self.pos(h)               
        h = self.encoder(h)          

        z = self.to_latent(h)         
        h2 = self.from_latent(z)    

        y = self.output_proj(h2)   
        return y
