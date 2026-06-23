import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, n_layers=1, dropout=0.0):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.layers(x)
