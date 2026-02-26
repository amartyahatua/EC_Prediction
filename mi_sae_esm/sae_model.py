import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    """Standard SAE with ReLU activation and L1 sparsity penalty."""

    def __init__(self, input_dim=320, hidden_dim=1280, l1_coeff=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff
        self.sae_type = 'standard'

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, features):
        return self.decoder(features)

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def compute_loss(self, x, reconstruction, features):
        recon_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = torch.mean(torch.abs(features))
        total_loss = recon_loss + self.l1_coeff * sparsity_loss
        return total_loss, recon_loss, sparsity_loss

    @torch.no_grad()
    def get_decoder_norms(self):
        return torch.norm(self.decoder.weight, dim=0)


class SparseAutoencoderTopK(nn.Module):
    """
    TopK SAE: forces exactly K features to be active per input.
    Eliminates dead features via auxiliary reconstruction loss on residuals.
    No L1 penalty needed — sparsity is enforced structurally.
    """

    def __init__(self, input_dim, hidden_dim, k, aux_coeff):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.aux_coeff = aux_coeff
        self.l1_coeff = 0.0  # For compatibility with evaluate_sae
        self.sae_type = 'topk'

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        pre_acts = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    def decode(self, features):
        return self.decoder(features)

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def compute_loss(self, x, reconstruction, features):
        recon_loss = F.mse_loss(reconstruction, x)

        # Auxiliary loss: reconstruct residual using ALL features
        # This gives gradient signal to dead features
        residual = (x - reconstruction).detach()
        aux_pre_acts = self.encoder(residual)
        aux_acts = F.relu(aux_pre_acts)
        aux_recon = self.decoder(aux_acts)
        aux_loss = F.mse_loss(aux_recon, residual)

        total_loss = recon_loss + self.aux_coeff * aux_loss
        return total_loss, recon_loss, aux_loss

    @torch.no_grad()
    def get_decoder_norms(self):
        return torch.norm(self.decoder.weight, dim=0)

