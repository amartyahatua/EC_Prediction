import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. SAE Model Definition
# ============================================================

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=5120, l1_coeff=0.6):
        """
        Sparse Autoencoder for protein representations

        Args:
            input_dim: Dimension of ESM-2 Layer 5 output (320 for 8M model)
            hidden_dim: Number of sparse features (typically 4x input_dim)
            l1_coeff: L1 sparsity penalty coefficient
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff

        # Encoder: maps representations to sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: reconstructs representations from sparse features
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize with small weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        """Encode input to sparse features"""
        return F.relu(self.encoder(x))

    def decode(self, features):
        """Decode sparse features back to input space"""
        return self.decoder(features)

    def forward(self, x):
        """Full forward pass"""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def compute_loss(self, x, reconstruction, features):
        """
        Compute total loss = reconstruction loss + sparsity penalty
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)

        # L1 sparsity penalty on features
        sparsity_loss = torch.mean(torch.abs(features))

        # Total loss
        total_loss = recon_loss + self.l1_coeff * sparsity_loss

        return total_loss, recon_loss, sparsity_loss


# ============================================================
# 2. Dataset Class for Layer 5 Representations
# ============================================================

class Layer5Dataset(Dataset):
    def __init__(self, representations, labels=None):
        """
        Dataset for Layer 5 representations

        Args:
            representations: numpy array of shape (num_samples, 320)
            labels: optional EC labels
        """
        self.representations = torch.FloatTensor(representations)
        self.labels = labels

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.representations[idx], self.labels[idx]
        return self.representations[idx]


# ============================================================
# 3. Training Function
# ============================================================

def train_sae(sae, train_loader, num_epochs=50, lr=1e-3, device='cpu'):
    """
    Train the Sparse Autoencoder

    Args:
        sae: SparseAutoencoder model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'

    Returns:
        training_history: dict with loss curves
    """
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    history = {
        'total_loss': [],
        'recon_loss': [],
        'sparsity_loss': [],
        'sparsity_level': []  # Track average number of active features
    }

    for epoch in range(num_epochs):
        sae.train()
        epoch_losses = {'total': 0, 'recon': 0, 'sparsity': 0}
        epoch_sparsity = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            x, _ = batch
            x = torch.tensor(x)
            x = x.to(device)

            # Forward pass
            reconstruction, features = sae(x)
            total_loss, recon_loss, sparsity_loss = sae.compute_loss(
                x, reconstruction, features
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['sparsity'] += sparsity_loss.item()

            # Calculate sparsity level (% of features active)
            active_features = (features > 0).float().sum(dim=1).mean()
            epoch_sparsity += active_features.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'active': f'{active_features.item():.1f}/{sae.hidden_dim}'
            })

        # Record epoch statistics
        history['total_loss'].append(epoch_losses['total'] / num_batches)
        history['recon_loss'].append(epoch_losses['recon'] / num_batches)
        history['sparsity_loss'].append(epoch_losses['sparsity'] / num_batches)
        history['sparsity_level'].append(epoch_sparsity / num_batches)

        print(f"Epoch {epoch + 1} - "
              f"Loss: {history['total_loss'][-1]:.4f}, "
              f"Recon: {history['recon_loss'][-1]:.4f}, "
              f"Active: {history['sparsity_level'][-1]:.1f}/{sae.hidden_dim}")

    return history


# ============================================================
# 4. Analysis Functions
# ============================================================

def analyze_feature_activations(sae, data_loader, ec_labels, device='cpu'):
    """
    Analyze which SAE features activate for which EC classes

    Returns:
        feature_ec_activations: dict mapping feature_idx -> EC class activations
    """
    sae.eval()

    # Collect all features and labels
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x, labels = batch
            # if isinstance(batch, tuple):
            #     x, labels = batch
            # else:
            #     x = batch
            #     labels = None

            x = x.to(device)
            features = sae.encode(x)

            all_features.append(features.cpu().numpy())
            if labels is not None:
                all_labels.extend(labels.numpy())

    all_features = np.vstack(all_features)  # (num_samples, hidden_dim)
    all_labels = np.array(all_labels)

    # For each feature, find which EC classes activate it most
    feature_ec_stats = {}

    for feature_idx in range(sae.hidden_dim):
        feature_activations = all_features[:, feature_idx]

        # Find samples where this feature is active (> threshold)
        active_mask = feature_activations > 0.1

        if active_mask.sum() > 0:
            active_ec_classes = all_labels[active_mask]
            ec_counts = {}
            for ec in active_ec_classes:
                ec_counts[ec] = ec_counts.get(ec, 0) + 1

            feature_ec_stats[feature_idx] = {
                'activation_rate': active_mask.mean(),
                'mean_activation': feature_activations[active_mask].mean(),
                'top_ec_classes': sorted(ec_counts.items(),
                                         key=lambda x: x[1],
                                         reverse=True)[:5]
            }

    return feature_ec_stats


# ============================================================
# 5. Main Training Pipeline
# ============================================================

def train_sae_pipeline(args, all_reprs, all_labels, N_LAYERS, best_result):
    """
    Complete pipeline for training SAE on Layer 5 representations
    """
    for nlevel in ['level_1', 'level_2', 'level_3', 'level_4']:
        best_layer = best_result[nlevel]['layer']
        best_acc = best_result[nlevel]['accuracy']

        print("=" * 60)
        print(f"SAE Training Pipeline for ESM-2 Layer {best_layer+1}")
        print("=" * 60)

        # Load your data from Phase 1
        # (You should have already extracted Layer 5 representations)
        print(f"\n1. Loading Layer {best_layer+1} representations...")


        layer_reprs = all_reprs[:, best_layer, :]
        ec_labels = all_labels[nlevel]  # For analysis

        # Create dataset and dataloader
        train_dataset = Layer5Dataset(layer_reprs, ec_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        print(f"Dataset size: {len(train_dataset)}")
        print(f"Input dimension: {layer_reprs.shape[1]}")

        # Initialize SAE
        print("\n2. Initializing SAE...")
        sae = SparseAutoencoder(
            input_dim=320,  # ESM-2 8M Layer 5 dimension
            hidden_dim=args.hidden_dim,  # 4x expansion
            l1_coeff=args.l1_coeff,
            # Tune this for desired sparsity
        )

        print(f"SAE Architecture:")
        print(f"  Input: {sae.input_dim}")
        print(f"  Hidden: {sae.hidden_dim}")
        print(f"  L1 coefficient: {sae.l1_coeff}")

        # Train SAE
        print("\n3. Training SAE...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        history = train_sae(
            sae=sae,
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            device=device
        )

        # Save model
        print("\n4. Saving model...")
        os.makedirs('../artifacts', exist_ok=True)
        torch.save(sae.state_dict(), f'../artifacts/sae_layer_{best_layer}.pt')
        print(f"Model saved to: sae_layer_{best_layer}.pt")

        # Analyze features
        print("\n5. Analyzing learned features...")
        feature_stats = analyze_feature_activations(
            sae, train_loader, ec_labels, device
        )

        print(f"\nFeature Statistics:")
        print(f"Total features: {sae.hidden_dim}")
        print(f"Active features (>1% activation): {len(feature_stats)}")

        # Print top 5 most interpretable features
        print("\nTop 5 most specific features:")
        sorted_features = sorted(
            feature_stats.items(),
            key=lambda x: len(x[1]['top_ec_classes']),
            reverse=False
        )[:5]

        for feat_idx, stats in sorted_features:
            print(f"\nFeature {feat_idx}:")
            print(f"  Activation rate: {stats['activation_rate']:.3f}")
            print(f"  Top EC classes:")
            for ec, count in stats['top_ec_classes'][:3]:
                print(f"    EC {ec}: {count} samples")

        #return sae, history, feature_stats


            # Plot training curves
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(history['total_loss'])
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')

            axes[1].plot(history['recon_loss'])
            axes[1].set_title('Reconstruction Loss')
            axes[1].set_xlabel('Epoch')

            axes[2].plot(history['sparsity_level'])
            axes[2].set_title('Average Active Features')
            axes[2].set_xlabel('Epoch')
            axes[2].axhline(sae.hidden_dim * 0.05, color='r', linestyle='--', label='5% target')
            axes[2].legend()
            os.makedirs('../plots', exist_ok=True)
            plt.tight_layout()
            plt.savefig(f'../plots/Layer_{best_layer+1}_{nlevel}_sae_training_curves.png')
            # plt.show()