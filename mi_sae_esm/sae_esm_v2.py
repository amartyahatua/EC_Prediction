"""
SAE Training Pipeline for ESM-2 Protein Language Model Interpretability (v2)
=============================================================================
Dataset: SwissProt (EC-annotated enzymes)
Model: ESM-2 (8M) — ALL layers (1-6), dim=320

Improvements over v1:
  - TopK SAE variant to eliminate dead features
  - L1 coefficient sweep support
  - Loss recovered metric (downstream probe)
  - Hierarchical EC specificity analysis (multi-threshold + per-level)
  - Specificity threshold curves
  - Cross-layer and cross-EC comparison plots

Usage examples:
  # Full pipeline (all layers, all EC levels)
  python sae_pipeline_v2.py --layers 1 2 3 4 5 6

  # Quick loss recovered for key layers
  python sae_pipeline_v2.py --layers 2 5 --dict_ratios 8 16 --compute_loss_recovered --ec_levels level_1 level_4

  # L1 sweep at Layer 5
  python sae_pipeline_v2.py --layers 5 --dict_ratios 16 --l1_coeffs 0.3 0.6 1.0 2.0 --ec_levels level_1

  # TopK SAE experiment at Layer 5
  python sae_pipeline_v2.py --layers 5 --dict_ratios 16 --sae_type topk --topk_values 32 64 128 --ec_levels level_1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict

try:
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False
    print("WARNING: esm package not found. Install with: pip install fair-esm")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: datasets package not found. Install with: pip install datasets")


# ============================================================
# 1. SwissProt Data Loading & Representation Extraction
# ============================================================

def load_swissprot_ec_data(max_sequences=10000, max_length=512):
    """Load EC-annotated enzyme sequences from SwissProt."""
    assert HAS_DATASETS, "Please install datasets: pip install datasets"

    print("Loading SwissProt-EC-leaf dataset...")
    dataset = load_dataset("lightonai/SwissProt-EC-leaf", split="train")

    print("\nDEBUG — First 3 entries:")
    for i in range(min(3, len(dataset))):
        entry = dataset[i]
        print(f"  Entry {i}: keys={list(entry.keys())}")
        for k, v in entry.items():
            print(f"    {k}: {type(v).__name__} = {repr(v)[:120]}")
    print()

    sequences = []
    ec_labels = {'level_1': [], 'level_2': [], 'level_3': [], 'level_4': []}
    skipped_reasons = {'no_seq': 0, 'no_labels': 0, 'too_long': 0, 'bad_ec': 0, 'incomplete': 0}

    for i, entry in enumerate(tqdm(dataset, desc="Processing SwissProt entries")):
        if i >= max_sequences:
            break

        seq = entry.get('seq', '')
        labels_str = entry.get('labels_str', [])

        if not seq:
            skipped_reasons['no_seq'] += 1; continue
        if not labels_str:
            skipped_reasons['no_labels'] += 1; continue
        if len(seq) > max_length:
            skipped_reasons['too_long'] += 1; continue

        if isinstance(labels_str, str):
            ec_str = labels_str
        elif isinstance(labels_str, list) and len(labels_str) > 0:
            ec_str = labels_str[0]
        else:
            skipped_reasons['no_labels'] += 1; continue

        ec = ec_str.replace('EC:', '').strip()
        ec_parts = ec.split('.')
        if len(ec_parts) < 4:
            skipped_reasons['bad_ec'] += 1; continue
        if '-' in ec:
            skipped_reasons['incomplete'] += 1; continue

        sequences.append(seq)
        ec_labels['level_1'].append(ec_parts[0])
        ec_labels['level_2'].append(f"{ec_parts[0]}.{ec_parts[1]}")
        ec_labels['level_3'].append(f"{ec_parts[0]}.{ec_parts[1]}.{ec_parts[2]}")
        ec_labels['level_4'].append(ec)

    print(f"\nSkipped reasons: {skipped_reasons}")

    for level in ec_labels:
        unique_labels = sorted(set(ec_labels[level]))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        ec_labels[level] = np.array([label_to_idx[l] for l in ec_labels[level]])
        print(f"  {level}: {len(unique_labels)} unique classes")

    print(f"Loaded {len(sequences)} EC-annotated sequences from SwissProt")
    return sequences, ec_labels


def extract_all_layers_and_cache(sequences, layers, batch_size=16, device='cpu', cache_dir='../artifacts'):
    """
    Extract representations from all specified layers in a single forward pass.
    Caches each layer individually to disk.
    """
    assert HAS_ESM, "Please install fair-esm: pip install fair-esm"
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading ESM-2 model (8M)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    hidden_dim = model.embed_dim
    print(f"  Model: {n_layers} layers, hidden_dim={hidden_dim}")

    # Filter valid sequences
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    valid_sequences, valid_orig_indices = [], []
    for idx, seq in enumerate(sequences):
        if all(c in valid_chars for c in seq.upper()):
            valid_sequences.append(seq)
            valid_orig_indices.append(idx)

    skipped = len(sequences) - len(valid_sequences)
    if skipped > 0:
        print(f"  Skipped {skipped} sequences with non-standard amino acids")
    print(f"  Processing {len(valid_sequences)} valid sequences across {len(layers)} layers")

    if len(valid_sequences) == 0:
        raise ValueError("No valid sequences found!")

    # Check cache
    layers_to_extract = []
    layer_reprs = {}
    cached_valid_indices = None

    for layer in layers:
        cache_path = os.path.join(cache_dir, f'swissprot_repr_layer{layer}.npz')
        if os.path.exists(cache_path):
            print(f"  Layer {layer}: loading from cache")
            data = np.load(cache_path, allow_pickle=True)
            layer_reprs[layer] = data['representations']
            cached_valid_indices = data['valid_indices'].tolist()
        else:
            layers_to_extract.append(layer)

    if not layers_to_extract:
        print("All layers loaded from cache.")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return layer_reprs, cached_valid_indices or valid_orig_indices

    print(f"  Extracting layers: {layers_to_extract}")

    layer_collections = {layer: [] for layer in layers_to_extract}
    final_valid_indices = []

    for i in tqdm(range(0, len(valid_sequences), batch_size), desc="Extracting all layers"):
        batch_seqs = valid_sequences[i:i + batch_size]
        batch_orig_indices = valid_orig_indices[i:i + batch_size]
        batch_data = [(f"seq_{i+j}", seq) for j, seq in enumerate(batch_seqs)]

        try:
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=layers_to_extract, return_contacts=False)

            for j in range(len(batch_seqs)):
                seq_len = len(batch_seqs[j])
                for layer in layers_to_extract:
                    token_repr = results["representations"][layer][j, 1:seq_len + 1, :]
                    layer_collections[layer].append(token_repr.mean(dim=0).cpu().numpy())
                final_valid_indices.append(batch_orig_indices[j])

        except Exception as e:
            print(f"  Warning: Batch {i} failed: {e}")
            for j, seq in enumerate(batch_seqs):
                try:
                    single_data = [(f"seq_{i+j}", seq)]
                    _, _, tokens = batch_converter(single_data)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        res = model(tokens, repr_layers=layers_to_extract, return_contacts=False)
                    for layer in layers_to_extract:
                        token_repr = res["representations"][layer][0, 1:len(seq) + 1, :]
                        layer_collections[layer].append(token_repr.mean(dim=0).cpu().numpy())
                    final_valid_indices.append(batch_orig_indices[j])
                except Exception as e2:
                    print(f"    Skipping seq_{i+j}: {e2}")

    # Deduplicate indices
    seen = set()
    deduped = []
    for idx in final_valid_indices:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    final_valid_indices = deduped

    # Cache each layer
    for layer in layers_to_extract:
        if len(layer_collections[layer]) == 0:
            raise ValueError(f"No representations extracted for layer {layer}!")
        layer_reprs[layer] = np.vstack(layer_collections[layer])
        cache_path = os.path.join(cache_dir, f'swissprot_repr_layer{layer}.npz')
        np.savez(cache_path, representations=layer_reprs[layer],
                 valid_indices=np.array(final_valid_indices))
        print(f"  Cached layer {layer}: {layer_reprs[layer].shape}")

    all_valid_indices = final_valid_indices if final_valid_indices else cached_valid_indices

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return layer_reprs, all_valid_indices


# ============================================================
# 2. SAE Models — Standard (ReLU + L1) and TopK
# ============================================================

class SparseAutoencoder(nn.Module):
    """Standard SAE with ReLU activation and L1 sparsity penalty."""

    def __init__(self, input_dim=320, hidden_dim=1280, l1_coeff=0.3):
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

    def __init__(self, input_dim=320, hidden_dim=5120, k=64, aux_coeff=0.1):
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


def create_sae(sae_type, input_dim, hidden_dim, l1_coeff=0.3, k=64, aux_coeff=0.1):
    """Factory function to create the appropriate SAE variant."""
    if sae_type == 'topk':
        return SparseAutoencoderTopK(input_dim=input_dim, hidden_dim=hidden_dim,
                                     k=k, aux_coeff=aux_coeff)
    else:
        return SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim,
                                 l1_coeff=l1_coeff)


# ============================================================
# 3. Dataset Class
# ============================================================

class RepresentationDataset(Dataset):
    def __init__(self, representations, labels=None):
        self.representations = torch.FloatTensor(representations)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        x = self.representations[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        return x, -1


# ============================================================
# 4. Training & Evaluation
# ============================================================

def train_sae(sae, train_loader, val_loader=None, num_epochs=50, lr=1e-3,
              device='cpu', early_stopping_patience=10):
    """Train SAE with early stopping. Works for both Standard and TopK variants."""
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        'train_total_loss': [], 'train_recon_loss': [], 'train_sparsity_loss': [],
        'train_l0_sparsity': [], 'train_active_features': [],
        'val_total_loss': [], 'val_recon_loss': [], 'val_l0_sparsity': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    sparsity_label = "Aux" if sae.sae_type == 'topk' else "L1"

    for epoch in range(num_epochs):
        sae.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            x, _ = batch
            x = x.to(device)

            reconstruction, features = sae(x)
            total_loss, recon_loss, sparsity_loss = sae.compute_loss(x, reconstruction, features)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                l0 = (features > 0).float().sum(dim=1).mean().item()
                active_features_total = (features > 0).any(dim=0).sum().item()

            epoch_metrics['total'] += total_loss.item()
            epoch_metrics['recon'] += recon_loss.item()
            epoch_metrics['sparsity'] += sparsity_loss.item()
            epoch_metrics['l0'] += l0
            epoch_metrics['active'] += active_features_total
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'L0': f'{l0:.1f}',
            })

        history['train_total_loss'].append(epoch_metrics['total'] / num_batches)
        history['train_recon_loss'].append(epoch_metrics['recon'] / num_batches)
        history['train_sparsity_loss'].append(epoch_metrics['sparsity'] / num_batches)
        history['train_l0_sparsity'].append(epoch_metrics['l0'] / num_batches)
        history['train_active_features'].append(epoch_metrics['active'] / num_batches)

        if val_loader is not None:
            val_metrics = evaluate_sae(sae, val_loader, device)
            history['val_total_loss'].append(val_metrics['total_loss'])
            history['val_recon_loss'].append(val_metrics['recon_loss'])
            history['val_l0_sparsity'].append(val_metrics['l0_sparsity'])

            scheduler.step(val_metrics['total_loss'])

            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_state = {k: v.clone() for k, v in sae.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Epoch {epoch + 1} — "
              f"Train Loss: {history['train_total_loss'][-1]:.4f}, "
              f"Recon: {history['train_recon_loss'][-1]:.6f}, "
              f"L0: {history['train_l0_sparsity'][-1]:.1f}/{sae.hidden_dim}, "
              f"{sparsity_label}: {history['train_sparsity_loss'][-1]:.6f}"
              + (f", Val Loss: {history['val_total_loss'][-1]:.4f}" if val_loader else ""))

    if best_state is not None:
        sae.load_state_dict(best_state)
        print(f"Restored best model (val loss: {best_val_loss:.4f})")

    return history


@torch.no_grad()
def evaluate_sae(sae, data_loader, device='cpu'):
    """Evaluate SAE on a dataset. Returns aggregated metrics."""
    sae.eval()
    metrics = defaultdict(float)
    num_batches = 0

    for batch in data_loader:
        x, _ = batch
        x = x.to(device)

        reconstruction, features = sae(x)
        total_loss, recon_loss, sparsity_loss = sae.compute_loss(x, reconstruction, features)
        l0 = (features > 0).float().sum(dim=1).mean().item()

        metrics['total_loss'] += total_loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['sparsity_loss'] += sparsity_loss.item()
        metrics['l0_sparsity'] += l0
        num_batches += 1

    return {k: v / num_batches for k, v in metrics.items()}


# ============================================================
# 5. Null / Random Baseline Tests
# ============================================================

def run_null_baselines(input_dim, dict_sizes, test_loader, device='cpu'):
    """Random SAE, mean, and PCA baselines for each dictionary size."""
    print("\n" + "=" * 60)
    print("Running Null Baselines")
    print("=" * 60)

    all_x = torch.cat([batch[0] for batch in test_loader], dim=0)
    data_mean = all_x.mean(dim=0)

    null_results = {}
    for hidden_dim in dict_sizes:
        print(f"\n--- Dictionary size: {hidden_dim} ({hidden_dim / input_dim:.0f}x) ---")

        # Random SAE
        random_sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, l1_coeff=0.0).to(device)
        random_sae.eval()
        random_metrics = evaluate_sae(random_sae, test_loader, device)
        print(f"  Random SAE — Recon: {random_metrics['recon_loss']:.6f}")

        # Mean baseline
        all_x_dev = all_x.to(device)
        mean_recon_loss = F.mse_loss(
            data_mean.unsqueeze(0).expand_as(all_x_dev).to(device), all_x_dev
        ).item()
        print(f"  Mean — Recon: {mean_recon_loss:.6f}")

        # PCA baseline
        n_components = min(hidden_dim, input_dim)
        centered = all_x - data_mean.unsqueeze(0)
        U, S, V = torch.svd(centered)
        pca_recon = centered @ V[:, :n_components] @ V[:, :n_components].T + data_mean.unsqueeze(0)
        pca_recon_loss = F.mse_loss(pca_recon, all_x).item()
        print(f"  PCA ({n_components} comp) — Recon: {pca_recon_loss:.6f}")

        null_results[hidden_dim] = {
            'random_sae_recon': random_metrics['recon_loss'],
            'random_sae_l0': random_metrics['l0_sparsity'],
            'mean_recon': mean_recon_loss,
            'pca_recon': pca_recon_loss,
        }

    return null_results


# ============================================================
# 6. Loss Recovered (Downstream Probe)
# ============================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_linear_probe(representations, labels, num_classes, device='cpu',
                       num_epochs=50, lr=1e-3, batch_size=256):
    """Train linear probe, return best validation accuracy."""
    dataset = RepresentationDataset(representations, labels)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    probe = LinearProbe(representations.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(num_epochs):
        probe.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(probe(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = probe(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        best_acc = max(best_acc, correct / total)

    return best_acc


# @torch.no_grad()
def compute_loss_recovered(sae, representations, labels, num_classes, device='cpu'):
    """
    Compute loss recovered:
      LR = (acc_reconstructed - acc_random) / (acc_original - acc_random)
    """
    sae.eval()
    sae_device = next(sae.parameters()).device

    # Get SAE reconstructions
    with torch.no_grad():
        x = torch.FloatTensor(representations).to(sae_device)
        reconstructed = []
        for i in range(0, len(x), 512):
            recon, _ = sae(x[i:i + 512])
            reconstructed.append(recon.cpu().numpy())
        reconstructed = np.vstack(reconstructed)

    # Random baseline
    random_reprs = representations.copy()
    np.random.shuffle(random_reprs)

    print("  Probe on original representations...")
    acc_original = train_linear_probe(representations, labels, num_classes, device)

    print("  Probe on SAE reconstructions...")
    acc_reconstructed = train_linear_probe(reconstructed, labels, num_classes, device)

    print("  Probe on random baseline...")
    acc_random = train_linear_probe(random_reprs, labels, num_classes, device)

    if acc_original - acc_random > 0.01:
        loss_recovered = (acc_reconstructed - acc_random) / (acc_original - acc_random)
    else:
        loss_recovered = 0.0

    print(f"  Acc original: {acc_original:.4f}, reconstructed: {acc_reconstructed:.4f}, "
          f"random: {acc_random:.4f} → LR: {loss_recovered:.4f}")

    return loss_recovered, acc_original, acc_reconstructed, acc_random


# ============================================================
# 7. Feature Analysis — Enhanced with Hierarchical Specificity
# ============================================================

@torch.no_grad()
def analyze_feature_activations(sae, data_loader, ec_labels_all_levels, test_idx, device='cpu'):
    """
    Enhanced feature analysis:
      - Standard specificity at multiple thresholds
      - Hierarchical EC specificity (coarsest level at which each feature is specific)
      - Per-feature activation stats

    Args:
        sae: trained SAE model
        data_loader: test data loader (labels should correspond to the primary EC level)
        ec_labels_all_levels: dict with 'level_1'...'level_4' arrays for ALL data
        test_idx: indices into ec_labels_all_levels for the test set

    Returns:
        feature_ec_stats: per-feature dict
        summary: aggregate stats including hierarchical breakdown
    """
    sae.eval()

    all_features = []
    all_primary_labels = []

    for batch in data_loader:
        x, labels = batch
        x = x.to(device)
        features = sae.encode(x)
        all_features.append(features.cpu().numpy())
        all_primary_labels.append(labels.numpy())

    all_features = np.vstack(all_features)
    all_primary_labels = np.concatenate(all_primary_labels)

    # Get all EC level labels for test set
    test_labels_by_level = {}
    for level in ['level_1', 'level_2', 'level_3', 'level_4']:
        test_labels_by_level[level] = ec_labels_all_levels[level][test_idx]

    # Basic activation stats
    active_mask = all_features > 0
    per_feature_activation_rate = active_mask.mean(axis=0)
    dead_features = (per_feature_activation_rate < 1e-4).sum()
    ultra_sparse = (per_feature_activation_rate < 0.01).sum()

    # Multi-threshold interpretability
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_counts = {t: 0 for t in thresholds}

    # Hierarchical specificity tracking
    hierarchical_counts = {'level_1': 0, 'level_2': 0, 'level_3': 0, 'level_4': 0, 'none': 0}

    feature_ec_stats = {}
    for feat_idx in range(sae.hidden_dim):
        feat_active = active_mask[:, feat_idx]
        if feat_active.sum() < 5:
            continue

        # Compute specificity at each EC level
        level_specificities = {}
        for level in ['level_1', 'level_2', 'level_3', 'level_4']:
            active_labels = test_labels_by_level[level][feat_active]
            ec_counts = {}
            for ec in active_labels:
                ec_counts[ec] = ec_counts.get(ec, 0) + 1
            total_active = len(active_labels)
            top_ec = sorted(ec_counts.items(), key=lambda x: x[1], reverse=True)
            specificity = top_ec[0][1] / total_active if top_ec else 0.0
            level_specificities[level] = {
                'specificity': specificity,
                'top_class': top_ec[0][0] if top_ec else None,
                'num_classes': len(ec_counts),
            }

        # Primary specificity (for the data_loader's label level)
        primary_spec = level_specificities.get('level_1', {}).get('specificity', 0.0)
        # Use the finest-grained specificity for the standard metric
        finest_spec = level_specificities.get('level_4', {}).get('specificity', 0.0)

        # Multi-threshold counting (using level_1 specificity)
        for t in thresholds:
            if primary_spec > t:
                threshold_counts[t] += 1

        # Hierarchical: find coarsest level where specificity > 0.5
        assigned_level = 'none'
        for level in ['level_4', 'level_3', 'level_2', 'level_1']:
            if level_specificities[level]['specificity'] > 0.5:
                assigned_level = level
        hierarchical_counts[assigned_level] += 1

        feature_ec_stats[feat_idx] = {
            'activation_rate': float(per_feature_activation_rate[feat_idx]),
            'mean_activation': float(all_features[feat_active, feat_idx].mean()),
            'level_specificities': {
                level: {
                    'specificity': float(v['specificity']),
                    'top_class': int(v['top_class']) if v['top_class'] is not None else None,
                    'num_classes': v['num_classes'],
                }
                for level, v in level_specificities.items()
            },
            'hierarchical_level': assigned_level,
        }

    summary = {
        'total_features': sae.hidden_dim,
        'dead_features': int(dead_features),
        'dead_feature_pct': float(dead_features / sae.hidden_dim * 100),
        'ultra_sparse_features': int(ultra_sparse),
        'mean_activation_rate': float(per_feature_activation_rate.mean()),
        'median_activation_rate': float(np.median(per_feature_activation_rate)),
        # Standard interpretability (>50% at level_1 for backward compatibility)
        'num_interpretable_features': threshold_counts[0.5],
        # Multi-threshold
        'threshold_counts': {str(t): c for t, c in threshold_counts.items()},
        # Hierarchical breakdown
        'hierarchical_counts': hierarchical_counts,
        'hierarchical_total_specific': sum(
            v for k, v in hierarchical_counts.items() if k != 'none'
        ),
    }

    return feature_ec_stats, summary


# ============================================================
# 8. Visualization
# ============================================================

def plot_training_curves(history, dict_size, layer, save_dir='../plots', config_label=''):
    """Plot training loss curves for a single SAE."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f'SAE Training — Layer {layer}, Dict {dict_size}'
    if config_label:
        title += f' ({config_label})'
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(history['train_total_loss'], label='Train')
    if history['val_total_loss']:
        axes[0, 0].plot(history['val_total_loss'], label='Val')
    axes[0, 0].set_title('Total Loss'); axes[0, 0].set_xlabel('Epoch'); axes[0, 0].legend()

    axes[0, 1].plot(history['train_recon_loss'], label='Train')
    if history['val_recon_loss']:
        axes[0, 1].plot(history['val_recon_loss'], label='Val')
    axes[0, 1].set_title('Reconstruction Loss (MSE)'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].legend()

    axes[1, 0].plot(history['train_l0_sparsity'], label='Train L0')
    if history['val_l0_sparsity']:
        axes[1, 0].plot(history['val_l0_sparsity'], label='Val L0')
    axes[1, 0].set_title('L0 Sparsity'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].legend()

    axes[1, 1].plot(history['train_sparsity_loss'])
    axes[1, 1].set_title('Sparsity/Aux Loss'); axes[1, 1].set_xlabel('Epoch')

    plt.tight_layout()
    fname = f'sae_training_layer{layer}_dict{dict_size}'
    if config_label:
        fname += f'_{config_label}'
    plt.savefig(os.path.join(save_dir, f'{fname}.png'), dpi=150)
    plt.close()


def plot_dict_size_comparison(all_results, null_results, layer, save_dir='../plots'):
    """Plot comparison across dictionary sizes for a single layer."""
    os.makedirs(save_dir, exist_ok=True)

    dict_sizes = sorted(all_results.keys())
    ratios = [d / 320 for d in dict_sizes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SAE Comparison — Layer {layer}', fontsize=14)

    # Reconstruction Error
    ax = axes[0, 0]
    ax.plot(ratios, [all_results[d]['final_metrics']['recon_loss'] for d in dict_sizes],
            'o-', label='Trained SAE', linewidth=2)
    ax.plot(ratios, [null_results[d]['random_sae_recon'] for d in dict_sizes],
            's--', label='Random SAE', alpha=0.7)
    ax.plot(ratios, [null_results[d]['mean_recon'] for d in dict_sizes],
            '^--', label='Mean', alpha=0.7)
    ax.plot(ratios, [null_results[d]['pca_recon'] for d in dict_sizes],
            'D--', label='PCA', alpha=0.7)
    ax.set_xlabel('Dict Ratio'); ax.set_ylabel('Recon MSE')
    ax.set_title('Reconstruction Error'); ax.legend(); ax.set_yscale('log')

    # L0
    ax = axes[0, 1]
    ax.bar(range(len(dict_sizes)),
           [all_results[d]['final_metrics']['l0_sparsity'] for d in dict_sizes],
           tick_label=[f'{r:.0f}x' for r in ratios])
    ax.set_xlabel('Dict Ratio'); ax.set_ylabel('L0'); ax.set_title('L0 Sparsity')

    # Loss Recovered
    ax = axes[1, 0]
    lr_vals = [all_results[d].get('loss_recovered') for d in dict_sizes]
    if all(v is not None for v in lr_vals):
        ax.bar(range(len(dict_sizes)), lr_vals,
               tick_label=[f'{r:.0f}x' for r in ratios], color='green', alpha=0.7)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Perfect')
        ax.set_xlabel('Dict Ratio'); ax.set_ylabel('Loss Recovered')
        ax.set_title('Loss Recovered'); ax.legend()
    else:
        ax.text(0.5, 0.5, 'Not computed', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    # Dead Features
    ax = axes[1, 1]
    ax.bar(range(len(dict_sizes)),
           [all_results[d]['feature_summary']['dead_feature_pct'] for d in dict_sizes],
           tick_label=[f'{r:.0f}x' for r in ratios], color='orange', alpha=0.7)
    ax.set_xlabel('Dict Ratio'); ax.set_ylabel('Dead %'); ax.set_title('Dead Features')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sae_dict_comparison_layer{layer}.png'), dpi=150)
    plt.close()


def plot_cross_layer_comparison(all_layer_results, layers, save_dir='../plots'):
    """Plot SAE metrics across all layers."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAE Performance Across ESM-2 Layers', fontsize=14)

    all_dict_sizes = sorted(set(
        d for layer in layers if layer in all_layer_results
        for d in all_layer_results[layer].keys()
    ))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_dict_sizes)))

    for metric_idx, (ax, metric_key, ylabel, title, use_log) in enumerate([
        (axes[0, 0], 'final_metrics.recon_loss', 'Recon MSE', 'Reconstruction Error', True),
        (axes[0, 1], 'final_metrics.l0_sparsity', 'L0', 'L0 Sparsity', False),
        (axes[1, 0], 'feature_summary.dead_feature_pct', 'Dead %', 'Dead Features', False),
        (axes[1, 1], 'feature_summary.num_interpretable_features', 'Count', 'Interpretable Features', False),
    ]):
        for di, d in enumerate(all_dict_sizes):
            vals, valid_layers = [], []
            for layer in layers:
                if layer in all_layer_results and d in all_layer_results[layer]:
                    r = all_layer_results[layer][d]
                    keys = metric_key.split('.')
                    v = r
                    for k in keys:
                        v = v.get(k) if isinstance(v, dict) else None
                    if v is not None:
                        vals.append(v)
                        valid_layers.append(layer)
            if vals:
                ax.plot(valid_layers, vals, 'o-', label=f'{d // 320}x',
                        color=colors[di], linewidth=2)
        ax.set_xlabel('Layer'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(title='Dict Ratio')
        if use_log:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sae_cross_layer_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/sae_cross_layer_comparison.png")


def plot_specificity_threshold_curve(all_results, layer, save_dir='../plots'):
    """Plot interpretable feature count vs specificity threshold."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Specificity Threshold Curve — Layer {layer}', fontsize=14)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_results)))

    for di, (hidden_dim, r) in enumerate(sorted(all_results.items())):
        tc = r['feature_summary'].get('threshold_counts', {})
        if tc:
            thresholds = sorted([float(t) for t in tc.keys()])
            counts = [tc[str(t)] for t in thresholds]
            ax.plot(thresholds, counts, 'o-', label=f'{hidden_dim // 320}x ({hidden_dim})',
                    color=colors[di], linewidth=2)

    ax.set_xlabel('Specificity Threshold')
    ax.set_ylabel('Interpretable Features')
    ax.legend(title='Dict Size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'specificity_threshold_layer{layer}.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/specificity_threshold_layer{layer}.png")


def plot_hierarchical_breakdown(all_results, layer, save_dir='../plots'):
    """Plot hierarchical EC specificity breakdown as stacked bar chart."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Hierarchical EC Specificity — Layer {layer}', fontsize=14)

    dict_sizes = sorted(all_results.keys())
    labels = [f'{d // 320}x' for d in dict_sizes]
    levels = ['level_1', 'level_2', 'level_3', 'level_4']
    level_labels = ['EC1', 'EC2', 'EC3', 'EC4']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    bottom = np.zeros(len(dict_sizes))
    for li, level in enumerate(levels):
        counts = []
        for d in dict_sizes:
            hc = all_results[d]['feature_summary'].get('hierarchical_counts', {})
            counts.append(hc.get(level, 0))
        ax.bar(range(len(dict_sizes)), counts, bottom=bottom, label=level_labels[li],
               color=colors[li], alpha=0.8)
        bottom += np.array(counts)

    ax.set_xticks(range(len(dict_sizes)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Dictionary Size Ratio')
    ax.set_ylabel('Features')
    ax.legend(title='Specific at Level')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'hierarchical_breakdown_layer{layer}.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/hierarchical_breakdown_layer{layer}.png")


def plot_l1_sweep(sweep_results, layer, hidden_dim, save_dir='../plots'):
    """Plot L1 coefficient sweep results."""
    os.makedirs(save_dir, exist_ok=True)

    l1_coeffs = sorted(sweep_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'L1 Coefficient Sweep — Layer {layer}, Dict {hidden_dim}', fontsize=14)

    recon = [sweep_results[c]['final_metrics']['recon_loss'] for c in l1_coeffs]
    l0 = [sweep_results[c]['final_metrics']['l0_sparsity'] for c in l1_coeffs]
    dead = [sweep_results[c]['feature_summary']['dead_feature_pct'] for c in l1_coeffs]
    interp = [sweep_results[c]['feature_summary']['num_interpretable_features'] for c in l1_coeffs]

    axes[0, 0].plot(l1_coeffs, recon, 'o-', linewidth=2, color='#2196F3')
    axes[0, 0].set_xlabel('L1 Coefficient'); axes[0, 0].set_ylabel('Recon MSE')
    axes[0, 0].set_title('Reconstruction Error'); axes[0, 0].set_xscale('log')

    axes[0, 1].plot(l1_coeffs, l0, 'o-', linewidth=2, color='#4CAF50')
    axes[0, 1].set_xlabel('L1 Coefficient'); axes[0, 1].set_ylabel('L0')
    axes[0, 1].set_title('L0 Sparsity'); axes[0, 1].set_xscale('log')

    axes[1, 0].plot(l1_coeffs, interp, 'o-', linewidth=2, color='#FF9800')
    axes[1, 0].set_xlabel('L1 Coefficient'); axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Interpretable Features (>50%)'); axes[1, 0].set_xscale('log')

    axes[1, 1].plot(l1_coeffs, dead, 'o-', linewidth=2, color='#F44336')
    axes[1, 1].set_xlabel('L1 Coefficient'); axes[1, 1].set_ylabel('Dead %')
    axes[1, 1].set_title('Dead Features'); axes[1, 1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'l1_sweep_layer{layer}_dict{hidden_dim}.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/l1_sweep_layer{layer}_dict{hidden_dim}.png")


# ============================================================
# 9. Main Pipeline
# ============================================================

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"SAE type: {args.sae_type}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # ---- Step 1: Load data ----
    print("\n" + "=" * 60)
    print("Step 1: Loading Data & Extracting Representations")
    print("=" * 60)

    ec_cache = os.path.join(args.save_dir, 'swissprot_ec_labels.npz')
    seq_cache = os.path.join(args.save_dir, 'swissprot_sequences.json')

    if os.path.exists(ec_cache) and os.path.exists(seq_cache) and not args.force_extract:
        print("Loading cached sequences and labels...")
        data = np.load(ec_cache, allow_pickle=True)
        ec_labels = data['ec_labels'].item()
        with open(seq_cache, 'r') as f:
            sequences = json.load(f)
        print(f"Loaded {len(sequences)} sequences")
    else:
        sequences, ec_labels = load_swissprot_ec_data(
            max_sequences=args.max_sequences, max_length=args.max_length)
        np.savez(ec_cache, ec_labels=ec_labels)
        with open(seq_cache, 'w') as f:
            json.dump(sequences, f)

    layers = args.layers
    layer_reprs, valid_indices = extract_all_layers_and_cache(
        sequences, layers=layers, batch_size=args.esm_batch_size,
        device=device, cache_dir=args.save_dir)

    # Align labels
    for level in ec_labels:
        ec_labels[level] = ec_labels[level][valid_indices]

    input_dim = layer_reprs[layers[0]].shape[1]
    n_total = layer_reprs[layers[0]].shape[0]
    print(f"Input dim: {input_dim}, Sequences: {n_total}")

    # Split
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # Determine configurations
    dict_sizes = [int(r * input_dim) for r in args.dict_ratios]
    l1_coeffs = args.l1_coeffs
    topk_values = args.topk_values

    # ---- Step 2: Process each layer ----
    all_layer_results = {}

    for target_layer in layers:
        print(f"\n{'#' * 70}")
        print(f"LAYER {target_layer}")
        print(f"{'#' * 70}")

        representations = layer_reprs[target_layer]
        layer_save_dir = os.path.join(args.save_dir, f'layer{target_layer}')
        layer_plot_dir = os.path.join(args.plot_dir, f'layer{target_layer}')
        os.makedirs(layer_save_dir, exist_ok=True)
        os.makedirs(layer_plot_dir, exist_ok=True)

        for ec_level in args.ec_levels:
            print(f"\n{'=' * 60}")
            print(f"Layer {target_layer} — {ec_level}")
            print(f"{'=' * 60}")

            labels = ec_labels[ec_level]
            num_classes = len(np.unique(labels))
            print(f"Classes: {num_classes}")

            train_dataset = RepresentationDataset(representations[train_idx], labels[train_idx])
            val_dataset = RepresentationDataset(representations[val_idx], labels[val_idx])
            test_dataset = RepresentationDataset(representations[test_idx], labels[test_idx])

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     num_workers=args.num_workers)

            # Null baselines
            null_results = run_null_baselines(input_dim, dict_sizes, test_loader, device)

            all_results = {}

            for hidden_dim in dict_sizes:
                ratio = hidden_dim / input_dim

                # Determine what to sweep
                if args.sae_type == 'topk':
                    sweep_configs = [{'k': k} for k in topk_values]
                else:
                    sweep_configs = [{'l1': c} for c in l1_coeffs]

                # Track L1 sweep results for plotting
                l1_sweep_results = {} if args.sae_type == 'standard' and len(l1_coeffs) > 1 else None

                for config in sweep_configs:
                    if args.sae_type == 'topk':
                        k = config['k']
                        config_name = f"layer{target_layer}_dict{hidden_dim}_topk{k}"
                        config_label = f"TopK k={k}"
                        sae = create_sae('topk', input_dim, hidden_dim, k=k,
                                         aux_coeff=args.aux_coeff)
                    else:
                        l1_coeff = config['l1']
                        config_name = f"layer{target_layer}_dict{hidden_dim}_l1{l1_coeff}"
                        config_label = f"L1={l1_coeff}"
                        sae = create_sae('standard', input_dim, hidden_dim, l1_coeff=l1_coeff)

                    print(f"\n{'=' * 50}")
                    print(f"Training: {config_name}")
                    print(f"Params: {sum(p.numel() for p in sae.parameters()):,}")
                    print(f"{'=' * 50}")

                    history = train_sae(
                        sae=sae, train_loader=train_loader, val_loader=val_loader,
                        num_epochs=args.num_epochs, lr=args.learning_rate,
                        device=device, early_stopping_patience=args.patience,
                    )

                    test_metrics = evaluate_sae(sae, test_loader, device)
                    print(f"\nTest — Recon: {test_metrics['recon_loss']:.6f}, "
                          f"L0: {test_metrics['l0_sparsity']:.1f}")

                    # Enhanced feature analysis
                    feature_stats, feature_summary = analyze_feature_activations(
                        sae, test_loader, ec_labels, test_idx, device
                    )
                    print(f"  Dead: {feature_summary['dead_features']} "
                          f"({feature_summary['dead_feature_pct']:.1f}%)")
                    print(f"  Interpretable (>50%): {feature_summary['num_interpretable_features']}")
                    print(f"  Threshold curve: {feature_summary.get('threshold_counts', {})}")
                    print(f"  Hierarchical: {feature_summary.get('hierarchical_counts', {})}")

                    # Loss recovered
                    loss_recovered = None
                    loss_recovered_detail = None
                    if args.compute_loss_recovered:
                        print(f"\nComputing loss recovered...")
                        lr, acc_orig, acc_recon, acc_rand = compute_loss_recovered(
                            sae, representations[test_idx], labels[test_idx],
                            num_classes, device
                        )
                        loss_recovered = lr
                        loss_recovered_detail = {
                            'loss_recovered': lr,
                            'acc_original': acc_orig,
                            'acc_reconstructed': acc_recon,
                            'acc_random': acc_rand,
                        }

                    result_entry = {
                        'config': {
                            'layer': target_layer, 'hidden_dim': hidden_dim,
                            'ratio': ratio, 'input_dim': input_dim,
                            'sae_type': args.sae_type,
                            **config,
                        },
                        'final_metrics': test_metrics,
                        'feature_summary': feature_summary,
                        'loss_recovered': loss_recovered,
                        'loss_recovered_detail': loss_recovered_detail,
                        'history': history,
                    }

                    # Store: for standard SAE with single L1, key by dict size
                    # For sweeps, store all but use last as primary
                    all_results[hidden_dim] = result_entry

                    if l1_sweep_results is not None:
                        l1_sweep_results[l1_coeff] = result_entry

                    # Save model
                    model_path = os.path.join(layer_save_dir,
                                              f'sae_{ec_level}_{config_name}.pt')
                    torch.save({
                        'model_state_dict': sae.state_dict(),
                        'config': result_entry['config'],
                        'test_metrics': test_metrics,
                        'feature_summary': feature_summary,
                        'loss_recovered_detail': loss_recovered_detail,
                    }, model_path)
                    print(f"Saved: {model_path}")

                    plot_training_curves(history, hidden_dim, target_layer,
                                        layer_plot_dir, config_label)

                # L1 sweep plot
                if l1_sweep_results is not None and len(l1_sweep_results) > 1:
                    plot_l1_sweep(l1_sweep_results, target_layer, hidden_dim, layer_plot_dir)

            # Store for cross-layer
            all_layer_results[target_layer] = {
                d: {k: v for k, v in r.items() if k != 'history'}
                for d, r in all_results.items()
            }

            # Per-layer plots
            plot_dict_size_comparison(all_results, null_results, target_layer, layer_plot_dir)
            plot_specificity_threshold_curve(all_results, target_layer, layer_plot_dir)
            plot_hierarchical_breakdown(all_results, target_layer, layer_plot_dir)

            # ---- Summary table ----
            print("\n" + "=" * 100)
            print(f"SUMMARY — Layer {target_layer}, {ec_level}")
            print("=" * 100)
            print(f"{'Dict':>8} {'Ratio':>6} {'Type':>8} {'Recon MSE':>12} {'L0':>8} "
                  f"{'Dead%':>7} {'Interp':>7} {'Hier':>6} {'LR':>8}")
            print("-" * 100)

            for d in sorted(all_results.keys()):
                r = all_results[d]
                lr_str = f"{r['loss_recovered']:.3f}" if r['loss_recovered'] is not None else "—"
                hier_total = r['feature_summary'].get('hierarchical_total_specific', '—')
                stype = r['config'].get('sae_type', 'std')[:4]
                print(f"{d:>8} {r['config']['ratio']:>5.0f}x {stype:>8} "
                      f"{r['final_metrics']['recon_loss']:>12.6f} "
                      f"{r['final_metrics']['l0_sparsity']:>8.1f} "
                      f"{r['feature_summary']['dead_feature_pct']:>6.1f}% "
                      f"{r['feature_summary']['num_interpretable_features']:>7} "
                      f"{hier_total:>6} {lr_str:>8}")

            # Null baselines
            print(f"\nNull Baselines:")
            for d in sorted(null_results.keys()):
                n = null_results[d]
                print(f"  {d}: Random={n['random_sae_recon']:.6f}, "
                      f"Mean={n['mean_recon']:.6f}, PCA={n['pca_recon']:.2e}")

            # Save results JSON
            results_path = os.path.join(
                layer_save_dir, f'results_layer{target_layer}_{ec_level}_{timestamp}.json')
            save_results = {}
            for d in all_results:
                r = all_results[d]
                save_results[str(d)] = {
                    'config': r['config'],
                    'test_metrics': r['final_metrics'],
                    'feature_summary': r['feature_summary'],
                    'loss_recovered': r['loss_recovered'],
                    'loss_recovered_detail': r.get('loss_recovered_detail'),
                }
            save_results['null_baselines'] = {str(k): v for k, v in null_results.items()}

            with open(results_path, 'w') as f:
                json.dump(save_results, f, indent=2)
            print(f"Saved: {results_path}")

    # ---- Cross-layer comparison ----
    if len(layers) > 1:
        print("\n" + "#" * 70)
        print("CROSS-LAYER COMPARISON")
        print("#" * 70)
        plot_cross_layer_comparison(all_layer_results, layers, args.plot_dir)

        cross_layer_path = os.path.join(args.save_dir, f'cross_layer_results_{timestamp}.json')
        cross_layer_save = {}
        for layer in all_layer_results:
            cross_layer_save[str(layer)] = {}
            for d in all_layer_results[layer]:
                r = all_layer_results[layer][d]
                cross_layer_save[str(layer)][str(d)] = {
                    'config': r['config'],
                    'test_metrics': r['final_metrics'],
                    'feature_summary': r['feature_summary'],
                    'loss_recovered': r.get('loss_recovered'),
                    'loss_recovered_detail': r.get('loss_recovered_detail'),
                }

        with open(cross_layer_path, 'w') as f:
            json.dump(cross_layer_save, f, indent=2)
        print(f"Saved: {cross_layer_path}")

        # Cross-layer summary
        print("\n" + "=" * 110)
        print("CROSS-LAYER SUMMARY (best dict size per layer)")
        print("=" * 110)
        print(f"{'Layer':>6} {'Dict':>8} {'Ratio':>6} {'Type':>6} {'Recon':>12} {'L0':>8} "
              f"{'Dead%':>7} {'Interp':>7} {'Hier':>6} {'LR':>8}")
        print("-" * 110)

        for layer in sorted(all_layer_results.keys()):
            best_d = min(all_layer_results[layer].keys(),
                         key=lambda d: all_layer_results[layer][d]['final_metrics']['recon_loss'])
            r = all_layer_results[layer][best_d]
            lr_str = f"{r['loss_recovered']:.3f}" if r.get('loss_recovered') is not None else "—"
            hier = r['feature_summary'].get('hierarchical_total_specific', '—')
            stype = r['config'].get('sae_type', 'std')[:4]
            print(f"{layer:>6} {best_d:>8} {r['config']['ratio']:>5.0f}x {stype:>6} "
                  f"{r['final_metrics']['recon_loss']:>12.6f} "
                  f"{r['final_metrics']['l0_sparsity']:>8.1f} "
                  f"{r['feature_summary']['dead_feature_pct']:>6.1f}% "
                  f"{r['feature_summary']['num_interpretable_features']:>7} "
                  f"{hier:>6} {lr_str:>8}")


# ============================================================
# 10. Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='SAE Training Pipeline for ESM-2 — v2 (TopK, L1 Sweep, Loss Recovered, Hierarchical)')

    # Data
    parser.add_argument('--max_sequences', type=int, default=10000)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6],
                        help='ESM-2 layers to process')
    parser.add_argument('--ec_levels', nargs='+',
                        default=['level_1', 'level_2', 'level_3', 'level_4'])

    # SAE type
    parser.add_argument('--sae_type', choices=['standard', 'topk'], default='standard',
                        help='SAE variant: standard (ReLU+L1) or topk')

    # Standard SAE config
    parser.add_argument('--dict_ratios', nargs='+', type=float, default=[1, 4, 8, 16])
    parser.add_argument('--l1_coeffs', nargs='+', type=float, default=[0.3],
                        help='L1 coefficients to sweep (standard SAE only)')

    # TopK SAE config
    parser.add_argument('--topk_values', nargs='+', type=int, default=[64],
                        help='K values for TopK SAE')
    parser.add_argument('--aux_coeff', type=float, default=0.1,
                        help='Auxiliary loss coefficient for TopK SAE')

    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--esm_batch_size', type=int, default=16)

    # Evaluation
    parser.add_argument('--compute_loss_recovered', action='store_true',
                        help='Compute loss recovered metric')

    # Paths
    parser.add_argument('--save_dir', type=str, default='../artifacts')
    parser.add_argument('--plot_dir', type=str, default='../plots')
    parser.add_argument('--force_extract', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)