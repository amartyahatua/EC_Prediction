import torch
import esm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import os

from sae_model import SparseAutoencoder, SparseAutoencoderTopK
from prepare_data import get_dataset


@torch.no_grad()
def evaluate_sae(sae, data_loader, device='cpu'):
    """Evaluate SAE on a dataset. Returns aggregated metrics."""
    sae.eval()
    metrics = defaultdict(float)
    num_batches = 0

    for batch in data_loader:
        x = batch
        x = x[0].to(device)

        reconstruction, features = sae(x)
        total_loss, recon_loss, sparsity_loss = sae.compute_loss(x, reconstruction, features)
        l0 = (features > 0).float().sum(dim=1).mean().item()

        metrics['total_loss'] += total_loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['sparsity_loss'] += sparsity_loss.item()
        metrics['l0_sparsity'] += l0
        num_batches += 1

    return {k: v / num_batches for k, v in metrics.items()}

def train_sae(args, sae, data_loader, device, hidden_dim_scle, topk):
    history = {
        'train_total_loss': [], 'train_recon_loss': [], 'train_sparsity_loss': [],
        'train_l0_sparsity': [], 'train_active_features': [],
        'val_total_loss': [], 'val_recon_loss': [], 'val_l0_sparsity': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    sparsity_label = "Aux"
    early_stopping_patience = 10

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(args.num_epochs):
        sae.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}')
        for data in pbar:
            data = data[0].to(device)
            reconstruction, features = sae(data)

            total_loss, recon_loss, sparsity_loss = sae.compute_loss(data, reconstruction, features)

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

        avg_loss = epoch_metrics['total'] / num_batches
        scheduler.step(avg_loss)

        history['train_total_loss'].append(epoch_metrics['total'] / num_batches)
        history['train_recon_loss'].append(epoch_metrics['recon'] / num_batches)
        history['train_sparsity_loss'].append(epoch_metrics['sparsity'] / num_batches)
        history['train_l0_sparsity'].append(epoch_metrics['l0'] / num_batches)
        history['train_active_features'].append(epoch_metrics['active'] / num_batches)
        print(f"Epoch {epoch + 1} — Avg Recon: {epoch_metrics['recon'] / num_batches:.6f}")

    save_path = os.path.join(args.save_dir, f"sae_dict{int(hidden_dim_scle)}_k{topk}.pt")
    torch.save(sae.state_dict(), save_path)
    print(f"Saved model to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='SAE Training Pipeline for ESM-2 — v2 (TopK, L1 Sweep, Loss Recovered, Hierarchical)')

    # Data
    parser.add_argument('--dataset_name', default="lightonai/SwissProt-EC-leaf")
    parser.add_argument('--max_sequences', type=int, default=10000)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6],
                        help='ESM-2 layers to process')
    parser.add_argument('--ec_levels', nargs='+',
                        default=['level_1', 'level_2', 'level_3', 'level_4'])

    # SAE type
    parser.add_argument('--sae_type', choices=['standard', 'topk'], default='topk',
                        help='SAE variant: standard (ReLU+L1) or topk')

    # Standard SAE config
    parser.add_argument('--input_dim', type=int, default=320)
    parser.add_argument('--hidden_dim', type=int, default=1280)

    parser.add_argument('--dict_ratios', nargs='+', type=float, default=[16])
    parser.add_argument('--l1_coeffs', nargs='+', type=float, default=[0.3],
                        help='L1 coefficients to sweep (standard SAE only)')

    # TopK SAE config
    parser.add_argument('--topk_values', nargs='+', type=int, default=[256],
                        help='K values for TopK SAE')
    parser.add_argument('--aux_coeff', type=float, default=0.1,
                        help='Auxiliary loss coefficient for TopK SAE')

    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
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

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    train_loader = get_dataset(args, model, alphabet)

    # ---- Step 2: Select model ----

    for hidden_dim_scle in args.dict_ratios:
        for topk in args.topk_values:
            print(f"\n--- Dictionary size: {hidden_dim_scle} and TOPK: {topk} ---")
            if args.sae_type == 'standard':
                sae = SparseAutoencoder(input_dim=args.input_dim, hidden_dim=args.input_dim*hidden_dim_scle, l1_coeff=0.3)
            else:
                sae = SparseAutoencoderTopK(input_dim=args.input_dim, hidden_dim=args.input_dim*hidden_dim_scle, k=topk, aux_coeff=args.aux_coeff)

            sae = sae.to(device)
            train_sae(args, sae, train_loader, device, hidden_dim_scle, topk)

if __name__ == '__main__':
    args = parse_args()
    main(args)