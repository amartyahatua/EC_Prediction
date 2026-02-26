import os
import torch
import argparse
import numpy as np
from get_dataset import *
from get_model import *
from layer_information import *
from analyze.sae_integration import *
from comparison.feature_comparison import FeatureMethodComparison
from mi_sae_esm.sae_model import SparseAutoencoder, loss_fn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_TOKEN"] = ""

RANDOM_STATE_SEED = 1829873
np.random.seed(RANDOM_STATE_SEED)
torch.manual_seed(RANDOM_STATE_SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_interprot_sae(representations, layer_num, args):
    """
    Train TopK SAE using InterProt implementation.

    Args:
        representations: numpy array [n_samples, d_model]
        layer_num: which ESM layer (for logging)
        args: command line arguments

    Returns:
        trained SAE model
    """

    print(f"\n{'=' * 60}")
    print(f"TRAINING INTERPROT TOPK SAE - Layer {layer_num}")
    print(f"{'=' * 60}")
    print(f"Samples: {representations.shape[0]}")
    print(f"Input dim: {representations.shape[1]}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"k (sparsity): {args.k_sparsity}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")

    # Initialize SAE
    sae = SparseAutoencoder(
        d_model=representations.shape[1],
        d_hidden=args.hidden_dim,
        k=args.k_sparsity,
        auxk=args.auxk,
        batch_size=args.batch_size,
        dead_steps_threshold=args.dead_steps_threshold
    ).to(device)

    # Prepare data
    dataset = TensorDataset(torch.tensor(representations, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)

    # Training loop
    print("\nTraining SAE...")
    for epoch in range(args.num_epochs):
        sae.train()
        epoch_mse = 0
        epoch_auxk = 0
        epoch_dead = 0
        n_batches = 0

        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)

            # Forward pass (returns 3 values!)
            recons, auxk_recons, num_dead = sae(batch)

            # Compute loss
            mse_loss, auxk_loss = loss_fn(batch, recons, auxk_recons)
            total_loss = mse_loss + auxk_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            # CRITICAL: Normalize decoder gradient
            sae.norm_grad()

            optimizer.step()

            # CRITICAL: Normalize decoder weights
            sae.norm_weights()

            # Track metrics
            epoch_mse += mse_loss.item()
            epoch_auxk += auxk_loss.item()
            epoch_dead += num_dead
            n_batches += 1

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == args.num_epochs - 1:
            avg_mse = epoch_mse / n_batches
            avg_auxk = epoch_auxk / n_batches
            avg_dead = epoch_dead / n_batches
            print(f"Epoch {epoch:2d}/{args.num_epochs}: "
                  f"MSE={avg_mse:.4f}, AuxK={avg_auxk:.4f}, Dead={avg_dead:.1f}")

    # Evaluate dead features
    print("\nEvaluating final SAE statistics...")
    sae.eval()
    all_acts = []
    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)
            acts = sae.get_acts(batch)
            all_acts.append(acts.cpu())

    all_acts = torch.cat(all_acts, dim=0)

    # Calculate dead features (features that never activate > 0.01)
    active = (all_acts.abs() > 0.01).any(dim=0)
    dead_features = (~active).sum().item()
    active_features = active.sum().item()
    dead_pct = (dead_features / args.hidden_dim) * 100

    # Calculate average sparsity
    avg_active = (all_acts != 0).float().sum(dim=-1).mean().item()

    # Calculate reconstruction error
    with torch.no_grad():
        test_batch = torch.tensor(representations[:1000], dtype=torch.float32).to(device)
        test_recons = sae.forward_val(test_batch)
        final_mse = F.mse_loss(test_recons, test_batch).item()
        relative_error = final_mse / test_batch.var().item()

    print(f"\n{'=' * 60}")
    print(f"TOPK SAE FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Reconstruction MSE: {final_mse:.4f}")
    print(f"Relative error: {relative_error:.4f}")
    print(f"Active features: {active_features}/{args.hidden_dim} ({100 * active_features / args.hidden_dim:.1f}%)")
    print(f"Dead features: {dead_features}/{args.hidden_dim} ({dead_pct:.1f}%)")
    print(f"Average sparsity per sample: {avg_active:.1f} (target: {args.k_sparsity})")

    if dead_pct < 40:
        print("✓ GOOD: Dead features < 40%")
    elif dead_pct < 60:
        print("⚠ MARGINAL: Dead features 40-60%")
    else:
        print("✗ BAD: Dead features > 60%")

    return sae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EC Prediction Training Script with TopK SAE")

    # Dataset parameters
    parser.add_argument("--dataset_size", type=int, default=10000, help="Set the number of data points")
    parser.add_argument("--dataset_name", type=str, default="DanielHesslow/SwissProt-EC", help="Set dataset name")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Set the model name")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for SAE training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (1e-3 recommended for TopK)")

    # TopK SAE parameters (NEW!)
    parser.add_argument("--hidden_dim", type=int, default=4096, help="SAE hidden dimension (4096 recommended)")
    parser.add_argument("--k_sparsity", type=int, default=128, help="TopK sparsity (64 or 128 recommended)")
    parser.add_argument("--auxk", type=int, default=256, help="Auxiliary k for dead neuron revival")
    parser.add_argument("--dead_steps_threshold", type=int, default=2000, help="Steps before neuron is 'dead'")

    # Comparison parameters
    parser.add_argument("--run_comparison", type=bool, default=True, help="Run baseline comparison")
    parser.add_argument("--comparison_sample_efficiency", type=bool, default=True,
                        help="Include sample efficiency tests")

    # Legacy parameters (for compatibility, not used with TopK)
    parser.add_argument("--l1_coeff", type=float, default=0.0001, help="IGNORED - TopK doesn't use L1")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("EC PREDICTION WITH INTERPROT TOPK SAE")
    print("=" * 80)
    print(f"SAE Configuration:")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  TopK sparsity: {args.k_sparsity}")
    print(f"  Auxiliary k: {args.auxk}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")

    # Load data and model
    print("\n" + "=" * 80)
    print("LOADING DATA AND MODEL")
    print("=" * 80)
    train_data, test_data = get_dataset(args.dataset_name)
    model, tokenizer = get_plm(args.model_name)
    N_LAYERS = model.config.num_hidden_layers + 1
    N_LEVELS = 4

    # Extract layer representations and labels
    print("\n" + "=" * 80)
    print("EXTRACTING LAYER REPRESENTATIONS")
    print("=" * 80)
    all_reprs, all_labels, results = get_layer_label_information(args, model, N_LAYERS, tokenizer, train_data)

    # Find best layer for each EC level
    print("\n" + "=" * 80)
    print("FINDING BEST LAYERS FOR EC HIERARCHY")
    print("=" * 80)
    best_result = ec_hierarchy_all_levels(args, model, tokenizer, train_data, N_LAYERS)

    # Train SAE models
    print("\n" + "=" * 80)
    print("TRAINING TOPK SPARSE AUTOENCODERS")
    print("=" * 80)

    sae_models = {}

    # Train SAE for each level's best layer
    for level_idx, level_name in enumerate(['level_1', 'level_2', 'level_3', 'level_4']):
        best_layer = best_result[level_name]['layer']
        best_acc = best_result[level_name]['accuracy']

        print(f"\n{'-' * 60}")
        print(f"Training SAE for {level_name.upper()}")
        print(f"Best layer: {best_layer} (accuracy: {best_acc:.4f})")
        print(f"{'-' * 60}")

        # Get representations for best layer
        layer_reprs = all_reprs[:, best_layer, :]

        # Train SAE
        sae_model = train_interprot_sae(layer_reprs, best_layer, args)

        # Save SAE
        os.makedirs('../artifacts', exist_ok=True)
        save_path = f'../artifacts/sae_layer_{best_layer}_{level_name}.pt'
        torch.save(sae_model.state_dict(), save_path)
        print(f"Saved SAE to: {save_path}")

        # Store in dictionary
        sae_models[level_name] = sae_model

    # Save layer information for each level
    print("\n" + "=" * 80)
    print("SAVING LAYER REPRESENTATIONS")
    print("=" * 80)
    os.makedirs('../artifacts', exist_ok=True)

    for i in range(N_LAYERS):
        layer_representations = all_reprs[:, i, :]
        for level_count in range(N_LEVELS):
            ec_labels_array = all_labels[f'level_{level_count + 1}']
            save_path = f'../artifacts/layer_{i + 1}_level_{level_count + 1}_representations.npz'
            save_representations_from_test_script(
                layer_data=layer_representations,
                ec_labels=ec_labels_array,
                save_path=save_path
            )
            print(f"  Saved: {save_path}")

    # Run baseline comparison
    if args.run_comparison:
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON")
        print("=" * 80)

        os.makedirs('../results', exist_ok=True)
        os.makedirs('../plots', exist_ok=True)

        # Run comparison for each EC level
        for level_idx, level_name in enumerate(['level_1', 'level_2', 'level_3', 'level_4']):
            print("\n" + "─" * 80)
            print(f"Comparing methods for {level_name.upper()}")
            print("─" * 80)

            # Get best layer for this level
            best_layer = best_result[level_name]['layer']
            best_acc = best_result[level_name]['accuracy']

            print(f"Best layer for {level_name}: Layer {best_layer} (accuracy: {best_acc:.4f})")

            # Get representations and labels
            layer_reprs = all_reprs[:, best_layer, :]
            ec_labels = all_labels[level_name]

            print(f"Data shape: {layer_reprs.shape}")
            print(f"Number of classes: {len(np.unique(ec_labels))}")

            # Initialize comparison
            comparison = FeatureMethodComparison(
                layer_representations=layer_reprs,
                ec_labels=ec_labels,
                n_components=args.hidden_dim,
                random_state=RANDOM_STATE_SEED
            )

            # Get trained SAE for this level
            sae_model = sae_models.get(level_name)

            if sae_model is None:
                print(f"WARNING: SAE not found for {level_name}, skipping SAE comparison")
                continue

            # Run comparison
            try:
                results_df = comparison.compare_all_methods(
                    sae_model=sae_model,
                    include_sample_efficiency=args.comparison_sample_efficiency
                )

                # Save results
                comparison.save_results(
                    results_df=results_df,
                    save_dir='../results',
                    level_name=level_name
                )

                print(f"\n✓ Comparison complete for {level_name}")

            except Exception as e:
                print(f"\n✗ ERROR during comparison for {level_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        print("\n" + "=" * 80)
        print("ALL COMPARISONS COMPLETE")
        print("=" * 80)
        print("Results saved to: ../results/")
        print("Plots saved to: ../results/")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)