"""
Complete End-to-End EC Classification Pipeline
Loads real datasets and performs layer-wise mechanistic interpretability analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

# ============================================================================
# PART 1: Data Loading (supports multiple formats)
# ============================================================================

def load_gensec_dataset(gensec_path: str) -> Tuple[List[str], List[str]]:
    """Load GENSEC dataset"""
    print(f"Loading GENSEC dataset from {gensec_path}")
    
    if gensec_path.endswith('.csv'):
        df = pd.read_csv(gensec_path)
    elif gensec_path.endswith('.parquet'):
        df = pd.read_parquet(gensec_path)
    elif gensec_path.endswith('.tsv'):
        df = pd.read_csv(gensec_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format")
    
    # Detect column names
    seq_cols = ['sequence', 'Sequence', 'seq', 'protein_sequence']
    ec_cols = ['EC', 'ec', 'ec_number', 'label']
    
    sequence_col = next((col for col in seq_cols if col in df.columns), None)
    ec_col = next((col for col in ec_cols if col in df.columns), None)
    
    if not sequence_col or not ec_col:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find sequence and EC columns")
    
    print(f"Using columns: sequence='{sequence_col}', EC='{ec_col}'")
    
    df = df.dropna(subset=[sequence_col, ec_col])
    
    sequences = df[sequence_col].tolist()
    ec_labels = df[ec_col].astype(str).tolist()
    
    print(f"Loaded {len(sequences)} sequences")
    return sequences, ec_labels


def load_csv_dataset(
    csv_path: str,
    sequence_column: str = 'sequence',
    ec_column: str = 'EC'
) -> Tuple[List[str], List[str]]:
    """Load from generic CSV file"""
    print(f"Loading CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if sequence_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{sequence_column}' not found")
    if ec_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{ec_column}' not found")
    
    df = df.dropna(subset=[sequence_column, ec_column])
    
    sequences = df[sequence_column].tolist()
    ec_labels = df[ec_column].astype(str).tolist()
    
    print(f"Loaded {len(sequences)} sequences")
    return sequences, ec_labels


def filter_dataset(
    sequences: List[str],
    ec_labels: List[str],
    min_length: int = 50,
    max_length: int = 1000,
    min_samples_per_class: int = 10,
    max_samples_per_class: int = None
) -> Tuple[List[str], List[str]]:
    """Filter and balance dataset"""
    print("\nFiltering dataset...")
    print(f"Initial: {len(sequences)} sequences")
    
    df = pd.DataFrame({'sequence': sequences, 'ec': ec_labels})
    
    # Length filter
    df['seq_len'] = df['sequence'].apply(len)
    df = df[(df['seq_len'] >= min_length) & (df['seq_len'] <= max_length)]
    print(f"After length filter ({min_length}-{max_length}): {len(df)}")
    
    # Complete EC numbers only (X.X.X.X)
    df['ec_levels'] = df['ec'].apply(lambda x: len(str(x).split('.')))
    df = df[df['ec_levels'] == 4]
    print(f"After complete EC filter: {len(df)}")
    
    # Filter by sample count per class
    ec_counts = df['ec'].value_counts()
    valid_ecs = ec_counts[ec_counts >= min_samples_per_class].index
    df = df[df['ec'].isin(valid_ecs)]
    print(f"After min samples filter (>={min_samples_per_class}): {len(df)}")
    
    # Balance classes if max_samples_per_class is set
    if max_samples_per_class:
        df = df.groupby('ec').apply(
            lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
        ).reset_index(drop=True)
        print(f"After balancing (max {max_samples_per_class} per class): {len(df)}")
    
    print(f"\nFinal dataset:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Unique EC numbers: {df['ec'].nunique()}")
    print(f"  Sequences per EC: {len(df) / df['ec'].nunique():.1f}")
    
    # Show class distribution
    print(f"\nTop 10 most common EC numbers:")
    print(df['ec'].value_counts().head(10))
    
    return df['sequence'].tolist(), df['ec'].tolist()


# ============================================================================
# PART 2: Dataset and Model Classes
# ============================================================================

class ECDataset(Dataset):
    """Dataset for protein sequences with EC labels"""
    
    def __init__(self, sequences: List[str], ec_labels: List[str], alphabet):
        self.sequences = sequences
        self.ec_labels = ec_labels
        self.alphabet = alphabet
        
        # Parse EC hierarchy
        self.ec_level1 = [ec.split('.')[0] for ec in ec_labels]
        self.ec_level2 = ['.'.join(ec.split('.')[:2]) for ec in ec_labels]
        self.ec_level3 = ['.'.join(ec.split('.')[:3]) for ec in ec_labels]
        self.ec_level4 = ec_labels
        
        # Create encodings
        self.ec1_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level1)))}
        self.ec2_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level2)))}
        self.ec3_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level3)))}
        self.ec4_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level4)))}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = {
            'level1': self.ec1_to_idx[self.ec_level1[idx]],
            'level2': self.ec2_to_idx[self.ec_level2[idx]],
            'level3': self.ec3_to_idx[self.ec_level3[idx]],
            'level4': self.ec4_to_idx[self.ec_level4[idx]],
        }
        return sequence, labels
    
    def get_num_classes(self, level: int) -> int:
        return {
            1: len(self.ec1_to_idx),
            2: len(self.ec2_to_idx),
            3: len(self.ec3_to_idx),
            4: len(self.ec4_to_idx),
        }[level]


class LinearProbe(nn.Module):
    """Linear classifier for probing"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


# ============================================================================
# PART 3: Training and Evaluation
# ============================================================================

def train_probe(
    probe: LinearProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 50,
    lr: float = 0.001
) -> Tuple[float, float]:
    """Train linear probe"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    
    # Training
    probe.train()
    for epoch in range(epochs):
        for representations, labels in train_loader:
            representations = representations.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe(representations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    probe.eval()
    
    def evaluate(loader):
        preds, targets = [], []
        with torch.no_grad():
            for representations, labels in loader:
                representations = representations.to(device)
                outputs = probe(representations)
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                targets.extend(labels.numpy())
        return accuracy_score(targets, preds)
    
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    
    return train_acc, val_acc


# ============================================================================
# PART 4: Main Analysis Pipeline
# ============================================================================

def analyze_layer_wise_ec_classification(
    sequences: List[str],
    ec_labels: List[str],
    model_name: str = "esm2_t12_35M_UR50D",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8
):
    """Main analysis pipeline"""
    
    print(f"\n{'='*70}")
    print(f"Starting Layer-wise EC Classification Analysis")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"Loading ESM model: {model_name}")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    
    layer_count = model.num_layers
    hidden_dim = model.embed_dim
    print(f"Model: {layer_count} layers, {hidden_dim} hidden dim")
    
    # Create dataset
    dataset = ECDataset(sequences, ec_labels, alphabet)
    train_idx, val_idx = train_test_split(
        range(len(sequences)), test_size=0.2, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)} sequences")
    print(f"  Val: {len(val_idx)} sequences")
    for level in [1, 2, 3, 4]:
        print(f"  EC Level {level}: {dataset.get_num_classes(level)} classes")
    
    # Extract representations
    print(f"\nExtracting layer-wise representations...")
    all_layer_representations = {i: [] for i in range(layer_count)}
    all_labels = {f'level{i}': [] for i in [1, 2, 3, 4]}
    
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in range(0, len(sequences), batch_size):
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{num_batches}", end='\r')
        
        batch_sequences = sequences[i:i+batch_size]
        batch_ec_labels = ec_labels[i:i+batch_size]
        
        batch_data = [(f"p{j}", seq) for j, seq in enumerate(batch_sequences)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=list(range(layer_count)))
        
        for layer_idx in range(layer_count):
            representations = results["representations"][layer_idx]
            pooled = representations[:, 1:-1, :].mean(dim=1).cpu()
            all_layer_representations[layer_idx].append(pooled)
        
        for j, ec_label in enumerate(batch_ec_labels):
            ec_parts = ec_label.split('.')
            all_labels['level1'].append(dataset.ec1_to_idx[ec_parts[0]])
            all_labels['level2'].append(dataset.ec2_to_idx['.'.join(ec_parts[:2])])
            all_labels['level3'].append(dataset.ec3_to_idx['.'.join(ec_parts[:3])])
            all_labels['level4'].append(dataset.ec4_to_idx[ec_label])
    
    print("\n  Extraction complete!")
    
    # Concatenate
    for layer_idx in range(layer_count):
        all_layer_representations[layer_idx] = torch.cat(
            all_layer_representations[layer_idx], dim=0
        )
    for level_key in all_labels:
        all_labels[level_key] = torch.tensor(all_labels[level_key])
    
    # Probe each layer
    print(f"\nProbing layers...")
    results = {f'level{i}': {'layers': [], 'train_acc': [], 'val_acc': []} 
               for i in [1, 2, 3, 4]}
    
    for level in [1, 2, 3, 4]:
        level_key = f'level{level}'
        num_classes = dataset.get_num_classes(level)
        
        print(f"\nEC Level {level} ({num_classes} classes):")
        
        for layer_idx in range(layer_count):
            layer_reps = all_layer_representations[layer_idx]
            labels = all_labels[level_key]
            
            train_reps = layer_reps[train_idx]
            val_reps = layer_reps[val_idx]
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            
            train_dataset = torch.utils.data.TensorDataset(train_reps, train_labels)
            val_dataset = torch.utils.data.TensorDataset(val_reps, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            probe = LinearProbe(hidden_dim, num_classes).to(device)
            train_acc, val_acc = train_probe(probe, train_loader, val_loader, device, epochs=30)
            
            results[level_key]['layers'].append(layer_idx)
            results[level_key]['train_acc'].append(train_acc)
            results[level_key]['val_acc'].append(val_acc)
            
            print(f"  Layer {layer_idx:2d}: Train={train_acc:.3f}, Val={val_acc:.3f}")
    
    return results


def plot_results(results: Dict, output_path: str = 'ec_results.png'):
    """Plot results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, level in enumerate([1, 2, 3, 4]):
        level_key = f'level{level}'
        ax = axes[idx]
        
        layers = results[level_key]['layers']
        train_acc = results[level_key]['train_acc']
        val_acc = results[level_key]['val_acc']
        
        ax.plot(layers, train_acc, 'o-', label='Train', linewidth=2, markersize=6, alpha=0.7)
        ax.plot(layers, val_acc, 's-', label='Val', linewidth=2, markersize=6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'EC Level {level}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        best_idx = np.argmax(val_acc)
        ax.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Mechanistic Interpretability: EC Classification', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EC Classification Analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--type', type=str, default='csv', choices=['csv', 'gensec'],
                       help='Data type')
    parser.add_argument('--seq-col', type=str, default='sequence', help='Sequence column')
    parser.add_argument('--ec-col', type=str, default='EC', help='EC column')
    parser.add_argument('--model', type=str, default='esm2_t12_35M_UR50D', 
                       help='ESM model name')
    parser.add_argument('--output', type=str, default='ec_results.png', 
                       help='Output plot path')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Min samples per EC class')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples per EC class (for balancing)')
    
    args = parser.parse_args()
    
    # Load data
    if args.type == 'gensec':
        sequences, ec_labels = load_gensec_dataset(args.data)
    else:
        sequences, ec_labels = load_csv_dataset(args.data, args.seq_col, args.ec_col)
    
    # Filter
    sequences, ec_labels = filter_dataset(
        sequences, ec_labels,
        min_samples_per_class=args.min_samples,
        max_samples_per_class=args.max_samples
    )
    
    # Analyze
    results = analyze_layer_wise_ec_classification(sequences, ec_labels, args.model)
    
    # Plot
    plot_results(results, args.output)
    
    # Summary
    print(f"\n{'='*70}")
    print("BEST LAYERS FOR EACH EC LEVEL:")
    print(f"{'='*70}")
    for level in [1, 2, 3, 4]:
        level_key = f'level{level}'
        val_accs = results[level_key]['val_acc']
        best_layer = np.argmax(val_accs)
        best_acc = val_accs[best_layer]
        print(f"EC Level {level}: Layer {best_layer:2d} (Val Acc = {best_acc:.3f})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # If no arguments provided, show usage
    import sys
    if len(sys.argv) == 1:
        print(__doc__)
        print("\nUSAGE EXAMPLES:\n")
        print("1. With GENSEC dataset:")
        print("   python ec_pipeline.py --data gensec.csv --type gensec\n")
        print("2. With custom CSV:")
        print("   python ec_pipeline.py --data proteins.csv --seq-col sequence --ec-col EC\n")
        print("3. With specific model and balancing:")
        print("   python ec_pipeline.py --data data.csv --model esm2_t33_650M_UR50D --max-samples 100\n")
        print("Add --help for all options")
    else:
        main()
