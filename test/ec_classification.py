"""
EC Number Classification with Layer-wise Probing
Demonstrates mechanistic interpretability of protein language models for enzyme function prediction
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ============================================================================
# PART 1: Dataset Creation (using toy data - replace with real GENSEC data)
# ============================================================================

class ECDataset(Dataset):
    """Dataset for protein sequences with EC number labels"""

    def __init__(self, sequences: List[str], ec_labels: List[str], alphabet):
        """
        Args:
            sequences: List of protein sequences (amino acid strings)
            ec_labels: List of EC numbers (e.g., "3.4.21.1")
            alphabet: ESM alphabet for tokenization
        """
        self.sequences = sequences
        self.ec_labels = ec_labels
        self.alphabet = alphabet

        # Parse EC numbers into hierarchical levels
        self.ec_level1 = [ec.split('.')[0] for ec in ec_labels]  # Class
        self.ec_level2 = ['.'.join(ec.split('.')[:2]) for ec in ec_labels]  # Subclass
        self.ec_level3 = ['.'.join(ec.split('.')[:3]) for ec in ec_labels]  # Sub-subclass
        self.ec_level4 = ec_labels  # Full EC number

        # Create label encodings
        self.ec1_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level1)))}
        self.ec2_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level2)))}
        self.ec3_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level3)))}
        self.ec4_to_idx = {ec: idx for idx, ec in enumerate(sorted(set(self.ec_level4)))}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Get labels at all hierarchy levels
        labels = {
            'level1': self.ec1_to_idx[self.ec_level1[idx]],
            'level2': self.ec2_to_idx[self.ec_level2[idx]],
            'level3': self.ec3_to_idx[self.ec_level3[idx]],
            'level4': self.ec4_to_idx[self.ec_level4[idx]],
        }

        return sequence, labels

    def get_num_classes(self, level: int) -> int:
        """Get number of classes at each EC hierarchy level"""
        mapping = {
            1: len(self.ec1_to_idx),
            2: len(self.ec2_to_idx),
            3: len(self.ec3_to_idx),
            4: len(self.ec4_to_idx),
        }
        return mapping[level]


def create_toy_dataset():
    """Create a toy dataset for demonstration purposes"""
    # In reality, you'd load this from GENSEC or UniProt
    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    ]

    ec_labels = [
        "3.4.21.1",   # Chymotrypsin (serine protease)
        "3.4.21.5",   # Thrombin (serine protease)
        "1.1.1.1",    # Alcohol dehydrogenase (oxidoreductase)
    ]

    # Duplicate to create more samples
    sequences = sequences * 50
    ec_labels = ec_labels * 50

    return sequences, ec_labels


# ============================================================================
# PART 2: Extract Layer-wise Representations
# ============================================================================

class LayerActivationExtractor:
    """Extract activations from all layers of ESM model"""

    def __init__(self, model, layer_count):
        self.model = model
        self.layer_count = layer_count
        self.activations = {}
        self.hooks = []

    def get_activation(self, name):
        def hook(model, input, output):
            # Store the output activations
            self.activations[name] = output.detach()
        return hook

    def register_hooks(self):
        """Register forward hooks for all layers"""
        for i in range(self.layer_count):
            hook = self.model.layers[i].register_forward_hook(
                self.get_activation(f'layer_{i}')
            )
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_representations(self, batch_tokens):
        """
        Extract representations from all layers for a batch of sequences

        Returns:
            Dict mapping layer_idx -> tensor of shape (batch_size, seq_len, hidden_dim)
        """
        self.activations = {}

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=list(range(self.layer_count)))

        # Extract per-layer representations
        layer_representations = {}
        for layer_idx in range(self.layer_count):
            # Get sequence representations (average over sequence length, excluding special tokens)
            representations = results["representations"][layer_idx]
            # Mean pooling over sequence length (excluding BOS and EOS tokens)
            layer_representations[layer_idx] = representations[:, 1:-1, :].mean(dim=1)

        return layer_representations


# ============================================================================
# PART 3: Linear Probing for EC Classification
# ============================================================================

class LinearProbe(nn.Module):
    """Simple linear classifier for probing layer representations"""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(
    probe: LinearProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 50,
    lr: float = 0.001
) -> Tuple[float, float]:
    """Train a linear probe and return train and validation accuracy"""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

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

    # Evaluate
    probe.eval()
    train_preds, train_labels = [], []
    with torch.no_grad():
        for representations, labels in train_loader:
            representations = representations.to(device)
            outputs = probe(representations)
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.numpy())

    val_preds, val_labels = [], []
    with torch.no_grad():
        for representations, labels in val_loader:
            representations = representations.to(device)
            outputs = probe(representations)
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    return train_acc, val_acc


# ============================================================================
# PART 4: Main Analysis Pipeline
# ============================================================================

def analyze_layer_wise_ec_classification(
    sequences: List[str],
    ec_labels: List[str],
    model_name: str = "esm2_t12_35M_UR50D",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Main analysis: probe each layer for EC classification ability

    Returns:
        Dictionary with results for each EC hierarchy level
    """

    print(f"Loading model: {model_name}")
    print("This will download the model (~140MB) on first run...")

    # Use torch.hub.load which is the most reliable method
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)

    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    layer_count = model.num_layers
    hidden_dim = model.embed_dim

    print(f"Model loaded successfully!")
    print(f"Model has {layer_count} layers with hidden dimension {hidden_dim}")

    # Create dataset
    dataset = ECDataset(sequences, ec_labels, alphabet)

    # Split into train/val
    train_idx, val_idx = train_test_split(
        range(len(sequences)), test_size=0.2, random_state=42
    )

    print(f"\nDataset statistics:")
    print(f"Total sequences: {len(sequences)}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    for level in [1, 2, 3, 4]:
        print(f"EC Level {level}: {dataset.get_num_classes(level)} classes")

    # Extract representations for all sequences at all layers
    print("\nExtracting layer-wise representations...")
    extractor = LayerActivationExtractor(model, layer_count)

    all_layer_representations = {i: [] for i in range(layer_count)}
    all_labels = {f'level{i}': [] for i in [1, 2, 3, 4]}

    batch_size = 8
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in range(0, len(sequences), batch_size):
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{num_batches}...", end='\r')

        batch_sequences = sequences[i:i+batch_size]
        batch_ec_labels = ec_labels[i:i+batch_size]

        # Prepare batch
        batch_data = [(f"protein{j}", seq) for j, seq in enumerate(batch_sequences)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # Extract representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=list(range(layer_count)))

        for layer_idx in range(layer_count):
            representations = results["representations"][layer_idx]
            # Mean pool over sequence (excluding special tokens)
            pooled = representations[:, 1:-1, :].mean(dim=1).cpu()
            all_layer_representations[layer_idx].append(pooled)

        # Store labels
        for j, ec_label in enumerate(batch_ec_labels):
            ec_parts = ec_label.split('.')
            all_labels['level1'].append(dataset.ec1_to_idx[ec_parts[0]])
            all_labels['level2'].append(dataset.ec2_to_idx['.'.join(ec_parts[:2])])
            all_labels['level3'].append(dataset.ec3_to_idx['.'.join(ec_parts[:3])])
            all_labels['level4'].append(dataset.ec4_to_idx[ec_label])

    print("\nRepresentation extraction complete!")

    # Concatenate all representations
    for layer_idx in range(layer_count):
        all_layer_representations[layer_idx] = torch.cat(
            all_layer_representations[layer_idx], dim=0
        )

    # Convert labels to tensors
    for level_key in all_labels:
        all_labels[level_key] = torch.tensor(all_labels[level_key])

    # Probe each layer for each EC hierarchy level
    print("\nProbing each layer for EC classification...")
    results = {f'level{i}': {'layers': [], 'train_acc': [], 'val_acc': []}
               for i in [1, 2, 3, 4]}

    for level in [1, 2, 3, 4]:
        level_key = f'level{level}'
        num_classes = dataset.get_num_classes(level)

        print(f"\n{'='*60}")
        print(f"EC Level {level} ({num_classes} classes)")
        print(f"{'='*60}")

        for layer_idx in range(layer_count):
            # Get representations and labels for this layer
            layer_reps = all_layer_representations[layer_idx]
            labels = all_labels[level_key]

            # Split train/val
            train_reps = layer_reps[train_idx]
            val_reps = layer_reps[val_idx]
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]

            # Create dataloaders
            train_dataset = torch.utils.data.TensorDataset(train_reps, train_labels)
            val_dataset = torch.utils.data.TensorDataset(val_reps, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)

            # Train probe
            probe = LinearProbe(hidden_dim, num_classes).to(device)
            train_acc, val_acc = train_probe(probe, train_loader, val_loader, device, epochs=30)

            results[level_key]['layers'].append(layer_idx)
            results[level_key]['train_acc'].append(train_acc)
            results[level_key]['val_acc'].append(val_acc)

            print(f"Layer {layer_idx:2d}: Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}")

    return results


def plot_results(results: Dict, output_path: str = 'ec_layer_probing_results.png'):
    """Plot layer-wise probing results"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, level in enumerate([1, 2, 3, 4]):
        level_key = f'level{level}'
        ax = axes[idx]

        layers = results[level_key]['layers']
        train_acc = results[level_key]['train_acc']
        val_acc = results[level_key]['val_acc']

        ax.plot(layers, train_acc, 'o-', label='Train Accuracy', linewidth=2, markersize=6, alpha=0.7)
        ax.plot(layers, val_acc, 's-', label='Val Accuracy', linewidth=2, markersize=6)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'EC Level {level} Classification', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])

        # Highlight best layer
        best_idx = np.argmax(val_acc)
        ax.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.suptitle('Mechanistic Interpretability: EC Classification Across Layers',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("EC Classification with Layer-wise Probing")
    print("Mechanistic Interpretability of Protein Language Models")
    print("="*60)
    print()

    # Create toy dataset (replace with real GENSEC data)
    print("Creating toy dataset...")
    sequences, ec_labels = create_toy_dataset()
    print(f"Created {len(sequences)} sequences with {len(set(ec_labels))} unique EC numbers")
    print()

    # Run analysis
    results = analyze_layer_wise_ec_classification(sequences, ec_labels)

    # Plot results
    plot_results(results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Best Layer for Each EC Level")
    print("="*60)
    for level in [1, 2, 3, 4]:
        level_key = f'level{level}'
        val_accs = results[level_key]['val_acc']
        best_layer = np.argmax(val_accs)
        best_acc = val_accs[best_layer]
        print(f"EC Level {level}: Layer {best_layer:2d} (Val Acc = {best_acc:.3f})")

    print("\n" + "="*60)
    print("Analysis complete! Check ec_layer_probing_results.png")
    print("="*60)
    print()
    print("INTERPRETATION:")
    print("- If early layers have high accuracy: model learns functional info early")
    print("- If late layers have high accuracy: model needs deep processing")
    print("- If accuracy increases with hierarchy level: model learns coarse-to-fine")
    print("="*60)
