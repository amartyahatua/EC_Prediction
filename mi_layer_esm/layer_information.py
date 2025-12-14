import numpy as np
from collections import Counter
import torch
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
RANDOM_STATE_SEED = 42
def get_all_layer_representations(model, tokenizer, sequence):
    """Extract representations from all 33 layers"""

    # Tokenize
    inputs = tokenizer(sequence, return_tensors="pt", padding=True)

    # Get representations from all layers
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (num_layers, batch, seq_len, hidden_dim)
    # We want one vector per layer, so we'll use mean pooling over sequence length

    layer_representations = []
    for layer_hidden_state in outputs.hidden_states:  # 33 layers
        # Mean pool over sequence length
        pooled = layer_hidden_state.mean(dim=1)  # (batch, hidden_dim)
        layer_representations.append(pooled.squeeze().numpy())

    return np.array(layer_representations)  # Shape: (33, 1280)

def probe_layer(layer_num, all_reprs, labels, test_size=0.2, min_samples_per_class=2):
    """
    Probe a specific layer for EC prediction
    all_reprs: (num_proteins, num_layers, hidden_dim)
    """

    # Extract representations for this layer
    X = all_reprs[:, layer_num, :]  # (num_proteins, hidden_dim)
    y = labels

    # Filter out classes with too few samples
    label_counts = Counter(y)
    valid_labels = [label for label, count in label_counts.items() if count >= 2]

    # Keep only samples with valid labels
    mask = np.isin(y, valid_labels)
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"  Filtered: {len(X)} -> {len(X_filtered)} samples "
          f"({len(np.unique(y))} -> {len(np.unique(y_filtered))} classes)")

    # Check if we have enough data
    if len(X_filtered) < 10:
        print(f"  Warning: Only {len(X_filtered)} samples after filtering!")
        return 0.0, 0.0

    # Split - with stratify to maintain class balance
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered,
            test_size=test_size,
            random_state=42,
            stratify=y_filtered
        )
    except ValueError as e:
        # If stratify still fails, try without it
        print(f"  Warning: Stratified split failed, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered,
            test_size=test_size,
            random_state=42
        )

    # Train linear probe
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return acc, f1

def prepare_probing_data(model, tokenizer, dataset_split, num_samples=50, min_class_size=10):
    """
    Extract representations and labels for probing
    Filter out classes with fewer than min_class_size samples
    """

    # First pass: collect all labels to count class frequencies
    print("First pass: counting class frequencies...")
    temp_labels = {f'level_{i}': [] for i in range(1, 5)}

    for i in range(min(num_samples, len(dataset_split))):
        sample = dataset_split[i]

        # Check how many labels this sample has
        num_labels = len(sample['labels'])

        # Only use samples with all 4 EC levels
        if num_labels < 4:
            continue

        for level_idx in range(4):
            temp_labels[f'level_{level_idx + 1}'].append(sample['labels'][level_idx])

    # Find valid classes (those with enough samples)
    valid_classes = {}
    for level_name, labels in temp_labels.items():
        if len(labels) == 0:
            print(f"Warning: No labels found for {level_name}")
            valid_classes[level_name] = set()
            continue

        label_counts = Counter(labels)
        valid = {label for label, count in label_counts.items() if count >= min_class_size}
        valid_classes[level_name] = valid
        print(f"{level_name}: {len(valid)} classes with >= {min_class_size} samples "
              f"(total classes: {len(label_counts)})")

    # Second pass: extract representations only for valid samples
    print("\nSecond pass: extracting representations...")
    all_layer_reprs = []
    ec_labels = {f'level_{i}': [] for i in range(1, 5)}

    for i in range(min(num_samples, len(dataset_split))):
        sample = dataset_split[i]

        # Skip samples without all 4 EC levels
        if len(sample['labels']) < 4:
            continue

        # Check if this sample has valid labels for at least one level
        has_valid_label = False
        for level_idx in range(4):
            level_name = f'level_{level_idx + 1}'
            if sample['labels'][level_idx] in valid_classes[level_name]:
                has_valid_label = True
                break

        if not has_valid_label:
            continue

        # Get representations
        model.eval()
        reprs = get_all_layer_representations(model, tokenizer, sample['seq'])
        all_layer_reprs.append(reprs)

        # Get labels for each EC level
        for level_idx in range(4):
            level_name = f'level_{level_idx + 1}'
            ec_labels[level_name].append(sample['labels'][level_idx])

        # if len(all_layer_reprs) % 100 == 0:
        #     print(f"Processed {len(all_layer_reprs)} valid proteins")

    print(f"\nTotal proteins kept: {len(all_layer_reprs)}")

    return np.array(all_layer_reprs), {
        'level_1': np.array(ec_labels['level_1']),
        'level_2': np.array(ec_labels['level_2']),
        'level_3': np.array(ec_labels['level_3']),
        'level_4': np.array(ec_labels['level_4'])
    }


def run_layer_probing_experiment(model, tokenizer, dataset_split, N_LAYERS, ec_level, num_samples):

    print(f"Extracting representations for {num_samples} proteins...")
    all_reprs, all_labels = prepare_probing_data(model, tokenizer, dataset_split, num_samples)

    labels = all_labels[ec_level]

    print(f"\nProbing all layers for {ec_level}...")
    results = []

    for layer in range(N_LAYERS):
        acc, f1 = probe_layer(layer, all_reprs, labels)
        results.append({'layer': layer, 'accuracy': acc, 'f1': f1})
        print(f"Layer {layer}: Accuracy={acc:.3f}, F1={f1:.3f}")

    return all_reprs, all_labels, results

def get_layer_label_information(args, model, N_LAYERS, tokenizer, train_data):

    for level in ['level_1', 'level_2', 'level_3', 'level_4']:
        print(f"\n{'=' * 50}")
        print(f"Testing {level}")
        print(f"{'=' * 50}")
        all_reprs, all_labels, results = run_layer_probing_experiment(
            model,
            tokenizer,
            train_data,
            N_LAYERS,
            ec_level=level,
            num_samples=args.dataset_size
        )

    return all_reprs, all_labels, results


def ec_hierarchy_all_levels(args, model, tokenizer, train_data, N_LAYERS):
    """
    Find the best layer for predicting each level of EC hierarchy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Get representations and labels for all layers
    all_reprs, all_labels, _ = get_layer_label_information(args, model, N_LAYERS, tokenizer, train_data)

    best_result = {}
    all_layer_results = {}  # Store accuracies for all layers
    N_LEVELS = 4

    for level_idx in range(N_LEVELS):
        level_name = f'level_{level_idx + 1}'
        labels = all_labels[level_name]

        best_acc = 0
        best_layer = 0
        layer_accuracies = []

        print(f"\nEvaluating {level_name}:")

        for layer in range(N_LAYERS):
            # Get representations for this layer
            X = all_reprs[:, layer, :]

            # Try stratified split first, fall back to random split if it fails
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=RANDOM_STATE_SEED, stratify=labels
                )
            except ValueError:
                # Stratified split failed (some classes have too few samples)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=RANDOM_STATE_SEED
                )

            # Train probe
            probe = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE_SEED)
            probe.fit(X_train, y_train)

            # Evaluate
            y_pred = probe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            layer_accuracies.append(acc)

            print(f"  Layer {layer}: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        best_result[level_name] = {
            'layer': best_layer,
            'accuracy': best_acc
        }
        all_layer_results[level_name] = layer_accuracies

        print(f"✓ Best layer for {level_name}: Layer {best_layer} (accuracy: {best_acc:.4f})")

    # Generate plot
    plot_ec_hierarchy_results(best_result, all_layer_results, N_LAYERS)

    return best_result


def plot_ec_hierarchy_results(best_result, all_layer_results, N_LAYERS):
    """
    Plot accuracy across layers for each EC hierarchy level.

    Args:
        best_result: dict with best layer info for each level
        all_layer_results: dict with accuracy for each layer and level
                          Format: {'level_1': [acc_layer0, acc_layer1, ...], ...}
        N_LAYERS: number of layers
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    level_names = ['EC Level 1 (Main Class)', 'EC Level 2 (Subclass)',
                   'EC Level 3 (Sub-subclass)', 'EC Level 4 (Specific)']

    for idx, (level_key, level_title) in enumerate(zip(['level_1', 'level_2', 'level_3', 'level_4'],
                                                       level_names)):
        ax = axes[idx]

        # Get accuracies for all layers
        accuracies = all_layer_results[level_key]
        layers = list(range(N_LAYERS))

        # Plot line
        ax.plot(layers, accuracies, 'o-', linewidth=2, markersize=8)

        # Mark best layer
        best_layer = best_result[level_key]['layer']
        best_acc = best_result[level_key]['accuracy']

        ax.axvline(best_layer, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(best_layer + 0.1, best_acc, f'Layer {best_layer}\n{best_acc:.3f}',
                color='red', fontsize=10, verticalalignment='center')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(level_title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/ec_hierarchy_all_levels.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to: ../plots/ec_hierarchy_all_levels.png")