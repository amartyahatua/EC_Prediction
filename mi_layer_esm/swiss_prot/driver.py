"""
Step 3: Layer Selection — Which layer has the most biological information?

Usage:
    python layer_selection.py --model_type huggingface --model_name facebook/esm2_t6_8M_UR50D --dataset_size 5000
    python layer_selection.py --model_type fair-esm --model_name esm2_t6_8M_UR50D --dataset_size 5000
    python layer_selection.py --model_type fair-esm --model_name esm2_t33_650M_UR50D --dataset_size 5000
"""

import numpy as np
from collections import Counter
import torch
import os
import json
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import matplotlib.pyplot as plt

RANDOM_STATE_SEED = 42


# ══════════════════════════════════════════════════════════════
# REPRESENTATION EXTRACTION
# ══════════════════════════════════════════════════════════════

def get_all_layer_representations(model, tokenizer, sequence, model_type):
    """
    Extract mean-pooled representations from ALL layers for a single protein.
    Special tokens (BOS/EOS) are excluded from pooling.

    Returns:
        np.array of shape (n_layers, hidden_dim)
    """
    if model_type == "fair-esm":
        batch_converter = tokenizer.get_batch_converter()
        data = [("protein", sequence)]
        _, _, tokens = batch_converter(data)

        n_layers = model.num_layers
        all_layers = list(range(0, n_layers + 1))

        with torch.no_grad():
            results = model(tokens, repr_layers=all_layers, return_contacts=False)

        layer_reprs = []
        for layer in all_layers:
            residue_reprs = results["representations"][layer][0, 1:len(sequence) + 1, :]
            layer_reprs.append(residue_reprs.mean(dim=0).cpu().numpy())

        return np.array(layer_reprs)

    else:  # huggingface
        inputs = tokenizer(sequence, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        seq_len = len(sequence)
        layer_reprs = []
        for layer_hidden_state in outputs.hidden_states:
            residue_reprs = layer_hidden_state[0, 1:seq_len + 1, :]
            layer_reprs.append(residue_reprs.mean(dim=0).numpy())

        return np.array(layer_reprs)


# ══════════════════════════════════════════════════════════════
# DATA PREPARATION — SINGLE PASS
# ══════════════════════════════════════════════════════════════

def prepare_probing_data(model, tokenizer, dataset, num_samples, min_class_size, model_type):
    """
    Extract representations and EC labels in a single pass.

    Returns:
        all_reprs: np.array (n_proteins, n_layers, hidden_dim)
        all_labels: dict with 'level_1'..'level_4', each np.array
    """
    # ── Pass 1: count class frequencies ──
    print("Pass 1: Counting class frequencies...")
    temp_labels = {f'level_{i}': [] for i in range(1, 5)}
    valid_indices = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if len(sample['labels']) < 4:
            continue
        valid_indices.append(i)
        for lvl in range(4):
            temp_labels[f'level_{lvl + 1}'].append(sample['labels'][lvl])

    valid_classes = {}
    for level_name, labels in temp_labels.items():
        if not labels:
            valid_classes[level_name] = set()
            continue
        counts = Counter(labels)
        valid = {l for l, c in counts.items() if c >= min_class_size}
        valid_classes[level_name] = valid
        print(f"  {level_name}: {len(valid)} classes with >={min_class_size} samples "
              f"(total: {len(counts)})")

    # ── Pass 2: extract representations ONCE ──
    print("\nPass 2: Extracting representations from all layers...")
    all_layer_reprs = []
    ec_labels = {f'level_{i}': [] for i in range(1, 5)}

    model.eval()
    kept = 0

    for i in valid_indices:
        sample = dataset[i]

        has_valid = any(
            sample['labels'][lvl] in valid_classes[f'level_{lvl + 1}']
            for lvl in range(4)
        )
        if not has_valid:
            continue

        reprs = get_all_layer_representations(model, tokenizer, sample['seq'], model_type)
        all_layer_reprs.append(reprs)

        for lvl in range(4):
            ec_labels[f'level_{lvl + 1}'].append(sample['labels'][lvl])

        kept += 1
        if kept % 200 == 0:
            print(f"  {kept} proteins processed")

    print(f"\nTotal proteins: {kept}")
    return np.array(all_layer_reprs), {k: np.array(v) for k, v in ec_labels.items()}


# ══════════════════════════════════════════════════════════════
# PROBING
# ══════════════════════════════════════════════════════════════

def probe_layer(layer_num, all_reprs, labels, test_size=0.2):
    """Train a linear probe on a specific layer's representations."""
    X = all_reprs[:, layer_num, :]
    y = labels

    # Filter classes with <2 samples
    counts = Counter(y)
    valid = [l for l, c in counts.items() if c >= 2]
    mask = np.isin(y, valid)
    X_f, y_f = X[mask], y[mask]

    if len(X_f) < 10:
        return 0.0, 0.0

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_f, y_f, test_size=test_size, random_state=RANDOM_STATE_SEED, stratify=y_f)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_f, y_f, test_size=test_size, random_state=RANDOM_STATE_SEED)

    probe = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE_SEED)
    probe.fit(X_tr, y_tr)

    y_pred = probe.predict(X_te)
    return accuracy_score(y_te, y_pred), f1_score(y_te, y_pred, average='weighted')


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_ec_hierarchy_results(best_result, all_layer_results, n_layers, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    titles = ['EC Level 1 (Main Class)', 'EC Level 2 (Subclass)',
              'EC Level 3 (Sub-subclass)', 'EC Level 4 (Specific Enzyme)']
    colors = ['#58a6ff', '#4caf50', '#f39c12', '#e74c3c']

    for idx in range(4):
        ax = axes[idx]
        key = f'level_{idx + 1}'
        accs = all_layer_results[key]
        layers = list(range(len(accs)))

        ax.plot(layers, accs, 'o-', linewidth=2, markersize=8,
                color=colors[idx], markeredgecolor='white', markeredgewidth=1)

        best_layer = best_result[key]['layer']
        best_acc = best_result[key]['accuracy']

        ax.axvline(best_layer, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot(best_layer, best_acc, 'r*', markersize=15, zorder=5)
        ax.annotate(f'Layer {best_layer}\n{best_acc:.3f}',
                    xy=(best_layer, best_acc),
                    xytext=(best_layer + 0.5, best_acc - 0.05),
                    fontsize=10, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(titles[idx], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

    fig.suptitle('ESM-2: Layer-wise EC Classification Accuracy\n'
                 '(Logistic regression on mean-pooled representations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'ec_hierarchy_all_levels.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer selection for ESM-2")
    parser.add_argument("--model_type", type=str, default="huggingface",
                        choices=["huggingface", "fair-esm"],
                        help="Model API: 'huggingface' or 'fair-esm'")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D",
                        help="Model name (HF: 'facebook/esm2_t6_8M_UR50D', "
                             "fair-esm: 'esm2_t6_8M_UR50D' or 'esm2_t33_650M_UR50D')")
    parser.add_argument("--dataset_size", type=int, default=5000,
                        help="Max number of proteins to use")
    parser.add_argument("--min_class_size", type=int, default=5,
                        help="Min proteins per EC class to keep")
    parser.add_argument("--save_dir", type=str, default="../plots",
                        help="Directory for plots and results")
    args = parser.parse_args()

    # ── Load model ──
    if args.model_type == "fair-esm":
        import esm
        loader = getattr(esm.pretrained, args.model_name)
        model, alphabet = loader()
        model.eval()
        tokenizer = alphabet
        n_layers = model.num_layers + 1  # +1 for embedding layer 0
    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        model.eval()
        n_layers = model.config.num_hidden_layers + 1

    # ── Load dataset ──
    dataset = load_dataset("lightonai/SwissProt-EC-leaf", split="train")

    print(f"Model: {args.model_name} ({args.model_type})")
    print(f"Layers to probe: {n_layers} (0 through {n_layers - 1})")
    print(f"Dataset: {len(dataset)} total, using up to {args.dataset_size}")
    print(f"Min class size: {args.min_class_size}")
    print()

    # ── Extract representations ONCE ──
    all_reprs, all_labels = prepare_probing_data(
        model, tokenizer, dataset,
        num_samples=args.dataset_size,
        min_class_size=args.min_class_size,
        model_type=args.model_type
    )

    # ── Probe each layer for each EC level ──
    best_result = {}
    all_layer_results = {}

    for level_idx in range(4):
        level_name = f'level_{level_idx + 1}'
        labels = all_labels[level_name]
        n_classes = len(set(labels))

        print(f"\n{'─' * 50}")
        print(f"EC Level {level_idx + 1} ({n_classes} classes, {len(labels)} proteins)")
        print(f"{'─' * 50}")

        layer_accs = []
        best_acc, best_layer = 0, 0

        for layer in range(n_layers):
            acc, f1 = probe_layer(layer, all_reprs, labels)
            layer_accs.append(acc)
            print(f"  Layer {layer}: accuracy={acc:.4f}, f1={f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        best_result[level_name] = {'layer': best_layer, 'accuracy': best_acc}
        all_layer_results[level_name] = layer_accs
        print(f"  → Best: Layer {best_layer} ({best_acc:.4f})")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY — Best layer per EC level")
    print(f"{'=' * 60}")
    for level_idx in range(4):
        key = f'level_{level_idx + 1}'
        info = best_result[key]
        print(f"  EC Level {level_idx + 1}: Layer {info['layer']} → {info['accuracy']:.4f}")

    # ── Save ──
    os.makedirs(args.save_dir, exist_ok=True)
    plot_ec_hierarchy_results(best_result, all_layer_results, n_layers, args.save_dir)

    save_data = {
        "model": args.model_name,
        "model_type": args.model_type,
        "dataset_size": args.dataset_size,
        "best_result": {k: {"layer": int(v["layer"]), "accuracy": float(v["accuracy"])}
                        for k, v in best_result.items()},
        "all_layer_results": {k: [float(a) for a in v]
                              for k, v in all_layer_results.items()}
    }
    json_path = os.path.join(args.save_dir, "layer_selection_results.json")
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()