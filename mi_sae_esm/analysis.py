"""
End-to-End SAE Feature Analysis for Enzyme Classification
==========================================================
Step 1: EC Specificity Analysis
Step 2: Linear Probe
Step 3: Progressive Causal Ablation
Step 4: Multi-layer Comparison (requires SAEs on all layers)
"""

import torch
import esm
import json
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sae_model import SparseAutoencoderTopK


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
TARGET_ECS = [
    'EC:2.7.7.6',
    'EC:4.2.1.11',
    'EC:6.1.1.17',
    'EC:3.6.4.12',
    'EC:2.7.11.1'
]
PROTEINS_PER_EC = 20
MAX_LENGTH = 512
LAYER = 5
SAE_PATH = "../artifacts/sae_dict16_k256.pt"


# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════
def setup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()

    sae = SparseAutoencoderTopK(input_dim=320, hidden_dim=5120, k=256, aux_coeff=0.1)
    sae.load_state_dict(torch.load(SAE_PATH, map_location=device))
    sae.eval()
    sae = sae.to(device)

    dataset = load_dataset("lightonai/SwissProt-EC-leaf", split="train")

    return model, alphabet, sae, dataset, device


# ══════════════════════════════════════════════════════════════
# COLLECT PROTEINS BY EC CLASS
# ══════════════════════════════════════════════════════════════
def collect_proteins(dataset):
    ec_proteins = defaultdict(list)

    for temp in dataset:
        if len(temp['seq']) > MAX_LENGTH:
            continue

        raw = temp['labels_str']
        if isinstance(raw, list):
            ec = raw[0]
        elif isinstance(raw, str) and raw.startswith('['):
            ec = raw.strip("[]'\" ")
        else:
            ec = raw

        if ec in TARGET_ECS and len(ec_proteins[ec]) < PROTEINS_PER_EC:
            ec_proteins[ec].append(temp)

    for ec in TARGET_ECS:
        print(f"  {ec}: {len(ec_proteins[ec])} proteins collected")

    return ec_proteins


# ══════════════════════════════════════════════════════════════
# EXTRACT SAE FEATURES FOR ONE PROTEIN
# ══════════════════════════════════════════════════════════════
def get_sae_features(protein, model, alphabet, sae, device, layer=5):
    batch_converter = alphabet.get_batch_converter()
    seq = protein['seq']
    data = [("test", seq)]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        results = model(tokens, repr_layers=[layer], return_contacts=False)
    residue_reprs = results["representations"][layer][0, 1:len(seq)+1, :]
    residue_reprs = (residue_reprs - residue_reprs.mean(dim=0)) / (residue_reprs.std(dim=0) + 1e-8)

    with torch.no_grad():
        _, features = sae(residue_reprs.to(device))

    return features  # (seq_len, 5120)


# ══════════════════════════════════════════════════════════════
# STEP 1: EC SPECIFICITY ANALYSIS
# ══════════════════════════════════════════════════════════════
def ec_specificity_analysis(ec_proteins, model, alphabet, sae, device):
    print("\n" + "=" * 60)
    print("STEP 1: EC SPECIFICITY ANALYSIS")
    print("=" * 60)

    ec_feature_profiles = defaultdict(list)

    for ec, proteins in ec_proteins.items():
        print(f"\n  Processing {ec}...")
        for i, protein in enumerate(proteins):
            features = get_sae_features(protein, model, alphabet, sae, device)
            mean_features = features.mean(dim=0).cpu().numpy()  # (5120,)
            ec_feature_profiles[ec].append(mean_features)
            if (i + 1) % 5 == 0:
                print(f"    {i + 1}/{len(proteins)}")

    # Stack into 2D arrays
    all_profiles = {}
    for ec in TARGET_ECS:
        profiles = ec_feature_profiles[ec]
        if len(profiles) > 0:
            all_profiles[ec] = np.vstack(profiles)  # (n_proteins, 5120)
        else:
            print(f"  WARNING: No profiles for {ec}")
            all_profiles[ec] = np.zeros((1, 5120))

    # Compute per-feature EC specificity (F-ratio: between-EC / within-EC variance)
    n_features = 5120
    specificity_scores = np.zeros(n_features)

    for f in range(n_features):
        ec_means = [all_profiles[ec][:, f].mean() for ec in TARGET_ECS]
        between_var = np.var(ec_means)

        within_vars = [all_profiles[ec][:, f].var() for ec in TARGET_ECS]
        within_var = np.mean(within_vars)

        if within_var > 1e-10:
            specificity_scores[f] = between_var / within_var
        else:
            specificity_scores[f] = 0

    # Report top EC-specific features
    top_specific = specificity_scores.argsort()[-20:][::-1]
    print(f"\n  Top 20 EC-specific features (by F-ratio):")
    for rank, fid in enumerate(top_specific):
        print(f"    Rank {rank + 1}: Feature {fid}, F-ratio: {specificity_scores[fid]:.3f}")
        for ec in TARGET_ECS:
            mean_act = all_profiles[ec][:, fid].mean()
            if mean_act > 0.01:
                print(f"      {ec}: mean activation = {mean_act:.4f}")

    return all_profiles, specificity_scores, top_specific


# ══════════════════════════════════════════════════════════════
# STEP 2: LINEAR PROBE
# ══════════════════════════════════════════════════════════════
def linear_probe(all_profiles):
    print("\n" + "=" * 60)
    print("STEP 2: LINEAR PROBE — EC Classification from SAE Features")
    print("=" * 60)

    X = []
    y = []
    for ec_idx, ec in enumerate(TARGET_ECS):
        for profile in all_profiles[ec]:
            X.append(profile)
            y.append(ec_idx)

    X = np.array(X)
    y = np.array(y)
    print(f"  Dataset: {X.shape[0]} proteins, {X.shape[1]} features, {len(TARGET_ECS)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[ec.replace("EC:", "") for ec in TARGET_ECS]
    ))

    # Top features per class
    print(f"\n  Top 10 features per EC class (by logistic regression weight):")
    for ec_idx, ec in enumerate(TARGET_ECS):
        weights = clf.coef_[ec_idx]
        top_features = weights.argsort()[-10:][::-1]
        print(f"\n    {ec}:")
        for fid in top_features:
            print(f"      Feature {fid}: weight = {weights[fid]:.4f}")

    return clf, acc, X_test, y_test


# ══════════════════════════════════════════════════════════════
# STEP 3: PROGRESSIVE CAUSAL ABLATION
# ══════════════════════════════════════════════════════════════
def causal_ablation_progressive(ec_proteins, model, alphabet, sae, device, clf):
    print("\n" + "=" * 60)
    print("STEP 3: PROGRESSIVE CAUSAL ABLATION")
    print("=" * 60)
    print("  Keeping only top-N features (by probe weight), zeroing the rest\n")

    ablation_levels = [10, 25, 50, 100, 200, 500, 1000, 2000, 5120]
    all_results = {}

    for ec_idx, ec in enumerate(TARGET_ECS):
        proteins = ec_proteins[ec][:10]
        weights = np.abs(clf.coef_[ec_idx])
        sorted_features = weights.argsort()[::-1]

        print(f"  {ec}:")

        ec_results = {}

        for n_keep in ablation_levels:
            keep_set = set(sorted_features[:n_keep].tolist())

            correct = 0
            total = 0

            for protein in proteins:
                features = get_sae_features(protein, model, alphabet, sae, device)
                mean_features = features.mean(dim=0).cpu().numpy().reshape(1, -1)

                ablated = np.zeros_like(mean_features)
                for f in keep_set:
                    ablated[0, f] = mean_features[0, f]

                pred = clf.predict(ablated)[0]
                if pred == ec_idx:
                    correct += 1
                total += 1

            acc = correct / total
            ec_results[n_keep] = acc
            print(f"    Keep top {n_keep:5d} features: accuracy = {acc:.2f}")

        all_results[ec] = ec_results

    # Summary: find minimum features needed for 90%+ accuracy per EC
    print(f"\n  Summary — Minimum features for ≥90% accuracy:")
    for ec in TARGET_ECS:
        for n_keep in ablation_levels:
            if all_results[ec][n_keep] >= 0.9:
                print(f"    {ec}: {n_keep} features sufficient")
                break

    return all_results


# ══════════════════════════════════════════════════════════════
# STEP 4: MULTI-LAYER COMPARISON
# ══════════════════════════════════════════════════════════════
def multi_layer_comparison(ec_proteins, model, alphabet, device):
    print("\n" + "=" * 60)
    print("STEP 4: MULTI-LAYER COMPARISON")
    print("=" * 60)

    layer_results = {}

    for layer in range(1, 7):  # ESM-2 8M has 6 layers
        print(f"\n  Layer {layer}/6:")

        sae_path = f"../artifacts/sae_dict16_k256_layer{layer}.pt"
        try:
            sae = SparseAutoencoderTopK(input_dim=320, hidden_dim=5120, k=256, aux_coeff=0.1)
            sae.load_state_dict(torch.load(sae_path, map_location=device))
            sae.eval()
            sae = sae.to(device)
        except FileNotFoundError:
            print(f"    SAE not found at {sae_path}, skipping")
            continue

        X = []
        y = []
        for ec_idx, ec in enumerate(TARGET_ECS):
            for protein in ec_proteins[ec][:10]:
                batch_converter = alphabet.get_batch_converter()
                seq = protein['seq']
                data = [("test", seq)]
                _, _, tokens = batch_converter(data)

                with torch.no_grad():
                    results = model(tokens, repr_layers=[layer], return_contacts=False)
                residue_reprs = results["representations"][layer][0, 1:len(seq) + 1, :]
                residue_reprs = (residue_reprs - residue_reprs.mean(dim=0)) / (residue_reprs.std(dim=0) + 1e-8)

                with torch.no_grad():
                    _, features = sae(residue_reprs.to(device))

                mean_features = features.mean(dim=0).cpu().numpy()
                X.append(mean_features)
                y.append(ec_idx)

        X = np.array(X)
        y = np.array(y)

        if len(X) < 10:
            print(f"    Not enough data, skipping")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        layer_results[layer] = acc
        print(f"    EC classification accuracy: {acc:.4f}")

    if layer_results:
        print(f"\n  Summary:")
        for layer, acc in sorted(layer_results.items()):
            bar = "█" * int(acc * 50)
            print(f"    Layer {layer}: {acc:.4f} {bar}")

    return layer_results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    model, alphabet, sae, dataset, device = setup()

    print("\nCollecting proteins by EC class...")
    ec_proteins = collect_proteins(dataset)

    # Step 1: EC Specificity
    all_profiles, specificity_scores, top_specific = ec_specificity_analysis(
        ec_proteins, model, alphabet, sae, device
    )

    # Step 2: Linear Probe
    clf, acc, X_test, y_test = linear_probe(all_profiles)

    # Step 3: Progressive Ablation
    ablation_results = causal_ablation_progressive(
        ec_proteins, model, alphabet, sae, device, clf
    )

    # Step 4: Multi-layer (only runs if SAEs exist for other layers)
    layer_results = multi_layer_comparison(ec_proteins, model, alphabet, device)

    # Save results
    results = {
        "linear_probe_accuracy": acc,
        "top_specific_features": top_specific.tolist(),
        "ablation_results": ablation_results,
        "layer_results": layer_results
    }

    with open("../artifacts/end_to_end_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll results saved to ../artifacts/end_to_end_results.json")


if __name__ == '__main__':
    main()