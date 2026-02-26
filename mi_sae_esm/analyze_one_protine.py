import torch
import esm
import json
import numpy as np
from datasets import load_dataset
from sae_model import SparseAutoencoderTopK


def analyze_protein(protein, model, alphabet, sae, device, layer=5, top_n_features=10, top_n_residues=8):
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    seq = protein['seq']
    data = [("test", seq)]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        results = model(tokens, repr_layers=[layer], return_contacts=False)
    residue_reprs = results["representations"][layer][0, 1:len(seq)+1, :]
    residue_reprs = (residue_reprs - residue_reprs.mean(dim=0)) / (residue_reprs.std(dim=0) + 1e-8)

    sae.eval()
    with torch.no_grad():
        _, features = sae(residue_reprs.to(device))

    feature_activity = features.sum(dim=0).cpu().numpy()
    top_feature_indices = feature_activity.argsort()[-top_n_features:][::-1]

    feature_list = []
    for feature_idx in top_feature_indices:
        activations = features[:, feature_idx].cpu().numpy()
        if activations.max() == 0:
            continue

        top_positions = activations.argsort()[-top_n_residues:][::-1]
        residues = []
        for pos in top_positions:
            if activations[pos] > 0:
                residues.append({
                    "pos": int(pos),
                    "aa": seq[pos],
                    "act": round(float(activations[pos]), 3)
                })

        if len(residues) == 0:
            continue

        aa_list = [r["aa"] for r in residues]
        unique_aas = set(aa_list)
        most_common_count = max(aa_list.count(aa) for aa in unique_aas)
        most_common_frac = most_common_count / len(aa_list)

        if most_common_frac >= 0.8:
            feature_type = "amino_acid"
            label = f"{aa_list[0]} detector"
        else:
            positions = sorted([r["pos"] for r in residues])
            span = positions[-1] - positions[0]
            if span < len(seq) * 0.3:
                feature_type = "motif"
                label = f"Region {positions[0]}-{positions[-1]} (mixed)"
            else:
                feature_type = "distributed"
                label = "Distributed pattern"

        feature_list.append({
            "id": int(feature_idx),
            "label": label,
            "type": feature_type,
            "total_activation": round(float(feature_activity[feature_idx]), 2),
            "residues": residues
        })

    active_count = int((features > 0).any(dim=0).sum().item())
    total_features = features.shape[1]

    return {
        "ec": protein['labels_str'],
        "length": len(seq),
        "active_features": active_count,
        "dead_features_pct": round((1 - active_count / total_features) * 100, 1),
        "features": feature_list
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()

    dataset = load_dataset("lightonai/SwissProt-EC-leaf", split="train")

    sae = SparseAutoencoderTopK(input_dim=320, hidden_dim=5120, k=256, aux_coeff=0.1)
    sae.load_state_dict(torch.load("../artifacts/sae_dict16_k256.pt", map_location=device))
    sae = sae.to(device)

    # Collect proteins with seq <= 512
    proteins = []
    for temp in dataset:
        if len(temp['seq']) <= 512:
            proteins.append(temp)
        if len(proteins) >= 20:
            break

    all_results = []
    for i, protein in enumerate(proteins):
        result = analyze_protein(protein, model, alphabet, sae, device)
        result["protein_idx"] = i
        all_results.append(result)

        motifs = sum(1 for f in result['features'] if f['type'] == 'motif')
        aa_det = sum(1 for f in result['features'] if f['type'] == 'amino_acid')
        print(f"Protein {i}: EC={result['ec']}, Length={result['length']}, "
              f"Motifs={motifs}, AA-det={aa_det}, Dead={result['dead_features_pct']}%")

    with open("../artifacts/feature_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} protein analyses to ../artifacts/feature_analysis.json")


if __name__ == '__main__':
    main()
