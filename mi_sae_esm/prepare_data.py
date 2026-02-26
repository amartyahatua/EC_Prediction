import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader


def get_single_protein(args, model, alphabet, protein_idx=0):
    dataset = load_dataset(args.dataset_name, split="train")
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    count = 0
    for temp in dataset:
        if len(temp['seq']) <= args.max_length:
            if count == protein_idx:
                protein = temp
                break
            count += 1

    seq = protein['seq']
    data = [("test", seq)]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        results = model(tokens, repr_layers=[5], return_contacts=False)

    residue_reprs = results["representations"][5][0, 1:len(seq)+1, :]

    # Same normalization as training
    residue_reprs = (residue_reprs - residue_reprs.mean(dim=0)) / (residue_reprs.std(dim=0) + 1e-8)

    return protein, residue_reprs


