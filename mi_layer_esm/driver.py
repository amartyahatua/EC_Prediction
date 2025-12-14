import os
import torch
import argparse
import numpy as np
from get_dataset import *
from get_model import *
from layer_information import *
from analyze.sae_integration import *
from mi_sae_esm.sae_esm import train_sae_pipeline

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_TOKEN"] = "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"

RANDOM_STATE_SEED = 1829873
np.random.seed(RANDOM_STATE_SEED)
torch.manual_seed(RANDOM_STATE_SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EC Prediction Training Script")
    parser.add_argument("--dataset_size", type=int, default=550, help="Set the number of data points")
    parser.add_argument("--dataset_name", type=str, default="DanielHesslow/SwissProt-EC", help="Set dataset name")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Set the modle name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=5120, help="Hidden dimension")
    parser.add_argument("--l1_coeff", type=float, default=0.6, help="Number of layers")

    args = parser.parse_args()
    train_data, test_data = get_dataset(args.dataset_name)
    model, tokenizer = get_plm(args.model_name)
    N_LAYERS = model.config.num_hidden_layers +1
    N_LAVEL = 4

    all_reprs, all_labels, results = get_layer_label_information(args, model, N_LAYERS, tokenizer, train_data)


    best_result = ec_hierarchy_all_levels(args, model, tokenizer,train_data,N_LAYERS)

    train_sae_pipeline(args, all_reprs, all_labels, N_LAYERS, best_result)

    # Save layer information for each lavel
    for i in range(N_LAYERS):
        layer_representations = all_reprs[:, i, :]
        for lavel_count in range(N_LAVEL):
            ec_labels_array = all_labels[f'level_{lavel_count+1}']  # For analysis
            save_representations_from_test_script(
                layer_data=layer_representations,  # Your extracted layer 5 data
                ec_labels=ec_labels_array,           # Your EC labels
                save_path=f'../artifacts/layer_{i+1}_{lavel_count}_representations.npz'
            )
    count = 0 # level
    for level in best_result.keys():
        print('Level: ',level)
        layer_representations = all_reprs[:, count, :]
        layer = best_result[level]['layer']
        sae_model_path = f'../artifacts/sae_layer_{layer}.pt',
        representations_path = f'../artifacts/layer_{layer}_{count}_representations.npz'
        output_dir = 'interpretation_results'
        count += 1


