import os
from transformers import EsmForMaskedLM, EsmTokenizer
from datasets import load_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import json
from collections import Counter

os.environ["HF_TOKEN"] = ""

# Load model
def get_model(model_name):
    """
    :param model_name:String
    :return: pLM model
    """
    model_name = "facebook/esm2_t6_8M_UR50D"  # Or another ESM-2 variant
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)  # Or EsmModel for embeddings
    return model, tokenizer

def get_dataset(dataset_name="DanielHesslow/SwissProt-EC"):
    """
    :param dataset_name:String
    :return: dataset with train, test and val datasets
    """

    # Example: Loading a SwissProt subset with Pfam labels
    dataset = load_dataset(dataset_name)
    # You can access the data splits if available (often just 'train')
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Example
    print("Example of one data point: ")
    print(train_data["seq"])
    print(test_data["labels"])
    print(test_data["labels_str"])
    print(test_data["id"])

    return train_data, test_data

def get_all_layer_representations(sequence):
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



# 1. Load dataset
pilot_size = 5000
train_data, test_data = get_dataset()
pilot_data = train_data.select(range(pilot_size))


# 2. Load ESM-2
model, tokenizer = get_model(model_name="facebook/esm2_t6_8M_UR50D")
model.eval()  # Set to evaluation mode


# 3. Extract representations for one protein


# 3. Example: Extract for first protein
sample = pilot_data[0]
layer_reprs = get_all_layer_representations(sample['seq'])
print(f"Shape: {layer_reprs.shape}")  # (33, 1280)
print(f"EC labels: {sample['labels_str']}")













#############################################################


# Load the EC index mapping
# You'll need to download idx_mapping.json from the HuggingFace dataset repo
# with open('idx_mapping.json', 'r') as f:
#     idx_to_ec = json.load(f)


# Function to extract EC level from labels
def get_ec_level_label(labels_list, level):
    """
    Extract EC label for a specific hierarchy level
    level: 0 (EC:X.-.-.-), 1 (EC:X.Y.-.-), 2 (EC:X.Y.Z.-), 3 (EC:X.Y.Z.W)
    """
    return labels_list[level]  # labels already ordered hierarchically


# Prepare data for probing
def prepare_probing_data(dataset_split, num_samples=5000, min_class_size=10):
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
        reprs = get_all_layer_representations(sample['seq'])
        all_layer_reprs.append(reprs)

        # Get labels for each EC level
        for level_idx in range(4):
            level_name = f'level_{level_idx + 1}'
            ec_labels[level_name].append(sample['labels'][level_idx])

        if len(all_layer_reprs) % 100 == 0:
            print(f"Processed {len(all_layer_reprs)} valid proteins")

    print(f"\nTotal proteins kept: {len(all_layer_reprs)}")

    return np.array(all_layer_reprs), {
        'level_1': np.array(ec_labels['level_1']),
        'level_2': np.array(ec_labels['level_2']),
        'level_3': np.array(ec_labels['level_3']),
        'level_4': np.array(ec_labels['level_4'])
    }


# Layer-wise probing for a specific EC level
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

# Run full experiment
def run_layer_probing_experiment(dataset_split, ec_level='level_1', num_samples=5000):
    """
    Probe all 33 layers for a specific EC hierarchy level
    """

    print(f"Extracting representations for {num_samples} proteins...")
    all_reprs, all_labels = prepare_probing_data(dataset_split, num_samples)

    labels = all_labels[ec_level]

    print(f"\nProbing all layers for {ec_level}...")
    results = []

    for layer in range(7):
        acc, f1 = probe_layer(layer, all_reprs, labels)
        results.append({'layer': layer, 'accuracy': acc, 'f1': f1})
        print(f"Layer {layer}: Accuracy={acc:.3f}, F1={f1:.3f}")

    return results




# Run the experiment
# results = run_layer_probing_experiment(
#     train_data,
#     ec_level='level_1',  # Start with EC level 1 (broadest)
#     num_samples=5000
# )

# Run for all 4 EC levels
for level in ['level_1', 'level_2', 'level_3', 'level_4']:
    print(f"\n{'='*50}")
    print(f"Testing {level}")
    print(f"{'='*50}")
    results = run_layer_probing_experiment(
        train_data,
        ec_level=level,
        num_samples=5000
    )

# Plot results
import matplotlib.pyplot as plt

# After running all 4 levels, create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ec_levels = ['level_1', 'level_2', 'level_3', 'level_4']
level_names = ['EC Level 1 (Main Class)', 'EC Level 2 (Subclass)',
               'EC Level 3 (Sub-subclass)', 'EC Level 4 (Specific)']

for idx, (level, name) in enumerate(zip(ec_levels, level_names)):
    ax = axes[idx // 2, idx % 2]

    # Run experiment
    results = run_layer_probing_experiment(
        train_data,
        ec_level=level,
        num_samples=5000
    )

    layers = [r['layer'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    ax.plot(layers, accuracies, marker='o', linewidth=2)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(7))

    # Highlight best layer
    best_layer = layers[np.argmax(accuracies)]
    best_acc = max(accuracies)
    ax.axvline(best_layer, color='red', linestyle='--', alpha=0.5)
    ax.text(best_layer, best_acc, f'  Layer {best_layer}\n  {best_acc:.3f}',
            fontsize=10, color='red')

plt.tight_layout()
plt.savefig('ec_hierarchy_all_levels.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nBest layer for EC Level 1: Layer {layers[np.argmax(accuracies)]}")
print(f"Best accuracy: {max(accuracies):.3f}")