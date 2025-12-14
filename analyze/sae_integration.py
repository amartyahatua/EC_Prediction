"""
Integration script for SAE Feature Interpretation
Connects to your existing EC prediction pipeline
"""

import torch
import numpy as np
from analyze.sae_feature_interpretation import run_complete_interpretation_pipeline, SAEFeatureInterpreter
import pickle
import os
from mi_sae_esm.sae_esm import SparseAutoencoder  # Adjust import as needed
from sklearn.model_selection import train_test_split

def integrate_with_existing_sae(
    sae_model_path='sae_layer5.pt',
    layer_representations_path='layer5_representations.npz',
    output_dir='interpretation_results',
    device='cpu'
):
    """
    Integration function that loads your trained SAE and runs interpretation
    
    Args:
        sae_model_path: Path to saved SAE model
        layer_representations_path: Path to saved layer representations
        output_dir: Where to save interpretation results
        device: 'cpu' or 'cuda'
    """
    
    print("\n" + "="*70)
    print("INTEGRATING SAE FEATURE INTERPRETATION WITH EXISTING PIPELINE")
    print("="*70)
    
    # 1. Load the SAE model
    print("\n1. Loading SAE model...")
    checkpoint = torch.load(sae_model_path, map_location=device)

    input_dim = checkpoint['encoder.weight'].shape[1]
    hidden_dim = checkpoint['encoder.weight'].shape[0]
    
    sae_model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        l1_coeff=0.3  # Match your training config
    )
    sae_model.load_state_dict(checkpoint)
    sae_model.to(device)
    sae_model.eval()
    
    print(f"   Loaded SAE: {input_dim} -> {hidden_dim}")
    
    # 2. Load layer representations and labels
    print("\n2. Loading layer representations...")
    
    if os.path.exists(layer_representations_path):
        data = np.load(layer_representations_path, allow_pickle=True)
        X = data['representations']
        y = data['labels']
        print(f"   Loaded {len(X)} samples")
    else:
        print(f"   Warning: {layer_representations_path} not found!")
        print("   You need to save your layer representations first.")
        print("   Generating dummy data for demonstration...")
        
        # Generate dummy data for demonstration
        X = np.random.randn(1000, input_dim).astype(np.float32)
        y = np.random.randint(0, 10, 1000)
    
    # 3. Train/test split
    print("\n3. Creating train/test split...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # 4. Run interpretation pipeline
    print("\n4. Running interpretation pipeline...")
    
    results = run_complete_interpretation_pipeline(
        sae_model=sae_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        device=device
    )
    
    return results


def save_representations_from_test_script(layer_data, ec_labels, save_path):
    np.savez(
        save_path,
        representations=layer_data,
        labels=ec_labels
    )
    print(f"Saved representations to {save_path}")


def analyze_per_ec_class_features(
    interpreter,
    analysis_results,
    X_test,
    y_test,
    ec_classes_to_analyze=None
):
    """
    Analyze which features are important for specific EC classes
    
    Args:
        interpreter: SAEFeatureInterpreter instance
        analysis_results: Results from interpretation pipeline
        X_test: Test data
        y_test: Test labels
        ec_classes_to_analyze: List of EC classes to analyze (None = all)
    """
    
    print("\n" + "="*70)
    print("PER-EC-CLASS FEATURE ANALYSIS")
    print("="*70)
    
    unique_classes = np.unique(y_test)
    if ec_classes_to_analyze is None:
        ec_classes_to_analyze = unique_classes
    
    ec_class_features = {}
    
    for ec_class in ec_classes_to_analyze:
        print(f"\n{'='*70}")
        print(f"Analyzing EC Class: {ec_class}")
        print(f"{'='*70}")
        
        # Get samples for this class
        class_mask = y_test == ec_class
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            print(f"  No samples found for {ec_class}")
            continue
        
        # Get predictions for this class
        results = analysis_results['results']
        class_predictions = results['predictions'][class_indices]
        class_correct = class_predictions == ec_class
        
        n_correct = np.sum(class_correct)
        n_total = len(class_indices)
        
        print(f"\nClass Statistics:")
        print(f"  Total samples: {n_total}")
        print(f"  Correct predictions: {n_correct} ({n_correct/n_total*100:.1f}%)")
        print(f"  Incorrect predictions: {n_total - n_correct} ({(n_total-n_correct)/n_total*100:.1f}%)")
        
        # Analyze features for correct predictions
        if n_correct > 0:
            correct_indices_in_class = class_indices[class_correct]
            
            # Aggregate feature activations
            feature_activations = {}
            
            for idx in correct_indices_in_class[:min(20, len(correct_indices_in_class))]:
                sample_features = results['features'][idx]
                
                # Get top active features
                active_features = torch.where(sample_features > 0.1)[0]
                
                for feat_idx in active_features:
                    feat_idx = feat_idx.item()
                    if feat_idx not in feature_activations:
                        feature_activations[feat_idx] = []
                    feature_activations[feat_idx].append(sample_features[feat_idx].item())
            
            # Find most consistent features
            consistent_features = [
                (feat_idx, np.mean(activations), len(activations))
                for feat_idx, activations in feature_activations.items()
            ]
            consistent_features.sort(key=lambda x: (x[2], x[1]), reverse=True)
            
            print(f"\nTop features for CORRECT predictions of {ec_class}:")
            print(f"{'Feature':<10} {'Mean Act':<12} {'Frequency':<12}")
            print("-"*40)
            
            for feat_idx, mean_act, freq in consistent_features[:10]:
                print(f"{feat_idx:<10} {mean_act:<12.4f} {freq:<12}")
        
        # Analyze features for incorrect predictions
        if n_total - n_correct > 0:
            incorrect_indices_in_class = class_indices[~class_correct]
            
            print(f"\nMisclassified as:")
            misclass_counts = {}
            for idx in incorrect_indices_in_class:
                pred = results['predictions'][idx]
                misclass_counts[pred] = misclass_counts.get(pred, 0) + 1
            
            for pred_class, count in sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {pred_class}: {count} times ({count/len(incorrect_indices_in_class)*100:.1f}%)")
        
        ec_class_features[ec_class] = {
            'total': n_total,
            'correct': n_correct,
            'accuracy': n_correct / n_total if n_total > 0 else 0,
            'top_features': consistent_features[:10] if n_correct > 0 else []
        }
    
    return ec_class_features


def compare_feature_overlap(analysis_results, top_k=20):
    """
    Compare overlap between features driving correct vs incorrect predictions
    """
    print("\n" + "="*70)
    print("FEATURE OVERLAP ANALYSIS")
    print("="*70)
    
    correct_features = set([f[0] for f in analysis_results['top_correct_features'][:top_k]])
    incorrect_features = set([f[0] for f in analysis_results['top_incorrect_features'][:top_k]])
    
    overlap = correct_features & incorrect_features
    correct_only = correct_features - incorrect_features
    incorrect_only = incorrect_features - correct_features
    
    print(f"\nTop {top_k} features:")
    print(f"  Features in CORRECT only: {len(correct_only)}")
    print(f"  Features in INCORRECT only: {len(incorrect_only)}")
    print(f"  Overlapping features: {len(overlap)}")
    
    if overlap:
        print(f"\nOverlapping features: {sorted(overlap)}")
        print("\nThese features contribute to both correct and incorrect predictions!")
        print("They may represent general EC properties rather than discriminative features.")
    
    if correct_only:
        print(f"\nFeatures unique to CORRECT predictions: {sorted(list(correct_only)[:10])}")
        print("These are good discriminative features!")
    
    if incorrect_only:
        print(f"\nFeatures unique to INCORRECT predictions: {sorted(list(incorrect_only)[:10])}")
        print("These features mislead the model!")
    
    return {
        'overlap': overlap,
        'correct_only': correct_only,
        'incorrect_only': incorrect_only
    }