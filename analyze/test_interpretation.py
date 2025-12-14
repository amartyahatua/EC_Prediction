import numpy as np
import torch
from sklearn.model_selection import train_test_split
from mi_sae_esm.sae_esm import SparseAutoencoder, train_sae
from analyze.sae_feature_interpretation import run_complete_interpretation_pipeline
from collections import Counter

from analyze.sae_integration import (
    analyze_per_ec_class_features,
    compare_feature_overlap
)


def run_complete_pipeline_with_interpretation(
    dataset_path='your_dataset.parquet',
    layer_to_analyze=5,
    output_dir='complete_results'
):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(" "*25 + "COMPLETE INTERPRETATION PIPELINE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Extract Layer Representations (from your existing code)
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 1: Extracting Layer {layer_to_analyze} Representations")
    print(f"{'='*80}")

    print("Loading pre-extracted representations...")
    representations_file = f'layer{layer_to_analyze}_representations.npz'
    
    if os.path.exists(representations_file):
        data = np.load(representations_file, allow_pickle=True)
        representations = data['representations']
        ec_labels = data['labels']
        print(f"Loaded {len(representations)} samples")
    else:
        print(f"ERROR: {representations_file} not found!")
        print("Please extract representations first.")
        return None
    
    # ========================================================================
    # STEP 2: Train SAE (from your existing SAE code)
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: Training Sparse Autoencoder")
    print(f"{'='*80}")
    

    
    input_dim = representations.shape[1]
    hidden_dim = input_dim * 4  # 4x expansion
    
    sae_model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        l1_coeff=0.3
    )
    
    # Train SAE
    sae_model = train_sae(
        sae_model,
        representations,
        epochs=50,
        batch_size=128,
        lr=1e-3
    )

    # Save SAE
    sae_path = os.path.join(output_dir, f'sae_layer{layer_to_analyze}.pt')
    torch.save(sae_model.state_dict(), sae_path)
    print(f"\nSAE saved to: {sae_path}")
    
    # ========================================================================
    # STEP 3: Train/Test Split
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Creating Train/Test Split")
    print(f"{'='*80}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        representations,
        ec_labels,
        test_size=0.2,
        random_state=42,
        stratify=ec_labels
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Unique classes: {len(np.unique(ec_labels))}")
    
    # ========================================================================
    # STEP 4: Feature Interpretation - Correct vs Incorrect
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: SAE Feature Interpretation")
    print(f"{'='*80}")
    
    interpretation_dir = os.path.join(output_dir, 'interpretation')
    
    interpretation_results = run_complete_interpretation_pipeline(
        sae_model=sae_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=interpretation_dir,
        device='cpu'
    )
    
    # ========================================================================
    # STEP 5: Per-EC Class Analysis
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: Per-EC Class Feature Analysis")
    print(f"{'='*80}")
    
    # Analyze top classes
    class_counts = {}
    for label in y_test:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_class_names = [c[0] for c in top_classes]
    
    print(f"\nAnalyzing top {len(top_class_names)} EC classes by sample count...")
    
    ec_features = analyze_per_ec_class_features(
        interpreter=interpretation_results['interpreter'],
        analysis_results=interpretation_results['analysis_results'],
        X_test=X_test,
        y_test=y_test,
        ec_classes_to_analyze=top_class_names
    )
    
    # ========================================================================
    # STEP 6: Feature Overlap Analysis
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 6: Feature Overlap Analysis")
    print(f"{'='*80}")
    
    overlap_results = compare_feature_overlap(
        interpretation_results['analysis_results'],
        top_k=20
    )
    
    # ========================================================================
    # STEP 7: Generate Summary Report
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 7: Generating Summary Report")
    print(f"{'='*80}")
    
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAE FEATURE INTERPRETATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Layer Analyzed: {layer_to_analyze}\n")
        f.write(f"Input Dimension: {input_dim}\n")
        f.write(f"SAE Hidden Dimension: {hidden_dim}\n")
        f.write(f"Total Samples: {len(representations)}\n")
        f.write(f"Train Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"Unique EC Classes: {len(np.unique(ec_labels))}\n\n")
        
        f.write(f"Probe Test Accuracy: {interpretation_results['test_accuracy']:.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TOP FEATURES FOR CORRECT CLASSIFICATIONS\n")
        f.write("-"*80 + "\n")
        
        for feat_idx, attrs in interpretation_results['analysis_results']['top_correct_features'][:20]:
            f.write(f"Feature {feat_idx}: "
                   f"Attribution={attrs['mean_attribution']:.4f}, "
                   f"Activation={attrs['mean_activation']:.4f}, "
                   f"Count={attrs['count']}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("TOP FEATURES FOR INCORRECT CLASSIFICATIONS\n")
        f.write("-"*80 + "\n")
        
        for feat_idx, attrs in interpretation_results['analysis_results']['top_incorrect_features'][:20]:
            f.write(f"Feature {feat_idx}: "
                   f"Supports Wrong={attrs['supports_wrong']}, "
                   f"Activation={attrs['mean_activation']:.4f}, "
                   f"Count={attrs['count']}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("FEATURE OVERLAP STATISTICS\n")
        f.write("-"*80 + "\n")
        
        f.write(f"Features unique to CORRECT: {len(overlap_results['correct_only'])}\n")
        f.write(f"Features unique to INCORRECT: {len(overlap_results['incorrect_only'])}\n")
        f.write(f"Overlapping features: {len(overlap_results['overlap'])}\n")
        
        if overlap_results['correct_only']:
            f.write(f"\nDiscriminative features (correct only): {sorted(list(overlap_results['correct_only']))}\n")
        
        if overlap_results['incorrect_only']:
            f.write(f"\nMisleading features (incorrect only): {sorted(list(overlap_results['incorrect_only']))}\n")
    
    print(f"\nSummary report saved to: {summary_path}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - SAE model: {sae_path}")
    print(f"  - Feature summary: {os.path.join(interpretation_dir, 'feature_summary.csv')}")
    print(f"  - Visualizations: {os.path.join(interpretation_dir, 'feature_comparison.png')}")
    print(f"  - Summary report: {summary_path}")
    
    return {
        'sae_model': sae_model,
        'interpretation_results': interpretation_results,
        'ec_features': ec_features,
        'overlap_results': overlap_results,
        'output_dir': output_dir
    }



def quick_interpretation_on_existing_sae(
    class_level,
    layer,
    sae_model_path,
    representations_path,
    output_dir
):

    print("\n" + "="*80)
    print("QUICK INTERPRETATION ON EXISTING SAE")
    print("="*80)

    # Load SAE
    print("\n1. Loading SAE model...")
    checkpoint = torch.load(sae_model_path, map_location='cpu')
    
    input_dim = checkpoint['encoder.weight'].shape[1]
    hidden_dim = checkpoint['encoder.weight'].shape[0]

    sae_model = SparseAutoencoder(input_dim, hidden_dim, l1_coeff=0.3)
    sae_model.load_state_dict(checkpoint)
    sae_model.eval()
    
    # Load data
    print("\n2. Loading representations...")
    data = np.load(representations_path, allow_pickle=True)
    representations = data['representations']
    ec_labels = data['labels']

    # Check and filter rare classes BEFORE train_test_split
    print("\nChecking class distribution...")
    class_counts = Counter(ec_labels)

    print(f"Total samples: {len(ec_labels)}")
    print(f"Total classes: {len(class_counts)}")

    # Find classes with insufficient samples
    MIN_SAMPLES_FOR_SPLIT = 2  # Minimum for stratified split
    rare_classes = {cls: cnt for cls, cnt in class_counts.items() if cnt < MIN_SAMPLES_FOR_SPLIT}

    if rare_classes:
        print(f"\n⚠️ Found {len(rare_classes)} classes with < {MIN_SAMPLES_FOR_SPLIT} samples:")
        for cls, cnt in sorted(rare_classes.items(), key=lambda x: x[1])[:10]:
            print(f"  Class {cls}: {cnt} sample(s)")

        # Filter out rare classes
        valid_classes = {cls for cls, cnt in class_counts.items() if cnt >= MIN_SAMPLES_FOR_SPLIT}
        mask = np.array([label in valid_classes for label in ec_labels])

        # Apply filter
        representations_filtered = representations[mask]
        ec_labels_filtered = ec_labels[mask]

        removed_samples = len(representations) - len(representations_filtered)
        removed_classes = len(class_counts) - len(valid_classes)

        print(f"\nFiltered out:")
        print(f"  {removed_samples} samples ({removed_samples / len(representations) * 100:.1f}%)")
        print(f"  {removed_classes} classes ({removed_classes / len(class_counts) * 100:.1f}%)")
        print(f"\nRemaining:")
        print(f"  {len(representations_filtered)} samples")
        print(f"  {len(valid_classes)} classes")

        # Use filtered data
        representations = representations_filtered
        ec_labels = ec_labels_filtered
    else:
        print("✓ All classes have sufficient samples for stratified split")





    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            representations, ec_labels,
            test_size=0.2, random_state=42, stratify=ec_labels
        )

        # Run interpretation
        print("\n3. Running interpretation...")
        results = run_complete_interpretation_pipeline(
            class_level,
            layer,
            sae_model=sae_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
        )

        # Additional analyses
        print("\n4. Additional analyses...")

        overlap = compare_feature_overlap(results['analysis_results'], top_k=20)

        # Get top classes
        class_counts = {}
        for label in y_test:
            class_counts[label] = class_counts.get(label, 0) + 1

        top_classes = [c[0] for c in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]]

        ec_features = analyze_per_ec_class_features(
            interpreter=results['interpreter'],
            analysis_results=results['analysis_results'],
            X_test=X_test,
            y_test=y_test,
            ec_classes_to_analyze=top_classes
        )

        print(f"\n{'='*80}")
        print("INTERPRETATION COMPLETE!")
        print(f"Results in: {output_dir}")
        print(f"{'='*80}")

        return results
    except:
        print(f"Got error in spliting data in train and test sets for Class level {class_level}")
        return None


if __name__ == "__main__":
    import sys
    import os

    print("SAE Feature Interpretation - Enhanced Test Script")
    print("="*80)

    # Check what files exist
    has_sae = os.path.exists('../artifacts/sae_layer_5.pt')
    has_reps = os.path.exists('../artifacts/layer_6_1_representations.npz')

    if has_sae and has_reps:
        print("\nFound existing SAE and representations!")
        print("Running quick interpretation...")
        class_level = 1
        layer = 5
        results = quick_interpretation_on_existing_sae(
            class_level,
            layer,
            sae_model_path=f'../artifacts/sae_layer_{layer}.pt',
            representations_path=f'../artifacts/layer_{layer+1}_{class_level}_representations.npz',
            output_dir='interpretation_results'
        )

    elif has_reps:
        print("\nFound representations but no SAE.")
        print("Running complete pipeline...")

        results = run_complete_pipeline_with_interpretation(
            layer_to_analyze=5,
            output_dir='complete_results'
        )

    else:
        print("\n" + "="*80)
        print("SETUP INSTRUCTIONS")
        print("="*80)
