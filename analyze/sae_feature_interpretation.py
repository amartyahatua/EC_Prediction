"""
SAE Feature Interpretation Pipeline
Identifies which features drive correct vs incorrect classifications
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import os
from mi_layer_esm.get_dataset import get_dataset


class SAEFeatureInterpreter:
    """
    Comprehensive SAE feature interpretation focusing on 
    correct vs incorrect classification patterns
    """
    
    def __init__(self, sae_model, device='cpu'):
        self.sae_model = sae_model
        self.device = device
        self.probe = None
        self.feature_attributions = {}
        
    def encode_data(self, X):
        """Encode data through SAE"""
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            features = self.sae_model.encode(X_tensor)
        return features

    def get_activating_proteins(self, feature_idx, X, protein_ids,
                                threshold_percentile=90, min_activation=0.1):
        """
        For a given feature, return which proteins activate it strongly

        Args:
            feature_idx: Index of the SAE feature to analyze
            X: Input representations (n_proteins, hidden_dim)
            protein_ids: Array of protein identifiers
            threshold_percentile: Percentile threshold for "high activation"
            min_activation: Minimum absolute activation to consider

        Returns:
            Dictionary with activating protein information
        """
        # Encode through SAE
        features = self.encode_data(X)
        feature_activations = features[:, feature_idx].cpu().numpy()

        # Filter by minimum activation
        min_mask = feature_activations > min_activation

        # Apply percentile threshold
        if min_mask.sum() > 0:
            threshold = np.percentile(feature_activations[min_mask], threshold_percentile)
        else:
            threshold = min_activation

        activating_mask = feature_activations > threshold

        # Get indices sorted by activation (highest first)
        activating_indices = np.where(activating_mask)[0]
        sorted_indices = activating_indices[np.argsort(-feature_activations[activating_indices])]

        return {
            'feature_idx': feature_idx,
            'protein_indices': sorted_indices,
            'protein_ids': protein_ids[sorted_indices] if protein_ids is not None else sorted_indices,
            'activations': feature_activations[sorted_indices],
            'n_proteins': len(sorted_indices),
            'threshold': threshold,
            'max_activation': feature_activations.max(),
            'mean_activation': feature_activations[activating_mask].mean() if activating_mask.sum() > 0 else 0
        }


    def analyze_top_features_with_proteins(self, X, protein_ids, top_features_per_class,
                                           top_k_features=5):
        """
        For each EC class's top features, find which proteins activate them.

        Args:
            X: Input representations (test set)
            protein_ids: Array of protein IDs (test set)
            top_features_per_class: Output from get_top_features_per_class()
            top_k_features: How many top features per class to analyze

        Returns:
            Dictionary with activation information per feature
        """
        print("\n" + "=" * 70)
        print("ANALYZING TOP FEATURES - FINDING ACTIVATING PROTEINS")
        print("=" * 70)

        # Encode data through SAE
        features = self.encode_data(X)

        feature_analysis = {}

        for ec_class, top_features in top_features_per_class.items():
            print(f"\n--- EC Class: {ec_class} ---")

            for i, (feat_idx, coef) in enumerate(top_features[:top_k_features], 1):
                print(f"\n  Feature {feat_idx} (coefficient={coef:.4f}):")

                # Get activations for this feature
                activating_info = self.get_activating_proteins(
                    feat_idx,
                    X,
                    protein_ids,
                    threshold_percentile=90,
                    min_activation=0.1
                )

                print(f"    Activates in: {activating_info['n_proteins']} proteins")
                print(f"    Max activation: {activating_info['max_activation']:.4f}")
                print(f"    Top 5 proteins:")

                for j in range(min(5, len(activating_info['protein_ids']))):
                    pid = activating_info['protein_ids'][j]
                    act = activating_info['activations'][j]
                    print(f"      {pid}: {act:.4f}")

                # Store for later use
                feature_key = f"EC{ec_class}_Feature{feat_idx}"
                feature_analysis[feature_key] = {
                    'ec_class': ec_class,
                    'feature_idx': feat_idx,
                    'coefficient': coef,
                    'activating_proteins': activating_info['protein_ids'],
                    'activations': activating_info['activations'],
                    'n_proteins': activating_info['n_proteins']
                }

        return feature_analysis


    def get_top_features_per_class(self, top_k=10):
        """
        For each EC class, find the features with highest coefficients.

        Returns:
            Dictionary mapping class_label -> list of (feature_idx, coefficient)
        """
        if self.probe is None:
            raise ValueError("Must train probe first!")

        coefficients = self.probe.coef_  # Shape: (n_classes, n_features)
        class_labels = self.probe.classes_

        top_features_per_class = {}

        for class_idx, class_label in enumerate(class_labels):
            # Get coefficients for this class
            class_coefs = coefficients[class_idx, :]

            # Get indices of top k features by absolute value
            top_indices = np.argsort(np.abs(class_coefs))[-top_k:][::-1]

            # Store feature index and coefficient value
            top_features_per_class[class_label] = [
                (feat_idx, class_coefs[feat_idx])
                for feat_idx in top_indices
            ]

        return top_features_per_class

    def train_probe(self, X_train, y_train, X_test, y_test):
        """Train linear probe on SAE features"""
        print("\n" + "="*60)
        print("Training Linear Probe on SAE Features")
        print("="*60)
        
        # Encode through SAE
        train_features = self.encode_data(X_train)
        test_features = self.encode_data(X_test)
        
        # Train probe
        self.probe = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        )
        self.probe.fit(train_features.cpu().numpy(), y_train)
        top_features_per_class=self.get_top_features_per_class()

        # Evaluate
        train_acc = self.probe.score(train_features.cpu().numpy(), y_train)
        test_acc = self.probe.score(test_features.cpu().numpy(), y_test)
        
        print(f"\nProbe Performance:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        return train_features, test_features, train_acc, test_acc
    
    def get_predictions_and_features(self, X, y):
        """Get predictions, confidences, and features for all samples"""
        features = self.encode_data(X)
        predictions = self.probe.predict(features.cpu().numpy())
        probabilities = self.probe.predict_proba(features.cpu().numpy())
        
        # Get confidence for predicted class
        confidences = np.max(probabilities, axis=1)
        
        # Separate correct and incorrect
        correct_mask = predictions == y
        
        results = {
            'features': features,
            'predictions': predictions,
            'probabilities': probabilities,
            'confidences': confidences,
            'correct_mask': correct_mask,
            'correct_indices': np.where(correct_mask)[0],
            'incorrect_indices': np.where(~correct_mask)[0]
        }
        
        return results
    
    def ablate_feature_and_predict(self, features, feature_idx):
        """Ablate a single feature and get new predictions"""
        ablated_features = features.clone()
        ablated_features[:, feature_idx] = 0
        
        predictions = self.probe.predict(ablated_features.cpu().numpy())
        probabilities = self.probe.predict_proba(ablated_features.cpu().numpy())
        
        return predictions, probabilities
    
    def  compute_feature_attribution(self, sample_features, sample_idx, true_label, predicted_label):
        """
        Compute feature attribution via ablation for a single sample
        Shows which features support correct vs incorrect prediction
        """
        n_features = sample_features.shape[0]
        
        # Baseline prediction
        baseline_probs = self.probe.predict_proba(
            sample_features.cpu().numpy().reshape(1, -1)
        )[0]
        
        true_idx = list(self.probe.classes_).index(true_label)
        pred_idx = list(self.probe.classes_).index(predicted_label)
        
        attributions = []
        
        for feat_idx in range(n_features):
            # Skip if feature is not active
            if sample_features[feat_idx].item() < 0.01:
                continue
                
            # Ablate feature
            ablated = sample_features.clone()
            ablated[feat_idx] = 0
            
            ablated_probs = self.probe.predict_proba(
                ablated.cpu().numpy().reshape(1, -1)
            )[0]
            
            # Change in probability for true class
            true_class_change = baseline_probs[true_idx] - ablated_probs[true_idx]
            
            # Change in probability for predicted class
            pred_class_change = baseline_probs[pred_idx] - ablated_probs[pred_idx]
            
            attributions.append({
                'feature_idx': feat_idx,
                'activation': sample_features[feat_idx].item(),
                'true_class_attribution': true_class_change,
                'pred_class_attribution': pred_class_change,
                'supports_correct': true_class_change > 0,
                'supports_incorrect': (predicted_label != true_label) and (pred_class_change > 0)
            })
        
        return attributions
    
    def analyze_correct_vs_incorrect_features(self, X, y, top_k=20, n_samples_per_category=50):
        """
        Main analysis: Compare features in correct vs incorrect predictions
        """
        print("\n" + "="*60)
        print("Analyzing Features: Correct vs Incorrect Classifications")
        print("="*60)
        
        results = self.get_predictions_and_features(X, y)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(y)}")
        print(f"  Correct predictions: {len(results['correct_indices'])} ({len(results['correct_indices'])/len(y)*100:.1f}%)")
        print(f"  Incorrect predictions: {len(results['incorrect_indices'])} ({len(results['incorrect_indices'])/len(y)*100:.1f}%)")
        
        # Sample from correct and incorrect
        n_correct_samples = min(n_samples_per_category, len(results['correct_indices']))
        n_incorrect_samples = min(n_samples_per_category, len(results['incorrect_indices']))
        
        correct_sample_indices = np.random.choice(
            results['correct_indices'], 
            n_correct_samples, 
            replace=False
        )
        incorrect_sample_indices = np.random.choice(
            results['incorrect_indices'], 
            n_incorrect_samples, 
            replace=False
        )
        
        print(f"\nAnalyzing {n_correct_samples} correct and {n_incorrect_samples} incorrect predictions...")
        
        # Aggregate attributions
        correct_attributions = defaultdict(lambda: {
            'total_attribution': 0.0,
            'count': 0,
            'mean_activation': 0.0,
            'activations': []
        })
        
        incorrect_attributions = defaultdict(lambda: {
            'total_attribution': 0.0,
            'count': 0,
            'mean_activation': 0.0,
            'activations': [],
            'supports_wrong': 0,
            'supports_right': 0
        })
        
        # Analyze correct predictions
        print("\n  Processing correct predictions...")
        for idx in correct_sample_indices:
            sample_attrs = self.compute_feature_attribution(
                results['features'][idx],
                idx,
                y[idx],
                results['predictions'][idx]
            )
            
            for attr in sample_attrs:
                feat_idx = attr['feature_idx']
                correct_attributions[feat_idx]['total_attribution'] += attr['true_class_attribution']
                correct_attributions[feat_idx]['count'] += 1
                correct_attributions[feat_idx]['activations'].append(attr['activation'])
        
        # Analyze incorrect predictions
        print("  Processing incorrect predictions...")
        for idx in incorrect_sample_indices:
            sample_attrs = self.compute_feature_attribution(
                results['features'][idx],
                idx,
                y[idx],
                results['predictions'][idx]
            )
            
            for attr in sample_attrs:
                feat_idx = attr['feature_idx']
                incorrect_attributions[feat_idx]['total_attribution'] += attr['true_class_attribution']
                incorrect_attributions[feat_idx]['count'] += 1
                incorrect_attributions[feat_idx]['activations'].append(attr['activation'])
                
                if attr['supports_incorrect']:
                    incorrect_attributions[feat_idx]['supports_wrong'] += 1
                if attr['supports_correct']:
                    incorrect_attributions[feat_idx]['supports_right'] += 1
        
        # Compute means
        for feat_idx in correct_attributions:
            attrs = correct_attributions[feat_idx]
            attrs['mean_attribution'] = attrs['total_attribution'] / attrs['count']
            attrs['mean_activation'] = np.mean(attrs['activations'])
        
        for feat_idx in incorrect_attributions:
            attrs = incorrect_attributions[feat_idx]
            attrs['mean_attribution'] = attrs['total_attribution'] / attrs['count']
            attrs['mean_activation'] = np.mean(attrs['activations'])
        
        # Find top features for correct predictions
        correct_features_ranked = sorted(
            correct_attributions.items(),
            key=lambda x: x[1]['mean_attribution'],
            reverse=True
        )[:top_k]
        
        # Find features that support incorrect predictions
        incorrect_features_ranked = sorted(
            incorrect_attributions.items(),
            key=lambda x: x[1]['supports_wrong'],
            reverse=True
        )[:top_k]
        
        # Find features that could have helped (high attribution for true class in incorrect predictions)
        could_help_features = sorted(
            incorrect_attributions.items(),
            key=lambda x: x[1]['mean_attribution'],
            reverse=True
        )[:top_k]
        
        return {
            'results': results,
            'correct_attributions': correct_attributions,
            'incorrect_attributions': incorrect_attributions,
            'top_correct_features': correct_features_ranked,
            'top_incorrect_features': incorrect_features_ranked,
            'could_help_features': could_help_features,
            'n_correct_samples': n_correct_samples,
            'n_incorrect_samples': n_incorrect_samples
        }
    
    def print_feature_analysis(self, analysis_results, top_k=10):
        """Print detailed feature analysis results"""
        print("\n" + "="*60)
        print("FEATURE ANALYSIS RESULTS")
        print("="*60)
        
        print("\n" + "-"*60)
        print(f"TOP {top_k} FEATURES DRIVING CORRECT CLASSIFICATIONS")
        print("-"*60)
        print(f"{'Feat':<6} {'Activation':<12} {'Attribution':<12} {'Frequency':<10}")
        print("-"*60)
        
        for feat_idx, attrs in analysis_results['top_correct_features'][:top_k]:
            print(f"{feat_idx:<6} {attrs['mean_activation']:<12.4f} "
                  f"{attrs['mean_attribution']:<12.4f} {attrs['count']:<10}")
        
        print("\n" + "-"*60)
        print(f"TOP {top_k} FEATURES DRIVING INCORRECT CLASSIFICATIONS")
        print("-"*60)
        print(f"{'Feat':<6} {'Activation':<12} {'Supports Wrong':<15} {'Frequency':<10}")
        print("-"*60)
        
        for feat_idx, attrs in analysis_results['top_incorrect_features'][:top_k]:
            print(f"{feat_idx:<6} {attrs['mean_activation']:<12.4f} "
                  f"{attrs['supports_wrong']:<15} {attrs['count']:<10}")
        
        print("\n" + "-"*60)
        print(f"TOP {top_k} FEATURES THAT COULD HAVE HELPED (Missing in Incorrect)")
        print("-"*60)
        print(f"{'Feat':<6} {'Activation':<12} {'Attribution':<12} {'Could Help':<12}")
        print("-"*60)
        
        for feat_idx, attrs in analysis_results['could_help_features'][:top_k]:
            print(f"{feat_idx:<6} {attrs['mean_activation']:<12.4f} "
                  f"{attrs['mean_attribution']:<12.4f} {attrs['supports_right']:<12}")
    
    def visualize_feature_comparison(self, analysis_results, save_path='feature_analysis.png'):
        """Create visualizations comparing correct vs incorrect features"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top features for correct predictions
        ax = axes[0, 0]
        correct_feats = analysis_results['top_correct_features'][:15]
        feat_indices = [f[0] for f in correct_feats]
        attributions = [f[1]['mean_attribution'] for f in correct_feats]
        
        ax.barh(range(len(feat_indices)), attributions, color='green', alpha=0.7)
        ax.set_yticks(range(len(feat_indices)))
        ax.set_yticklabels([f'F{idx}' for idx in feat_indices])
        ax.set_xlabel('Mean Attribution')
        ax.set_title('Top Features Driving Correct Classifications')
        ax.invert_yaxis()
        
        # 2. Features supporting wrong predictions
        ax = axes[0, 1]
        incorrect_feats = analysis_results['top_incorrect_features'][:15]
        feat_indices = [f[0] for f in incorrect_feats]
        wrong_support = [f[1]['supports_wrong'] for f in incorrect_feats]
        
        ax.barh(range(len(feat_indices)), wrong_support, color='red', alpha=0.7)
        ax.set_yticks(range(len(feat_indices)))
        ax.set_yticklabels([f'F{idx}' for idx in feat_indices])
        ax.set_xlabel('Count Supporting Wrong Prediction')
        ax.set_title('Features Driving Incorrect Classifications')
        ax.invert_yaxis()
        
        # 3. Feature activation comparison
        ax = axes[1, 0]
        
        # Get common features
        correct_attrs = analysis_results['correct_attributions']
        incorrect_attrs = analysis_results['incorrect_attributions']
        
        common_features = set(correct_attrs.keys()) & set(incorrect_attrs.keys())
        common_features = sorted(common_features)[:30]
        
        correct_activations = [correct_attrs[f]['mean_activation'] for f in common_features]
        incorrect_activations = [incorrect_attrs[f]['mean_activation'] for f in common_features]
        
        x = np.arange(len(common_features))
        width = 0.35
        
        ax.bar(x - width/2, correct_activations, width, label='Correct', alpha=0.7, color='green')
        ax.bar(x + width/2, incorrect_activations, width, label='Incorrect', alpha=0.7, color='red')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Activation')
        ax.set_title('Feature Activation: Correct vs Incorrect')
        ax.legend()
        ax.set_xticks(x[::3])
        ax.set_xticklabels([f'F{f}' for f in common_features[::3]], rotation=45)
        
        # 4. Attribution scatter
        ax = axes[1, 1]
        
        correct_attrs_list = [(f, attrs['mean_attribution']) for f, attrs in correct_attrs.items()]
        incorrect_attrs_list = [(f, attrs['mean_attribution']) for f, attrs in incorrect_attrs.items()]
        
        # Match features
        correct_dict = dict(correct_attrs_list)
        incorrect_dict = dict(incorrect_attrs_list)
        
        common = set(correct_dict.keys()) & set(incorrect_dict.keys())
        
        x_vals = [correct_dict[f] for f in common]
        y_vals = [incorrect_dict[f] for f in common]
        
        ax.scatter(x_vals, y_vals, alpha=0.5)
        ax.plot([min(x_vals), max(x_vals)], [min(x_vals), max(x_vals)], 
                'r--', alpha=0.5, label='x=y')
        ax.set_xlabel('Attribution in Correct Predictions')
        ax.set_ylabel('Attribution in Incorrect Predictions')
        ax.set_title('Feature Attribution Comparison')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        
        return fig
    
    def analyze_specific_errors(self, X, y, analysis_results, n_examples=5):
        """
        Analyze specific incorrect predictions in detail
        """
        print("\n" + "="*60)
        print("DETAILED ANALYSIS OF INCORRECT PREDICTIONS")
        print("="*60)
        
        results = analysis_results['results']
        incorrect_indices = results['incorrect_indices'][:n_examples]
        
        for i, idx in enumerate(incorrect_indices, 1):
            print(f"\n{'='*60}")
            print(f"Example {i}/{n_examples} - Sample Index: {idx}")
            print(f"{'='*60}")
            
            true_label = y[idx]
            pred_label = results['predictions'][idx]
            confidence = results['confidences'][idx]
            
            print(f"\nTrue Label: {true_label}")
            print(f"Predicted Label: {pred_label}")
            print(f"Confidence: {confidence:.4f}")
            
            # Get feature attributions
            sample_attrs = self.compute_feature_attribution(
                results['features'][idx],
                idx,
                true_label,
                pred_label
            )
            
            # Sort by activation
            sample_attrs.sort(key=lambda x: abs(x['pred_class_attribution']), reverse=True)
            
            print(f"\nTop features supporting WRONG prediction:")
            print(f"{'Feature':<10} {'Activation':<12} {'Wrong Support':<15}")
            print("-"*40)
            
            wrong_support = [a for a in sample_attrs if a['supports_incorrect']][:5]
            for attr in wrong_support:
                print(f"{attr['feature_idx']:<10} {attr['activation']:<12.4f} "
                      f"{attr['pred_class_attribution']:<15.4f}")
            
            print(f"\nTop features supporting CORRECT prediction (but weak):")
            print(f"{'Feature':<10} {'Activation':<12} {'Right Support':<15}")
            print("-"*40)
            
            right_support = sorted(
                [a for a in sample_attrs if a['supports_correct']], 
                key=lambda x: x['true_class_attribution'],
                reverse=True
            )[:5]
            
            for attr in right_support:
                print(f"{attr['feature_idx']:<10} {attr['activation']:<12.4f} "
                      f"{attr['true_class_attribution']:<15.4f}")
    
    def create_feature_summary_table(self, analysis_results, save_path='feature_summary.csv'):
        """Create a comprehensive summary table of all features"""
        
        correct_attrs = analysis_results['correct_attributions']
        incorrect_attrs = analysis_results['incorrect_attributions']
        
        all_features = set(correct_attrs.keys()) | set(incorrect_attrs.keys())
        
        summary_data = []
        
        for feat_idx in sorted(all_features):
            row = {
                'feature_idx': feat_idx,
                'correct_count': correct_attrs.get(feat_idx, {}).get('count', 0),
                'correct_mean_activation': correct_attrs.get(feat_idx, {}).get('mean_activation', 0),
                'correct_mean_attribution': correct_attrs.get(feat_idx, {}).get('mean_attribution', 0),
                'incorrect_count': incorrect_attrs.get(feat_idx, {}).get('count', 0),
                'incorrect_mean_activation': incorrect_attrs.get(feat_idx, {}).get('mean_activation', 0),
                'incorrect_mean_attribution': incorrect_attrs.get(feat_idx, {}).get('mean_attribution', 0),
                'supports_wrong': incorrect_attrs.get(feat_idx, {}).get('supports_wrong', 0),
                'supports_right': incorrect_attrs.get(feat_idx, {}).get('supports_right', 0),
            }
            
            # Compute ratio
            if row['incorrect_count'] > 0:
                row['correct_incorrect_ratio'] = row['correct_count'] / row['incorrect_count']
            else:
                row['correct_incorrect_ratio'] = float('inf') if row['correct_count'] > 0 else 0
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('correct_mean_attribution', ascending=False)
        df.to_csv(save_path, index=False)
        
        print(f"\nFeature summary table saved to: {save_path}")
        
        return df


def run_complete_interpretation_pipeline(
    class_level,
    layer,
    sae_model,
    X_train, y_train,
    X_test, y_test,
    output_dir='sae_interpretation_results',
    device='cpu'
):
    """
    Run the complete SAE interpretation pipeline
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(" "*20 + "FEATURE INTERPRETATION PIPELINE")
    print("="*70)

    # Initialize interpreter
    interpreter = SAEFeatureInterpreter(sae_model, device=device)
    
    # Train probe
    train_features, test_features, train_acc, test_acc = interpreter.train_probe(
        X_train, y_train, X_test, y_test
    )

    
    # Analyze correct vs incorrect features
    analysis_results = interpreter.analyze_correct_vs_incorrect_features(
        X_test, y_test,
        top_k=20,
        n_samples_per_category=100
    )
    
    # Print results
    interpreter.print_feature_analysis(analysis_results, top_k=15)

    # Print top features
    print("="*70)
    print("Get top features")
    top_features = interpreter.get_top_features_per_class(top_k=10)
    print("="*70)
    # Print results
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES PER EC CLASS")
    print("=" * 60)

    for ec_class, features in top_features.items():
        print(f"\nEC Class: {ec_class}")
        print(f"{'Feature':<10} {'Coefficient':<15}")
        print("-" * 25)
        for feat_idx, coef in features:
            print(f"{feat_idx:<10} {coef:>14.4f}")


    train_data, test_data = get_dataset()
    # Extract everything from test set
    protein_ids_all = np.array(test_data['id'])
    sequences_all = test_data['seq']
    labels_all = test_data['labels']  # or test_data['labels_str']

    # Create dictionaries for easy lookup
    sequence_dict = {pid: seq for pid, seq in zip(protein_ids_all, sequences_all)}
    metadata_dict = {pid: {'ec_number': label, 'protein_id': pid}
                     for pid, label in zip(protein_ids_all, labels_all)}

    print(f"Loaded {len(protein_ids_all)} proteins from test set")
    print(f"Example protein ID: {protein_ids_all[0]}")
    print(f"Example sequence length: {len(sequences_all[0])}")


    feature_analysis = interpreter.analyze_top_features_with_proteins(
        X_test,
        protein_ids_all,  # You need to pass this
        top_features,
        top_k_features=5
    )

    print(f"\nTotal features analyzed: {len(feature_analysis)}")



    # Visualize
    fig = interpreter.visualize_feature_comparison(
        analysis_results,
        save_path=os.path.join(output_dir,  f'feature_ClassLevel_{class_level}_Layer_{layer}_comparison.png')
    )
    
    # Analyze specific errors
    interpreter.analyze_specific_errors(X_test, y_test, analysis_results, n_examples=5)
    
    # Create summary table
    summary_df = interpreter.create_feature_summary_table(
        analysis_results,
        save_path=os.path.join(output_dir, 'feature_summary.csv')
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    return {
        'interpreter': interpreter,
        'analysis_results': analysis_results,
        'summary_df': summary_df,
        'test_accuracy': test_acc
    }
