"""
Feature Method Comparison Framework
Compares SAE features against baselines: Raw, PCA, NMF
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os


class FeatureMethodComparison:
    """
    Compare different feature extraction methods for protein representations

    Methods:
    - Raw: Original layer representations (no transformation)
    - PCA: Principal Component Analysis
    - NMF: Non-negative Matrix Factorization
    - SAE: Sparse Autoencoder

    Metrics:
    - Sparsity: How sparse are the features?
    - Specialization: How class-specific are features?
    - Classification: Downstream task performance
    """

    def __init__(self, layer_representations, ec_labels, n_components=None,  # None = auto
                 random_state=1829873, protein_ids=None):
        """
        Initialize comparison framework

        Args:
            layer_representations: (n_samples, hidden_dim) - raw ESM-2 layer outputs
            ec_labels: (n_samples,) - EC class labels
            n_components: feature dimension for PCA/NMF/SAE (None = auto-determine)
            random_state: random seed for reproducibility
            protein_ids: (n_samples,) - protein identifiers (optional, for structure analysis)
        """
        self.X = layer_representations
        self.y = ec_labels
        self.protein_ids = protein_ids
        self.random_state = random_state

        # Auto-determine n_components if not provided
        if n_components is None:
            # Use 2x input dimension as default (common practice for SAEs)
            n_components = self.X.shape[1] * 2
            print(f"  Auto-setting n_components={n_components} (2x input dimension)")

        # Cap at maximum possible for PCA/NMF
        max_components = min(self.X.shape[0], self.X.shape[1])
        if n_components > max_components:
            print(f"  WARNING: Requested n_components={n_components} > max possible ({max_components})")
            print(f"  For PCA/NMF, will use n_components={max_components}")
            print(f"  For SAE, will use requested n_components={n_components}")
            self.n_components_baseline = max_components  # For PCA/NMF
            self.n_components = n_components  # For SAE (can exceed input dim)
        else:
            self.n_components = n_components
            self.n_components_baseline = n_components

        print(f"Initialized comparison with:")
        print(f"  Samples: {self.X.shape[0]}")
        print(f"  Input dimension: {self.X.shape[1]}")
        print(f"  Output dimension (SAE): {self.n_components}")
        print(f"  Output dimension (PCA/NMF): {self.n_components_baseline}")
        print(f"  Classes: {len(np.unique(self.y))}")

    def extract_features(self, method_name, **method_kwargs):
        """
        Extract features using specified method
        """

        if method_name == 'raw':
            print("  Using raw representations (no transformation)")
            return self.X, None

        elif method_name == 'pca':
            print(f"  Fitting PCA with {self.n_components_baseline} components...")
            pca = PCA(n_components=self.n_components_baseline, random_state=self.random_state)
            features = pca.fit_transform(self.X)
            print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
            return features, pca

        elif method_name == 'nmf':
            print(f"  Fitting NMF with {self.n_components_baseline} components...")
            # NMF requires non-negative inputs
            X_nonneg = self.X - self.X.min() + 1e-10
            nmf = NMF(
                n_components=self.n_components_baseline,
                random_state=self.random_state,
                max_iter=500,
                init='random'
            )
            features = nmf.fit_transform(X_nonneg)
            print(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")
            return features, nmf

        elif method_name == 'sae':
            print(f"  Using provided SAE model...")
            sae = method_kwargs.get('sae_model')
            if sae is None:
                raise ValueError("Must provide trained SAE model via sae_model=...")

            # Assume SAE has an encode method
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sae.to(device)
            sae.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(self.X).to(device)
                features = sae.encode(X_tensor).cpu().numpy()

            return features, sae

        else:
            raise ValueError(f"Unknown method: {method_name}")

    def diagnose_sae(self, sae_model, features):
        """
        Diagnose what's wrong with SAE
        """
        print("\n" + "=" * 60)
        print("SAE DIAGNOSTICS")
        print("=" * 60)

        # 1. Check reconstruction quality
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sae_model.to(device)
        sae_model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X[:100]).to(device)
            output = sae_model(X_tensor)

            # FIX: Handle both tuple and tensor returns
            if isinstance(output, tuple):
                reconstructed = output[0].cpu().numpy()  # First element is reconstruction
            else:
                reconstructed = output.cpu().numpy()

            # Reconstruction error
            mse = np.mean((self.X[:100] - reconstructed) ** 2)
            relative_error = mse / np.var(self.X[:100])

            print(f"\n1. Reconstruction Quality:")
            print(f"   MSE: {mse:.4f}")
            print(f"   Relative error: {relative_error:.4f}")
            print(f"   {'✓ GOOD' if relative_error < 0.1 else '✗ BAD'} (should be < 0.1)")

        # 2. Check feature activation statistics
        print(f"\n2. Feature Activation Stats:")
        print(f"   Mean activation: {np.abs(features).mean():.4f}")
        print(f"   Max activation: {np.abs(features).max():.4f}")
        print(f"   Fraction > 0.01: {(np.abs(features) > 0.01).mean():.4f}")
        print(f"   Fraction > 0.1: {(np.abs(features) > 0.1).mean():.4f}")

        # 3. Check for dead features
        active_per_feature = (np.abs(features) > 0.01).mean(axis=0)
        dead_features = (active_per_feature < 0.001).sum()

        print(f"\n3. Dead Features:")
        print(f"   Dead features: {dead_features}/{features.shape[1]} ({100 * dead_features / features.shape[1]:.1f}%)")
        print(f"   {'✓ GOOD' if dead_features < features.shape[1] * 0.1 else '✗ BAD'} (should be < 10%)")

        # 4. Check feature variance
        feature_vars = np.var(features, axis=0)
        low_var = (feature_vars < 0.01).sum()

        print(f"\n4. Feature Variance:")
        print(f"   Mean variance: {feature_vars.mean():.4f}")
        print(f"   Low variance features: {low_var}/{features.shape[1]} ({100 * low_var / features.shape[1]:.1f}%)")

        # 5. Check if features are just noise
        from scipy.stats import spearmanr

        # Compute pairwise correlations (sample 100 features)
        n_sample = min(100, features.shape[1])
        sample_features = features[:, :n_sample]
        correlations = []
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                if np.std(sample_features[:, i]) > 0 and np.std(sample_features[:, j]) > 0:
                    corr, _ = spearmanr(sample_features[:, i], sample_features[:, j])
                    correlations.append(abs(corr))

        if correlations:
            print(f"\n5. Feature Independence:")
            print(f"   Mean |correlation|: {np.mean(correlations):.4f}")
            print(f"   {'✓ GOOD' if np.mean(correlations) < 0.3 else '✗ BAD'} (should be < 0.3)")

        print("\n" + "=" * 60)

    def compute_sparsity(self, features, threshold=0.1):
        """
        Compute sparsity metrics

        Args:
            features: (n_samples, n_features) feature matrix
            threshold: activation threshold for L0 sparsity

        Returns:
            dict with sparsity metrics
        """
        # L0 sparsity: fraction of activations above threshold
        l0_sparsity = (np.abs(features) > threshold).mean()

        # L1/L2 ratio (Hoyer sparsity measure)
        l1_norm = np.abs(features).sum(axis=1)
        l2_norm = np.linalg.norm(features, axis=1)
        hoyer_sparsity = (np.sqrt(features.shape[1]) - (l1_norm / l2_norm)) / (np.sqrt(features.shape[1]) - 1)

        return {
            'l0_sparsity': l0_sparsity,
            'hoyer_sparsity': hoyer_sparsity.mean(),
            'mean_activation': np.abs(features).mean(),
            'median_activation': np.median(np.abs(features)),
            'max_activation': np.abs(features).max(),
            'std_activation': np.abs(features).std()
        }

    def compute_specialization(self, features):
        """
        Compute how class-specific features are

        Lower entropy = more specialized to specific classes

        Args:
            features: (n_samples, n_features) feature matrix

        Returns:
            dict with specialization metrics
        """
        specialization_scores = []
        unique_classes = np.unique(self.y)
        n_classes = len(unique_classes)

        for feat_idx in range(features.shape[1]):
            # Distribution of feature activation across classes
            activations_per_class = []

            for class_label in unique_classes:
                mask = (self.y == class_label)
                mean_act = np.abs(features[mask, feat_idx]).mean()
                activations_per_class.append(mean_act)

            # Normalize to get probability distribution
            activations_per_class = np.array(activations_per_class)

            if activations_per_class.sum() > 0:
                prob_dist = activations_per_class / activations_per_class.sum()

                # Entropy (normalized by max possible entropy)
                feat_entropy = entropy(prob_dist + 1e-10) / np.log(n_classes)

                # Specialization = 1 - normalized_entropy
                # High specialization (close to 1) = feature specific to few classes
                # Low specialization (close to 0) = feature activates uniformly across classes
                specialization = 1 - feat_entropy
            else:
                specialization = 0  # dead feature

            specialization_scores.append(specialization)

        specialization_scores = np.array(specialization_scores)

        return {
            'mean_specialization': specialization_scores.mean(),
            'median_specialization': np.median(specialization_scores),
            'std_specialization': specialization_scores.std(),
            'highly_specialized': (specialization_scores > 0.8).sum(),
            'moderately_specialized': ((specialization_scores > 0.5) & (specialization_scores <= 0.8)).sum(),
            'general_features': (specialization_scores <= 0.5).sum()
        }

    def compute_monosemanticity(self, features, k=100):
        """
        Compute monosemanticity: does each feature activate for one semantic concept?

        Args:
            features: (n_samples, n_features) feature matrix
            k: number of top activating samples to consider

        Returns:
            dict with monosemanticity metrics
        """
        monosemanticity_scores = []

        for feat_idx in range(features.shape[1]):
            activations = features[:, feat_idx]

            # Get top-k activating samples
            if len(activations) < k:
                top_k_indices = np.argsort(activations)[::-1]
            else:
                top_k_indices = np.argsort(activations)[-k:]

            # Check EC class agreement among top-k
            top_k_labels = self.y[top_k_indices]
            unique_labels = np.unique(top_k_labels)

            # Monosemantic = activates for few classes
            # Polysemantic = activates for many unrelated classes
            label_diversity = len(unique_labels) / min(k, len(activations))
            monosemanticity = 1 - label_diversity

            monosemanticity_scores.append(monosemanticity)

        monosemanticity_scores = np.array(monosemanticity_scores)

        return {
            'mean_monosemanticity': monosemanticity_scores.mean(),
            'median_monosemanticity': np.median(monosemanticity_scores),
            'highly_monosemantic': (monosemanticity_scores > 0.7).sum()
        }

    def compute_classification_metrics(self, features):
        """
        Train linear probe and compute classification performance

        Args:
            features: (n_samples, n_features) feature matrix

        Returns:
            dict with classification metrics
        """
        # Try stratified split, fall back to random if fails
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, self.y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=self.y
            )
        except ValueError:
            print("  Warning: Stratified split failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                features, self.y,
                test_size=0.2,
                random_state=self.random_state
            )

        # Train linear probe
        probe = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'  # handle class imbalance
        )
        probe.fit(X_train, y_train)

        # Predictions
        y_pred_train = probe.predict(X_train)
        y_pred_test = probe.predict(X_test)

        return {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'f1_macro': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'overfitting_gap': accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test)
        }

    def compute_sample_efficiency(self, features, train_sizes=[50, 100, 200, 500]):
        """
        Test how many samples are needed to reach good performance

        Args:
            features: (n_samples, n_features) feature matrix
            train_sizes: list of training set sizes to test

        Returns:
            dict with sample efficiency metrics
        """
        # Filter train_sizes that are feasible
        max_train = int(0.8 * len(self.y))
        train_sizes = [s for s in train_sizes if s < max_train]

        if not train_sizes:
            return {'learning_curve': []}

        # Fixed test set
        X_full_train, X_test, y_full_train, y_test = train_test_split(
            features, self.y,
            test_size=0.2,
            random_state=self.random_state
        )

        learning_curve = []

        for n in train_sizes:
            # Subsample training data
            if n >= len(y_full_train):
                continue

            indices = np.random.RandomState(self.random_state).choice(
                len(y_full_train), size=n, replace=False
            )
            X_train = X_full_train[indices]
            y_train = y_full_train[indices]

            # Train and evaluate
            probe = LogisticRegression(max_iter=1000, random_state=self.random_state)
            probe.fit(X_train, y_train)
            acc = probe.score(X_test, y_test)

            learning_curve.append({'train_size': n, 'accuracy': acc})

        return {
            'learning_curve': learning_curve,
            'sample_efficiency_score': np.mean([d['accuracy'] for d in learning_curve]) if learning_curve else 0
        }

    def compare_all_methods(self, sae_model=None, include_sample_efficiency=False):
        """
        Run comparison across all methods

        Args:
            sae_model: trained SAE model (optional)
            include_sample_efficiency: whether to run sample efficiency tests (slower)

        Returns:
            pandas DataFrame with results
        """
        results = {}

        # Define methods to compare
        methods = ['raw', 'pca', 'nmf']
        if sae_model is not None:
            methods.append('sae')

        for method_name in methods:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {method_name.upper()}")
            print('=' * 60)

            try:
                # Extract features
                if method_name == 'sae':
                    features, model = self.extract_features(method_name, sae_model=sae_model)
                    self.diagnose_sae(sae_model, features)
                elif method_name == 'raw':
                    features = self.extract_features(method_name)[0]
                    model = None
                else:
                    features, model = self.extract_features(method_name)

                print(f"Feature shape: {features.shape}")

                # Compute metrics
                print("\n1. Computing sparsity metrics...")
                sparsity = self.compute_sparsity(features)

                print("2. Computing specialization metrics...")
                specialization = self.compute_specialization(features)

                print("3. Computing monosemanticity metrics...")
                monosemanticity = self.compute_monosemanticity(features)

                print("4. Computing classification metrics...")
                classification = self.compute_classification_metrics(features)

                # Optional: sample efficiency (slower)
                if include_sample_efficiency:
                    print("5. Computing sample efficiency...")
                    sample_eff = self.compute_sample_efficiency(features)
                else:
                    sample_eff = {}

                # Store results
                results[method_name] = {
                    **sparsity,
                    **specialization,
                    **monosemanticity,
                    **classification,
                    **sample_eff
                }

                # Print summary
                print(f"\n{'─' * 60}")
                print(f"SUMMARY FOR {method_name.upper()}:")
                print(f"{'─' * 60}")
                print(f"  Sparsity (L0):        {sparsity['l0_sparsity']:.4f}")
                print(f"  Specialization:       {specialization['mean_specialization']:.4f}")
                print(f"  Monosemanticity:      {monosemanticity['mean_monosemanticity']:.4f}")
                print(f"  Test Accuracy:        {classification['test_accuracy']:.4f}")
                print(f"  F1 (macro):           {classification['f1_macro']:.4f}")
                print(f"  Overfitting Gap:      {classification['overfitting_gap']:.4f}")

            except Exception as e:
                print(f"ERROR processing {method_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(results).T

        return results_df

    def plot_comparison(self, results_df, save_path):
        """
        Create visualization of comparison results

        Args:
            results_df: DataFrame from compare_all_methods
            save_path: path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics = [
            ('l0_sparsity', 'L0 Sparsity\n(lower = sparser)', True),
            ('mean_specialization', 'Feature Specialization\n(higher = better)', False),
            ('mean_monosemanticity', 'Monosemanticity\n(higher = better)', False),
            ('test_accuracy', 'Test Accuracy\n(higher = better)', False),
            ('f1_macro', 'F1 Score (Macro)\n(higher = better)', False),
            ('overfitting_gap', 'Overfitting Gap\n(lower = better)', True)
        ]

        for idx, (metric, title, lower_is_better) in enumerate(metrics):
            ax = axes[idx]

            if metric not in results_df.columns:
                ax.text(0.5, 0.5, f'{metric}\nnot available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=11, fontweight='bold')
                continue

            # Plot bars
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            results_df[metric].plot(kind='bar', ax=ax, color=colors[:len(results_df)])

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_xlabel('')

            # Rotate x labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add value labels on bars
            for i, v in enumerate(results_df[metric]):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

            # Highlight best method
            if lower_is_better:
                best_idx = results_df[metric].idxmin()
            else:
                best_idx = results_df[metric].idxmax()

            best_pos = list(results_df.index).index(best_idx)
            ax.get_children()[best_pos].set_edgecolor('red')
            ax.get_children()[best_pos].set_linewidth(2)

        plt.tight_layout()

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nComparison plot saved to: {save_path}")

    def save_results(self, results_df, save_dir, level_name):
        """
        Save comparison results to CSV and create plots

        Args:
            results_df: DataFrame from compare_all_methods
            save_dir: directory to save results
            level_name: name for this comparison (e.g., 'level_1')
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(save_dir, f'comparison_{level_name}.csv')
        results_df.to_csv(csv_path)
        print(f"\nResults saved to: {csv_path}")

        # Create plot
        plot_path = os.path.join(save_dir, f'comparison_{level_name}.png')
        self.plot_comparison(results_df, plot_path)

        # Print summary table
        print(f"\n{'=' * 80}")
        print(f"FINAL COMPARISON RESULTS - {level_name}")
        print(f"{'=' * 80}")

        # Select key metrics for display
        display_metrics = [
            'l0_sparsity',
            'mean_specialization',
            'test_accuracy',
            'f1_macro',
            'overfitting_gap'
        ]

        display_df = results_df[[m for m in display_metrics if m in results_df.columns]]
        print(display_df.to_string())
        print(f"{'=' * 80}\n")