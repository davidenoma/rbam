import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.model_selection import train_test_split


def explain_vae_with_shap_robust(vae_model, X_data, sample_size=100, max_evals=50):
    """
    Robust SHAP analysis for VAE with proper error handling.

    Args:
        vae_model: Trained VAE model
        X_data: Input data for explanation (numpy array)
        sample_size: Number of background samples for SHAP
        max_evals: Maximum evaluations for SHAP

    Returns:
        dict: SHAP analysis results or None if failed
    """
    try:
        print("Initializing robust SHAP analysis for VAE...")

        # Ensure data is numpy array and properly formatted
        if isinstance(X_data, pd.DataFrame):
            X_data = X_data.values
        X_data = np.array(X_data, dtype=np.float32)

        print(f"Data shape: {X_data.shape}")
        print(f"Data type: {X_data.dtype}")

        # Create background dataset with proper indexing
        np.random.seed(42)  # For reproducibility
        n_samples = X_data.shape[0]
        background_size = min(sample_size, n_samples)
        explain_size = min(20, n_samples)  # Reduced for stability

        # Use proper sampling
        background_indices = np.random.choice(n_samples, size=background_size, replace=False)
        background_data = X_data[background_indices].copy()

        explain_indices = np.random.choice(n_samples, size=explain_size, replace=False)
        explain_data = X_data[explain_indices].copy()

        print(f"Background data shape: {background_data.shape}")
        print(f"Explain data shape: {explain_data.shape}")

        # Create wrapper functions for SHAP analysis
        def encoder_mean_output(x):
            """Wrapper to get encoder mean output"""
            try:
                # Ensure input is properly formatted
                x = tf.cast(x, tf.float32)
                encoded = vae_model.encoder(x, training=False)
                z_mean, _ = tf.split(encoded, num_or_size_splits=2, axis=1)
                return z_mean
            except Exception as e:
                print(f"Error in encoder_mean_output: {e}")
                raise

        def reconstruction_output(x):
            """Wrapper to get full reconstruction"""
            try:
                x = tf.cast(x, tf.float32)
                return vae_model(x, training=False)
            except Exception as e:
                print(f"Error in reconstruction_output: {e}")
                raise

        # Test the wrapper functions first
        print("Testing wrapper functions...")
        test_output = encoder_mean_output(background_data[:2])
        print(f"Encoder test output shape: {test_output.shape}")

        # SHAP analysis for encoder (latent space construction)
        print("Analyzing encoder (latent space construction)...")
        encoder_explainer = shap.DeepExplainer(encoder_mean_output, background_data)
        encoder_shap_values = encoder_explainer.shap_values(explain_data, check_additivity=False)

        print(f"Encoder SHAP values shape: {np.array(encoder_shap_values).shape}")

        return {
            'encoder_shap_values': encoder_shap_values,
            'explain_data': explain_data,
            'background_data': background_data,
            'encoder_explainer': encoder_explainer,
            'success': True
        }

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Attempting alternative analysis...")
        return {'success': False, 'error': str(e)}


def analyze_feature_importance_alternative(vae_model, X_data, num_features=20):
    """
    Alternative feature importance analysis using gradient-based methods.

    Args:
        vae_model: Trained VAE model
        X_data: Input data
        num_features: Number of top features to analyze

    Returns:
        dict: Feature importance analysis results
    """
    print("Performing gradient-based feature importance analysis...")

    try:
        # Convert to tensor if needed
        if not isinstance(X_data, tf.Tensor):
            X_data = tf.constant(X_data, dtype=tf.float32)

        # Sample data for analysis
        sample_size = min(100, X_data.shape[0])
        sample_indices = np.random.choice(X_data.shape[0], sample_size, replace=False)
        sample_data = tf.gather(X_data, sample_indices)

        # Calculate gradients for each latent dimension
        with tf.GradientTape() as tape:
            tape.watch(sample_data)
            # Get encoder output
            encoded = vae_model.encoder(sample_data, training=False)
            z_mean, z_log_var = tf.split(encoded, num_or_size_splits=2, axis=1)

        # Calculate gradients
        gradients = tape.gradient(z_mean, sample_data)

        if gradients is not None:
            # Calculate feature importance as mean absolute gradient
            feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0)

            # Get top features
            top_indices = tf.nn.top_k(feature_importance, k=min(num_features, len(feature_importance))).indices
            top_importance = tf.gather(feature_importance, top_indices)

            return {
                'feature_importance': feature_importance.numpy(),
                'top_indices': top_indices.numpy(),
                'top_importance': top_importance.numpy(),
                'gradients': gradients.numpy(),
                'success': True
            }
        else:
            print("Could not compute gradients")
            return {'success': False}

    except Exception as e:
        print(f"Gradient analysis failed: {e}")
        return {'success': False, 'error': str(e)}


def plot_feature_importance(importance_results, feature_names=None, save_path=None):
    """
    Plot feature importance results.

    Args:
        importance_results: Results from feature importance analysis
        feature_names: Optional feature names
        save_path: Path to save plots
    """
    if not importance_results.get('success', False):
        print("No successful importance results to plot")
        return

    print("Creating feature importance plots...")

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Plot 1: Top feature importance
    top_indices = importance_results['top_indices']
    top_importance = importance_results['top_importance']

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importance_results['feature_importance']))]

    top_features = [feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                    for i in top_indices]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_importance)), top_importance)
    plt.yticks(range(len(top_importance)), top_features)
    plt.xlabel('Feature Importance (Mean Absolute Gradient)')
    plt.title(f'Top {len(top_importance)} Features by Importance for Latent Space Construction')
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "feature_importance_gradient.png"),
                    dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Feature importance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(importance_results['feature_importance'], bins=50, alpha=0.7)
    plt.xlabel('Feature Importance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Importance Scores')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(os.path.join(save_path, "importance_distribution.png"),
                    dpi=300, bbox_inches='tight')
    plt.show()


def analyze_latent_space_interpretability(vae_model, X_data, y_data=None, save_path=None):
    """
    Comprehensive latent space interpretability analysis.

    Args:
        vae_model: Trained VAE model
        X_data: Input data
        y_data: Optional phenotype data for supervised analysis
        save_path: Path to save results

    Returns:
        dict: Latent space analysis results
    """
    print("Starting comprehensive latent space interpretability analysis...")

    try:
        # Extract latent representations
        encoded = vae_model.encoder.predict(X_data, verbose=0)
        z_mean, z_log_var = tf.split(encoded, num_or_size_splits=2, axis=1)
        z_mean = z_mean.numpy()
        z_log_var = z_log_var.numpy()

        latent_dim = z_mean.shape[1]
        print(f"Latent space dimensionality: {latent_dim}")

        # 1. Principal Component Analysis of latent space
        print("Performing PCA on latent space...")
        n_components = min(10, latent_dim)
        pca = PCA(n_components=n_components)
        z_pca = pca.fit_transform(z_mean)

        # 2. Calculate latent dimension statistics
        latent_variances = np.var(z_mean, axis=0)
        latent_means = np.mean(z_mean, axis=0)
        latent_correlations = np.corrcoef(z_mean.T)

        # 3. Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # PCA explained variance
        axes[0, 0].plot(np.cumsum(pca.explained_variance_ratio_))
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Cumulative Explained Variance')
        axes[0, 0].set_title('PCA of Latent Space')
        axes[0, 0].grid(True, alpha=0.3)

        # Latent space visualization colored by phenotype
        if y_data is not None:
            scatter = axes[0, 1].scatter(z_pca[:, 0], z_pca[:, 1], c=y_data, cmap='viridis', alpha=0.6)
            axes[0, 1].set_xlabel('First Principal Component')
            axes[0, 1].set_ylabel('Second Principal Component')
            axes[0, 1].set_title('Latent Space Colored by Phenotype')
            plt.colorbar(scatter, ax=axes[0, 1])
        else:
            axes[0, 1].scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6)
            axes[0, 1].set_xlabel('First Principal Component')
            axes[0, 1].set_ylabel('Second Principal Component')
            axes[0, 1].set_title('Latent Space (First 2 PCs)')

        # Latent dimension variances
        top_var_dims = min(20, len(latent_variances))
        axes[1, 0].bar(range(top_var_dims), latent_variances[:top_var_dims])
        axes[1, 0].set_xlabel('Latent Dimension')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].set_title(f'Variance per Latent Dimension (Top {top_var_dims})')

        # Correlation heatmap
        correlation_subset = latent_correlations[:min(20, latent_dim), :min(20, latent_dim)]
        im = axes[1, 1].imshow(correlation_subset, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Latent Dimension Correlations')
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('Latent Dimension')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "latent_space_analysis.png"),
                       dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Analyze most informative latent dimensions
        print("Analyzing most informative latent dimensions...")

        # Sort dimensions by variance
        var_sorted_indices = np.argsort(latent_variances)[::-1]

        # If we have phenotype data, analyze separation
        separation_scores = None
        if y_data is not None:
            separation_scores = []
            for dim in range(latent_dim):
                # Calculate separation between classes in each dimension
                class_0_values = z_mean[y_data == 0, dim]
                class_1_values = z_mean[y_data == 1, dim]

                if len(class_0_values) > 0 and len(class_1_values) > 0:
                    # Calculate mean difference normalized by pooled std
                    mean_diff = abs(np.mean(class_0_values) - np.mean(class_1_values))
                    pooled_std = np.sqrt((np.var(class_0_values) + np.var(class_1_values)) / 2)
                    separation = mean_diff / (pooled_std + 1e-8)
                else:
                    separation = 0.0

                separation_scores.append(separation)

            separation_scores = np.array(separation_scores)
            sep_sorted_indices = np.argsort(separation_scores)[::-1]

            # Plot top separating dimensions
            plt.figure(figsize=(15, 5))
            for i, dim_idx in enumerate(sep_sorted_indices[:5]):
                plt.subplot(1, 5, i+1)
                plt.hist(z_mean[y_data == 0, dim_idx], alpha=0.7, label='Class 0', bins=30)
                plt.hist(z_mean[y_data == 1, dim_idx], alpha=0.7, label='Class 1', bins=30)
                plt.xlabel(f'Latent Dim {dim_idx}')
                plt.ylabel('Frequency')
                plt.title(f'Separation Score: {separation_scores[dim_idx]:.2f}')
                plt.legend()

            plt.suptitle('Top 5 Most Separating Latent Dimensions')
            plt.tight_layout()
            if save_path:
                plt.savefig(os.path.join(save_path, "latent_separation_analysis.png"),
                           dpi=300, bbox_inches='tight')
            plt.show()

        return {
            'pca': pca,
            'z_pca': z_pca,
            'z_mean': z_mean,
            'z_log_var': z_log_var,
            'latent_variances': latent_variances,
            'latent_means': latent_means,
            'latent_correlations': latent_correlations,
            'var_sorted_indices': var_sorted_indices,
            'separation_scores': separation_scores,
            'success': True
        }

    except Exception as e:
        print(f"Latent space analysis failed: {e}")
        return {'success': False, 'error': str(e)}


def analyze_reconstruction_patterns(vae_model, X_data, save_path=None):
    """
    Analyze reconstruction patterns to understand what the VAE learns.

    Args:
        vae_model: Trained VAE model
        X_data: Input data
        save_path: Path to save plots

    Returns:
        dict: Reconstruction analysis results
    """
    print("Analyzing reconstruction patterns...")

    try:
        # Get reconstructions
        sample_size = min(1000, X_data.shape[0])
        sample_indices = np.random.choice(X_data.shape[0], sample_size, replace=False)
        sample_data = X_data[sample_indices]

        reconstructions = vae_model.predict(sample_data, verbose=0)

        # Calculate reconstruction errors per feature
        reconstruction_errors = np.mean((sample_data - reconstructions) ** 2, axis=0)

        # Plot reconstruction error distribution
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(reconstruction_errors, bins=50, alpha=0.7)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Feature Reconstruction Errors')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        top_error_indices = np.argsort(reconstruction_errors)[-20:]
        plt.barh(range(20), reconstruction_errors[top_error_indices])
        plt.yticks(range(20), [f"Feature_{i}" for i in top_error_indices])
        plt.xlabel('Reconstruction Error')
        plt.title('Top 20 Features with Highest Reconstruction Error')

        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "reconstruction_analysis.png"),
                       dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'reconstruction_errors': reconstruction_errors,
            'high_error_features': top_error_indices,
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'success': True
        }

    except Exception as e:
        print(f"Reconstruction analysis failed: {e}")
        return {'success': False, 'error': str(e)}


def plot_shap_analysis_robust(shap_results, save_path=None):
    """
    Robust SHAP plotting with proper error handling.
    """
    try:
        if not shap_results.get('success', False):
            return

        print("Creating SHAP visualization plots...")

        encoder_shap_values = shap_results['encoder_shap_values']
        explain_data = shap_results['explain_data']

        # Handle different SHAP value formats
        if isinstance(encoder_shap_values, list):
            # Multiple outputs (multiple latent dimensions)
            shap_values_to_plot = encoder_shap_values[0] if len(encoder_shap_values) > 0 else encoder_shap_values
        else:
            shap_values_to_plot = encoder_shap_values

        # Plot 1: Summary plot
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(
                shap_values_to_plot,
                explain_data,
                show=False,
                max_display=20
            )
            plt.title("SHAP Summary: Feature Importance for Latent Space Construction")
            if save_path:
                plt.savefig(os.path.join(save_path, "shap_summary_robust.png"),
                           dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create summary plot: {e}")
            plt.close()

        # Plot 2: Feature importance bar plot
        if len(shap_values_to_plot.shape) >= 2:
            mean_importance = np.mean(np.abs(shap_values_to_plot), axis=0)
            top_indices = np.argsort(mean_importance)[-20:]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_indices)), mean_importance[top_indices])
            plt.yticks(range(len(top_indices)), [f"Feature_{i}" for i in top_indices])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Top 20 Features by SHAP Importance')
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, "shap_feature_importance_robust.png"),
                           dpi=300, bbox_inches='tight')
            plt.show()

    except Exception as e:
        print(f"Error in SHAP plotting: {e}")


def comprehensive_vae_analysis(vae_model, X_data, y_data=None, save_path=None):
    """
    Comprehensive VAE analysis combining multiple interpretability methods.

    Args:
        vae_model: Trained VAE model
        X_data: Input data
        y_data: Optional labels for supervised analysis
        save_path: Path to save results
    """
    print("Starting comprehensive VAE interpretability analysis...")

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    results = {}

    # 1. Try SHAP analysis first
    print("\n=== SHAP Analysis ===")
    shap_results = explain_vae_with_shap_robust(vae_model, X_data, sample_size=50)

    if shap_results.get('success', False):
        print("SHAP analysis successful!")
        results['shap'] = shap_results

        # Plot SHAP results
        plot_shap_analysis_robust(shap_results, save_path=save_path)
    else:
        print("SHAP analysis failed, using alternative methods...")

    # 2. Gradient-based feature importance
    print("\n=== Gradient-based Feature Importance ===")
    importance_results = analyze_feature_importance_alternative(vae_model, X_data)

    if importance_results.get('success', False):
        print("Gradient analysis successful!")
        results['gradient_importance'] = importance_results
        plot_feature_importance(importance_results, save_path=save_path)

    # 3. Latent space analysis
    print("\n=== Latent Space Analysis ===")
    latent_results = analyze_latent_space_interpretability(vae_model, X_data, y_data, save_path)
    results['latent_analysis'] = latent_results

    # 4. Reconstruction analysis
    print("\n=== Reconstruction Quality Analysis ===")
    reconstruction_results = analyze_reconstruction_patterns(vae_model, X_data, save_path)
    results['reconstruction'] = reconstruction_results

    return results


def run_vae_interpretability_analysis(vae_model, X_train, y_train, snp_data_loc):
    """
    Main function to run comprehensive VAE interpretability analysis.

    Args:
        vae_model: Trained VAE model
        X_train: Training data
        y_train: Training labels
        snp_data_loc: Path to SNP data file

    Returns:
        dict: Analysis results
    """
    print("="*60)
    print("STARTING VAE INTERPRETABILITY ANALYSIS")
    print("="*60)

    # Create output directory
    analysis_output_dir = f"vae_analysis/{os.path.splitext(os.path.basename(snp_data_loc))[0]}"
    os.makedirs(analysis_output_dir, exist_ok=True)

    try:
        # Run comprehensive analysis
        results = comprehensive_vae_analysis(
            vae_model, X_train, y_train, save_path=analysis_output_dir
        )

        # Save results
        results_file = os.path.join(analysis_output_dir, "analysis_results.npz")

        # Prepare data for saving (only numpy arrays)
        save_dict = {}
        for key, value in results.items():
            if isinstance(value, dict) and value.get('success', False):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        save_dict[f"{key}_{subkey}"] = subvalue

        if save_dict:
            np.savez(results_file, **save_dict)

        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {analysis_output_dir}")

        return results

    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


# Example usage function that can be imported and used in main script
def analyze_vae_interpretability(vae_model, X_train, y_train, snp_data_loc):
    """
    Wrapper function for easy importing and usage.

    Args:
        vae_model: Trained VAE model
        X_train: Training data
        y_train: Training labels
        snp_data_loc: Path to SNP data file

    Returns:
        dict: Analysis results
    """
    return run_vae_interpretability_analysis(vae_model, X_train, y_train, snp_data_loc)


# For integration into main script, add this import and call:
#
# # Import at the top of rbam_main.py:
# from runner.rbam_shap_analysis import analyze_vae_interpretability
#
# # Add this line after VAE training and before classifier training:
# if 'best_vae_model' in locals() and best_vae_model is not None:
#     interpretability_results = analyze_vae_interpretability(
#         best_vae_model, X_train, y_train, snp_data_loc
#     )