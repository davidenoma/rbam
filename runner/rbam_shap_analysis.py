import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.model_selection import train_test_split


def analyze_latent_space_interpretability(vae_model, X_data, y_data=None, save_path=None):
    """
    Analyze latent space structure and interpretability.
    """
    print("Analyzing latent space structure...")

    try:
        # Handle DataFrame input
        if isinstance(X_data, pd.DataFrame):
            X_data_np = X_data.values
        else:
            X_data_np = np.array(X_data)

        # Sample data for analysis
        sample_size = min(1000, X_data_np.shape[0])
        sample_indices = np.random.choice(X_data_np.shape[0], sample_size, replace=False)
        sample_data = X_data_np[sample_indices]

        # Get latent representations
        encoded = vae_model.encoder.predict(sample_data, verbose=0)
        z_mean, z_log_var = tf.split(encoded, num_or_size_splits=2, axis=1)
        z_mean_np = z_mean.numpy()
        z_log_var_np = z_log_var.numpy()

        # Calculate latent space statistics
        latent_dim = z_mean_np.shape[1]

        # 1. Latent dimension variance analysis
        latent_variances = np.var(z_mean_np, axis=0)
        latent_means = np.mean(z_mean_np, axis=0)

        # 2. PCA on latent space
        if latent_dim > 2:
            pca = PCA(n_components=min(10, latent_dim))
            z_pca = pca.fit_transform(z_mean_np)
            explained_variance_ratio = pca.explained_variance_ratio_
        else:
            z_pca = z_mean_np
            explained_variance_ratio = np.array([1.0, 0.0])

        # 3. Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Latent dimension variances
        axes[0, 0].bar(range(latent_dim), latent_variances)
        axes[0, 0].set_xlabel('Latent Dimension')
        axes[0, 0].set_ylabel('Variance')
        axes[0, 0].set_title('Variance per Latent Dimension')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Latent dimension means
        axes[0, 1].bar(range(latent_dim), latent_means)
        axes[0, 1].set_xlabel('Latent Dimension')
        axes[0, 1].set_ylabel('Mean')
        axes[0, 1].set_title('Mean per Latent Dimension')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: PCA explained variance
        if len(explained_variance_ratio) > 1:
            axes[1, 0].bar(range(len(explained_variance_ratio)), explained_variance_ratio)
            axes[1, 0].set_xlabel('PCA Component')
            axes[1, 0].set_ylabel('Explained Variance Ratio')
            axes[1, 0].set_title('PCA Explained Variance in Latent Space')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: 2D latent space visualization (colored by phenotype if available)
        if y_data is not None:
            y_sample = y_data[sample_indices] if len(y_data) > sample_size else y_data
            scatter = axes[1, 1].scatter(z_pca[:, 0], z_pca[:, 1], c=y_sample,
                                         cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=axes[1, 1])
            axes[1, 1].set_title('Latent Space (colored by phenotype)')
        else:
            axes[1, 1].scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6)
            axes[1, 1].set_title('Latent Space Distribution')

        axes[1, 1].set_xlabel('First Component')
        axes[1, 1].set_ylabel('Second Component')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, "latent_space_analysis.png"),
                        dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Analyze latent space clustering if phenotype data available
        clustering_results = None
        if y_data is not None:
            from sklearn.metrics import silhouette_score
            try:
                y_sample = y_data[sample_indices] if len(y_data) > sample_size else y_data

                # Calculate silhouette score for phenotype separation
                if len(np.unique(y_sample)) > 1:
                    sil_score = silhouette_score(z_mean_np, y_sample)
                    clustering_results = {
                        'silhouette_score': sil_score,
                        'phenotype_separation': True
                    }
                    print(f"Latent space silhouette score for phenotype separation: {sil_score:.3f}")
            except Exception as e:
                print(f"Could not compute clustering metrics: {e}")

        return {
            'latent_dim': latent_dim,
            'latent_variances': latent_variances,
            'latent_means': latent_means,
            'pca_explained_variance': explained_variance_ratio,
            'total_variance_explained': np.sum(explained_variance_ratio[:2]),
            'clustering_results': clustering_results,
            'z_mean_sample': z_mean_np,
            'z_pca': z_pca,
            'success': True
        }

    except Exception as e:
        print(f"Latent space analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
def explain_vae_with_shap_robust(vae_model, X_data, snp_names=None, sample_size=100, max_evals=50):
    """
    Robust SHAP analysis for VAE with proper TF2 compatibility.
    """
    try:
        print("Initializing robust SHAP analysis for VAE...")

        # Ensure data is properly formatted
        if isinstance(X_data, pd.DataFrame):
            if snp_names is None:
                snp_names = X_data.columns.tolist()
            X_data = X_data.values
        X_data = np.array(X_data, dtype=np.float32)

        print(f"Original data shape: {X_data.shape}")

        # Sample individuals instead of reducing features
        np.random.seed(42)
        n_samples = X_data.shape[0]
        sample_indices = np.random.choice(n_samples, size=min(100, n_samples), replace=False)
        X_sampled = X_data[sample_indices].copy()

        print(f"Sampled data shape: {X_sampled.shape}")

        # Create smaller background and explain sets from sampled data
        background_size = min(30, len(X_sampled))
        explain_size = min(5, len(X_sampled))

        # Sample from the already sampled data
        background_indices = np.random.choice(len(X_sampled), size=background_size, replace=False)
        background_data = X_sampled[background_indices].copy()

        explain_indices = np.random.choice(len(X_sampled), size=explain_size, replace=False)
        explain_data = X_sampled[explain_indices].copy()

        print(f"Background data shape: {background_data.shape}")
        print(f"Explain data shape: {explain_data.shape}")

        # Initialize the encoder by calling it once with sample data
        try:
            print("Initializing encoder model...")
            # Call the encoder once to build it
            _ = vae_model.encoder(background_data[:1])

            # Now create the SHAP-compatible model
            encoder_output = vae_model.encoder.output
            z_mean, _ = tf.split(encoder_output, num_or_size_splits=2, axis=1)

            # Create a new model for SHAP using the same input as the encoder
            shap_model = tf.keras.Model(inputs=vae_model.encoder.input, outputs=z_mean)

            print("Testing SHAP-compatible model...")
            test_output = shap_model(background_data[:1])
            print(f"SHAP model test output shape: {test_output.shape}")

            # SHAP analysis with the Keras model
            print("Creating SHAP explainer with Keras model...")
            explainer = shap.DeepExplainer(shap_model, background_data)

            print("Computing SHAP values...")
            shap_values = explainer.shap_values(explain_data, check_additivity=False)

            if shap_values is None:
                return {'success': False, 'error': 'SHAP values computation returned None'}

            print(f"SHAP values computed successfully")

            return {
                'encoder_shap_values': shap_values,
                'explain_data': explain_data,
                'background_data': background_data,
                'explainer': explainer,
                'snp_names': snp_names,
                'sampled_indices': sample_indices,
                'success': True
            }

        except Exception as shap_error:
            print(f"SHAP with Keras model failed: {shap_error}")

            # Fallback: Use Permutation explainer with proper max_evals
            print("Trying Permutation explainer as fallback...")

            def model_predict(x):
                # Ensure the encoder is called properly
                x = tf.constant(x, dtype=tf.float32)
                encoded = vae_model.encoder(x, training=False)
                z_mean, _ = tf.split(encoded, num_or_size_splits=2, axis=1)
                return z_mean.numpy()

            # Test the model_predict function
            print("Testing model_predict function...")
            test_pred = model_predict(background_data[:1])
            print(f"Model predict test output shape: {test_pred.shape}")

            # For large feature sets, use a more efficient approach
            # Limit max_evals to a reasonable number
            max_features_for_permutation = 1000
            safe_max_evals = max_features_for_permutation * 2 + 1

            if X_sampled.shape[1] > max_features_for_permutation:
                print(f"Too many features ({X_sampled.shape[1]}) for Permutation explainer.")
                print("Using alternative gradient-based approach...")

                # Use gradient-based method instead
                return analyze_feature_importance_alternative(vae_model, X_sampled, snp_names)

            print(f"Using max_evals={safe_max_evals} for {X_sampled.shape[1]} features")

            # Use permutation explainer with proper settings
            explainer = shap.explainers.Permutation(
                model_predict,
                background_data,
                max_evals=safe_max_evals
            )

            print("Computing SHAP values with Permutation explainer...")
            shap_values = explainer(explain_data)

            return {
                'encoder_shap_values': shap_values.values,
                'explain_data': explain_data,
                'background_data': background_data,
                'explainer': explainer,
                'snp_names': snp_names,
                'sampled_indices': sample_indices,
                'success': True
            }

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def analyze_feature_importance_alternative(vae_model, X_data, snp_names=None, num_features=20):
    """
    Alternative feature importance analysis using gradient-based methods.
    """
    print("Performing gradient-based feature importance analysis...")

    try:
        # Handle DataFrame input
        if isinstance(X_data, pd.DataFrame):
            if snp_names is None:
                snp_names = X_data.columns.tolist()
            X_data = X_data.values

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
            top_k = min(num_features, len(feature_importance))
            top_indices = tf.nn.top_k(feature_importance, k=top_k).indices
            top_importance = tf.gather(feature_importance, top_indices)

            # Convert to numpy for easier handling
            feature_importance_np = feature_importance.numpy()
            top_indices_np = top_indices.numpy()
            top_importance_np = top_importance.numpy()

            return {
                'feature_importance': feature_importance_np,
                'top_indices': top_indices_np,
                'top_importance': top_importance_np,
                'snp_names': snp_names,
                'success': True
            }
        else:
            print("Could not compute gradients")
            return {'success': False}

    except Exception as e:
        print(f"Gradient analysis failed: {e}")
        return {'success': False, 'error': str(e)}


def plot_feature_importance(importance_results, save_path=None):
    """
    Plot feature importance results using SNP names.
    """
    if not importance_results.get('success', False):
        print("No successful importance results to plot")
        return

    print("Creating feature importance plots...")

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Get data
    top_indices = importance_results['top_indices']
    top_importance = importance_results['top_importance']
    snp_names = importance_results.get('snp_names', None)

    # Create feature names
    if snp_names is not None and len(snp_names) > max(top_indices):
        top_features = [snp_names[i] for i in top_indices]
    else:
        top_features = [f"SNP_{i}" for i in top_indices]

    # Plot 1: Top feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_importance)), top_importance)
    plt.yticks(range(len(top_importance)), top_features)
    plt.xlabel('Feature Importance (Mean Absolute Gradient)')
    plt.title(f'Top {len(top_importance)} SNPs by Importance for Latent Space Construction')
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "snp_importance_gradient.png"),
                    dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Feature importance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(importance_results['feature_importance'], bins=50, alpha=0.7)
    plt.xlabel('SNP Importance')
    plt.ylabel('Frequency')
    plt.title('Distribution of SNP Importance Scores')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(os.path.join(save_path, "snp_importance_distribution.png"),
                    dpi=300, bbox_inches='tight')
    plt.show()


def analyze_reconstruction_patterns(vae_model, X_data, snp_names=None, save_path=None):
    """
    Analyze reconstruction patterns using proper data handling.
    """
    print("Analyzing reconstruction patterns...")

    try:
        # Handle DataFrame input
        if isinstance(X_data, pd.DataFrame):
            if snp_names is None:
                snp_names = X_data.columns.tolist()
            X_data_np = X_data.values
        else:
            X_data_np = np.array(X_data)

        # Get reconstructions
        sample_size = min(1000, X_data_np.shape[0])
        sample_indices = np.random.choice(X_data_np.shape[0], sample_size, replace=False)
        sample_data = X_data_np[sample_indices]

        reconstructions = vae_model.predict(sample_data, verbose=0)

        # Calculate reconstruction errors per feature
        reconstruction_errors = np.mean((sample_data - reconstructions) ** 2, axis=0)

        # Get top error indices
        top_error_indices = np.argsort(reconstruction_errors)[-20:]

        # Create SNP names for top errors
        if snp_names is not None and len(snp_names) >= len(reconstruction_errors):
            top_error_snps = [snp_names[i] for i in top_error_indices]
        else:
            top_error_snps = [f"SNP_{i}" for i in top_error_indices]

        # Plot reconstruction error distribution
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(reconstruction_errors, bins=50, alpha=0.7)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of SNP Reconstruction Errors')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.barh(range(20), reconstruction_errors[top_error_indices])
        plt.yticks(range(20), top_error_snps, fontsize=8)
        plt.xlabel('Reconstruction Error')
        plt.title('Top 20 SNPs with Highest Reconstruction Error')

        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "snp_reconstruction_analysis.png"),
                        dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'reconstruction_errors': reconstruction_errors,
            'high_error_indices': top_error_indices,
            'high_error_snps': top_error_snps,
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'snp_names': snp_names,
            'success': True
        }

    except Exception as e:
        print(f"Reconstruction analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def plot_shap_analysis_robust(shap_results, save_path=None):
    """
    Robust SHAP plotting with SNP names.
    """
    try:
        if not shap_results.get('success', False):
            print("No successful SHAP results to plot")
            return

        print("Creating SHAP visualization plots...")

        encoder_shap_values = shap_results['encoder_shap_values']
        explain_data = shap_results['explain_data']
        snp_names = shap_results.get('snp_names', None)

        # Handle different SHAP value formats
        if isinstance(encoder_shap_values, list):
            shap_values_to_plot = encoder_shap_values[0] if len(encoder_shap_values) > 0 else encoder_shap_values
        else:
            shap_values_to_plot = encoder_shap_values

        # Plot feature importance bar plot with SNP names
        if len(shap_values_to_plot.shape) >= 2:
            mean_importance = np.mean(np.abs(shap_values_to_plot), axis=0)
            top_indices = np.argsort(mean_importance)[-20:]

            # Create SNP names for top features
            if snp_names is not None and len(snp_names) >= len(mean_importance):
                top_snps = [snp_names[i] for i in top_indices]
            else:
                top_snps = [f"SNP_{i}" for i in top_indices]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_indices)), mean_importance[top_indices])
            plt.yticks(range(len(top_indices)), top_snps, fontsize=8)
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Top 20 SNPs by SHAP Importance')
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, "shap_snp_importance.png"),
                            dpi=300, bbox_inches='tight')
            plt.show()

    except Exception as e:
        print(f"Error in SHAP plotting: {e}")


def comprehensive_vae_analysis(vae_model, X_data, y_data=None, snp_names=None, save_path=None):
    """
    Comprehensive VAE analysis with SNP name support.
    """
    print("Starting comprehensive VAE interpretability analysis...")

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    results = {}

    # Extract SNP names if X_data is DataFrame
    if isinstance(X_data, pd.DataFrame) and snp_names is None:
        snp_names = X_data.columns.tolist()

    # 1. Try SHAP analysis first
    print("\n=== SHAP Analysis ===")
    shap_results = explain_vae_with_shap_robust(vae_model, X_data, snp_names=snp_names, sample_size=50)

    if shap_results.get('success', False):
        print("SHAP analysis successful!")
        results['shap'] = shap_results
        plot_shap_analysis_robust(shap_results, save_path=save_path)
    else:
        print("SHAP analysis failed, using alternative methods...")

    # 2. Gradient-based feature importance
    print("\n=== Gradient-based Feature Importance ===")
    importance_results = analyze_feature_importance_alternative(vae_model, X_data, snp_names=snp_names)

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
    reconstruction_results = analyze_reconstruction_patterns(vae_model, X_data, snp_names=snp_names,
                                                             save_path=save_path)
    results['reconstruction'] = reconstruction_results

    return results


# Keep all other functions the same, but update the main analysis function:
def run_vae_interpretability_analysis(vae_model, X_train, y_train, snp_data_loc):
    """
    Main function to run comprehensive VAE interpretability analysis with SNP names.
    """
    print("=" * 60)
    print("STARTING VAE INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    # Create output directory
    analysis_output_dir = f"vae_analysis/{os.path.splitext(os.path.basename(snp_data_loc))[0]}"
    os.makedirs(analysis_output_dir, exist_ok=True)

    try:
        # Extract SNP names if X_train is DataFrame
        snp_names = None
        if isinstance(X_train, pd.DataFrame):
            snp_names = X_train.columns.tolist()
            print(f"Found {len(snp_names)} SNP names")

        # Run comprehensive analysis
        results = comprehensive_vae_analysis(
            vae_model, X_train, y_train, snp_names=snp_names, save_path=analysis_output_dir
        )

        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {analysis_output_dir}")

        return results

    except Exception as e:
        print(f"Analysis failed: {e}")
        return None