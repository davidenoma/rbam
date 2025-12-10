import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.model_selection import train_test_split


def calculate_shap_scores_per_snp(shap_values, snp_names):
    """
    Calculate aggregated SHAP scores per SNP.
    """
    try:
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values[0]) if len(shap_values) > 0 else np.array(shap_values)
        else:
            shap_array = np.array(shap_values)

        print(f"Processing SHAP values with shape: {shap_array.shape}")

        # Calculate mean absolute SHAP values across samples
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)

        # Calculate mean SHAP values (with sign)
        mean_shap = np.mean(shap_array, axis=0)

        # Calculate standard deviation of SHAP values
        std_shap = np.std(shap_array, axis=0)

        # Calculate max absolute SHAP values
        max_abs_shap = np.max(np.abs(shap_array), axis=0)

        # Create SNP names if not provided
        if snp_names is None:
            snp_names = [f"SNP_{i}" for i in range(len(mean_abs_shap))]

        # Ensure we have the right number of SNP names
        n_snps = len(mean_abs_shap)
        if len(snp_names) < n_snps:
            snp_names.extend([f"SNP_{i}" for i in range(len(snp_names), n_snps)])

        # Create DataFrame with SHAP scores
        shap_scores_df = pd.DataFrame({
            'SNP_Name': snp_names[:n_snps],
            'SNP_Index': range(n_snps),
            'Mean_Absolute_SHAP': mean_abs_shap,
            'Mean_SHAP': mean_shap,
            'Std_SHAP': std_shap,
            'Max_Absolute_SHAP': max_abs_shap,
            'Importance_Rank': range(1, n_snps + 1)
        })

        # Sort by absolute importance and update ranks
        shap_scores_df = shap_scores_df.sort_values('Mean_Absolute_SHAP', ascending=False)
        shap_scores_df['Importance_Rank'] = range(1, len(shap_scores_df) + 1)

        print(f"Calculated SHAP scores for {len(shap_scores_df)} SNPs")
        return shap_scores_df

    except Exception as e:
        print(f"Error calculating SHAP scores per SNP: {e}")
        import traceback
        traceback.print_exc()
        return None


def explain_vae_with_shap_robust(vae_model, X_data, snp_names=None, sample_size=100, max_evals=50):
    """
    Robust SHAP analysis for VAE with proper TF2 compatibility and score calculation.
    """
    try:
        print("Initializing robust SHAP analysis for VAE...")

        # Handle DataFrame input and extract SNP names
        if isinstance(X_data, pd.DataFrame):
            if snp_names is None:
                snp_names = X_data.columns.tolist()
            X_data_np = X_data.values
        else:
            X_data_np = np.array(X_data)

        X_data_np = X_data_np.astype(np.float32)
        print(f"Data shape: {X_data_np.shape}")

        # Create very small sample sizes for high-dimensional data
        np.random.seed(42)
        n_samples = X_data_np.shape[0]
        n_features = X_data_np.shape[1]

        # For high-dimensional data, use very small samples
        background_size = min(20, n_samples)
        explain_size = min(10, n_samples)

        # Sample data
        background_indices = np.random.choice(n_samples, size=background_size, replace=False)
        background_data = X_data_np[background_indices].copy()

        explain_indices = np.random.choice(n_samples, size=explain_size, replace=False)
        explain_data = X_data_np[explain_indices].copy()

        print(f"Background data shape: {background_data.shape}")
        print(f"Explain data shape: {explain_data.shape}")

        # First, call the encoder to initialize it
        print("Initializing encoder...")
        _ = vae_model.encoder(background_data[:1])

        try:
            # Try DeepExplainer with a proper Keras model
            print("Trying DeepExplainer approach...")

            # Create a simplified encoder model
            encoder_input = tf.keras.layers.Input(shape=(X_data_np.shape[1],))
            encoder_output = vae_model.encoder(encoder_input)
            z_mean, _ = tf.split(encoder_output, num_or_size_splits=2, axis=1)

            # Create a new model for SHAP
            shap_model = tf.keras.Model(inputs=encoder_input, outputs=z_mean)

            print("Testing SHAP-compatible model...")
            test_output = shap_model(background_data[:1])
            print(f"SHAP model test output shape: {test_output.shape}")

            # Use smaller background for DeepExplainer
            small_background = background_data[:5]
            explainer = shap.DeepExplainer(shap_model, small_background)

            print("Computing SHAP values with DeepExplainer...")
            shap_values = explainer.shap_values(explain_data, check_additivity=False)

            if shap_values is not None:
                print("DeepExplainer SHAP analysis successful!")

                # Calculate aggregated SHAP scores per SNP
                shap_scores_per_snp = calculate_shap_scores_per_snp(shap_values, snp_names)

                return {
                    'encoder_shap_values': shap_values,
                    'explain_data': explain_data,
                    'background_data': background_data,
                    'explainer': explainer,
                    'snp_names': snp_names,
                    'shap_scores_per_snp': shap_scores_per_snp,
                    'method': 'DeepExplainer',
                    'success': True
                }

        except Exception as deep_error:
            print(f"DeepExplainer failed: {deep_error}")

        # Fallback: Use gradient-based feature importance
        print("Using gradient-based feature importance as SHAP alternative...")

        sample_data = tf.constant(explain_data, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(sample_data)
            encoded = vae_model.encoder(sample_data, training=False)
            z_mean, _ = tf.split(encoded, num_or_size_splits=2, axis=1)
            # Use mean of latent dimensions as target
            target = tf.reduce_mean(z_mean, axis=1)

        gradients = tape.gradient(target, sample_data)

        if gradients is not None:
            # Convert gradients to SHAP-like values
            pseudo_shap_values = gradients.numpy()

            # Calculate aggregated scores per SNP
            shap_scores_per_snp = calculate_shap_scores_per_snp(pseudo_shap_values, snp_names)

            print("Gradient-based pseudo-SHAP analysis successful!")
            return {
                'encoder_shap_values': pseudo_shap_values,
                'explain_data': explain_data,
                'background_data': background_data,
                'snp_names': snp_names,
                'shap_scores_per_snp': shap_scores_per_snp,
                'method': 'GradientBased',
                'success': True
            }

        return {'success': False, 'error': 'All SHAP methods failed'}

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def save_shap_scores(shap_results, save_path):
    """
    Save SHAP scores to CSV files with comprehensive information.
    """
    try:
        if not shap_results.get('success', False):
            print("No successful SHAP results to save")
            return None

        os.makedirs(save_path, exist_ok=True)

        shap_scores_df = shap_results.get('shap_scores_per_snp', None)

        if shap_scores_df is not None and len(shap_scores_df) > 0:
            print(f"Saving SHAP scores for {len(shap_scores_df)} SNPs...")

            # Save complete SHAP scores
            complete_scores_file = os.path.join(save_path, "shap_scores_all_snps.csv")
            shap_scores_df.to_csv(complete_scores_file, index=False)
            print(f"✓ Complete SHAP scores saved to: {complete_scores_file}")

            files_saved = [complete_scores_file]

            # Save top 100 SNPs
            if len(shap_scores_df) >= 100:
                top100_snps_file = os.path.join(save_path, "shap_scores_top100_snps.csv")
                shap_scores_df.head(100).to_csv(top100_snps_file, index=False)
                print(f"✓ Top 100 SNP SHAP scores saved to: {top100_snps_file}")
                files_saved.append(top100_snps_file)

            # Save top 50 SNPs
            if len(shap_scores_df) >= 50:
                top50_snps_file = os.path.join(save_path, "shap_scores_top50_snps.csv")
                shap_scores_df.head(50).to_csv(top50_snps_file, index=False)
                print(f"✓ Top 50 SNP SHAP scores saved to: {top50_snps_file}")
                files_saved.append(top50_snps_file)

            # Save top 20 SNPs
            top20_snps_file = os.path.join(save_path, "shap_scores_top20_snps.csv")
            shap_scores_df.head(20).to_csv(top20_snps_file, index=False)
            print(f"✓ Top 20 SNP SHAP scores saved to: {top20_snps_file}")
            files_saved.append(top20_snps_file)

            # Save raw SHAP values as well
            if 'encoder_shap_values' in shap_results:
                raw_shap_file = os.path.join(save_path, "raw_shap_values.npz")
                np.savez(
                    raw_shap_file,
                    encoder_shap_values=shap_results['encoder_shap_values'],
                    explain_data=shap_results['explain_data'],
                    snp_names=np.array(shap_results.get('snp_names', [])) if shap_results.get('snp_names') else None
                )
                print(f"✓ Raw SHAP values saved to: {raw_shap_file}")
                files_saved.append(raw_shap_file)

            # Create detailed summary statistics
            summary_file = os.path.join(save_path, "shap_analysis_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("SHAP Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Method used: {shap_results.get('method', 'Unknown')}\n")
                f.write(f"Total SNPs analyzed: {len(shap_scores_df)}\n")
                f.write(f"Mean absolute SHAP score: {shap_scores_df['Mean_Absolute_SHAP'].mean():.8f}\n")
                f.write(f"Median absolute SHAP score: {shap_scores_df['Mean_Absolute_SHAP'].median():.8f}\n")
                f.write(f"Max absolute SHAP score: {shap_scores_df['Mean_Absolute_SHAP'].max():.8f}\n")
                f.write(f"95th percentile score: {shap_scores_df['Mean_Absolute_SHAP'].quantile(0.95):.8f}\n")
                f.write(f"90th percentile score: {shap_scores_df['Mean_Absolute_SHAP'].quantile(0.90):.8f}\n\n")

                f.write("Top 10 SNPs:\n")
                for i, (_, row) in enumerate(shap_scores_df.head(10).iterrows(), 1):
                    f.write(f"{i:2d}. {row['SNP_Name']}: {row['Mean_Absolute_SHAP']:.8f}\n")

            print(f"✓ SHAP analysis summary saved to: {summary_file}")
            files_saved.append(summary_file)

            # Save significance thresholds
            threshold_file = os.path.join(save_path, "shap_significance_thresholds.csv")
            percentiles = [99, 95, 90, 75, 50]
            threshold_data = []

            for p in percentiles:
                threshold = shap_scores_df['Mean_Absolute_SHAP'].quantile(p / 100)
                n_snps_above = len(shap_scores_df[shap_scores_df['Mean_Absolute_SHAP'] >= threshold])
                threshold_data.append({
                    'Percentile': p,
                    'Threshold': threshold,
                    'SNPs_Above_Threshold': n_snps_above
                })

            threshold_df = pd.DataFrame(threshold_data)
            threshold_df.to_csv(threshold_file, index=False)
            print(f"✓ SHAP significance thresholds saved to: {threshold_file}")
            files_saved.append(threshold_file)

            return {
                'n_snps_saved': len(shap_scores_df),
                'files_saved': files_saved,
                'shap_scores_df': shap_scores_df
            }

        else:
            print("No SHAP scores DataFrame found to save")
            return None

    except Exception as e:
        print(f"Error saving SHAP scores: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_shap_analysis_robust(shap_results, save_path=None):
    """
    Enhanced SHAP plotting with SNP names and comprehensive visualizations.
    """
    try:
        if not shap_results.get('success', False):
            print("No successful SHAP results to plot")
            return

        print(f"Creating SHAP visualization plots using {shap_results.get('method', 'unknown')} method...")

        shap_scores_df = shap_results.get('shap_scores_per_snp', None)

        if shap_scores_df is not None and len(shap_scores_df) > 0:
            # Plot 1: Top 20 SNPs bar plot
            plt.figure(figsize=(12, 8))
            top_20 = shap_scores_df.head(20)
            colors = ['red' if x < 0 else 'green' for x in top_20['Mean_SHAP']]

            plt.barh(range(len(top_20)), top_20['Mean_Absolute_SHAP'], color=colors, alpha=0.7)
            plt.yticks(range(len(top_20)), top_20['SNP_Name'], fontsize=8)
            plt.xlabel(f'Mean |SHAP Value| ({shap_results.get("method", "Unknown")})')
            plt.title('Top 20 SNPs by SHAP Importance\n(Green: Positive Effect, Red: Negative Effect)')
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, "shap_top20_snps.png"),
                            dpi=300, bbox_inches='tight')
            plt.show()

            # Plot 2: SHAP score distribution
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(shap_scores_df['Mean_Absolute_SHAP'], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Mean Absolute SHAP Value')
            plt.ylabel('Number of SNPs')
            plt.title('Distribution of SNP SHAP Importance Scores')
            plt.axvline(shap_scores_df['Mean_Absolute_SHAP'].mean(), color='red',
                        linestyle='--', label=f'Mean: {shap_scores_df["Mean_Absolute_SHAP"].mean():.6f}')
            plt.axvline(np.percentile(shap_scores_df['Mean_Absolute_SHAP'], 95), color='orange',
                        linestyle='--',
                        label=f'95th percentile: {np.percentile(shap_scores_df["Mean_Absolute_SHAP"], 95):.6f}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Log scale distribution
            plt.subplot(1, 2, 2)
            plt.hist(shap_scores_df['Mean_Absolute_SHAP'], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Mean Absolute SHAP Value')
            plt.ylabel('Number of SNPs')
            plt.title('Distribution (Log Scale)')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(os.path.join(save_path, "shap_distribution.png"),
                            dpi=300, bbox_inches='tight')
            plt.show()

            print(f"SHAP visualizations created for {len(shap_scores_df)} SNPs")

        else:
            print("No SHAP scores DataFrame found for plotting")

    except Exception as e:
        print(f"Error in SHAP plotting: {e}")
        import traceback
        traceback.print_exc()


# Keep all your other existing functions unchanged
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


# Keep your existing analyze_latent_space_interpretability function unchanged
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

        # Plot 4: 2D latent space visualization
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

        return {
            'latent_dim': latent_dim,
            'latent_variances': latent_variances,
            'latent_means': latent_means,
            'pca_explained_variance': explained_variance_ratio,
            'total_variance_explained': np.sum(explained_variance_ratio[:2]),
            'z_mean_sample': z_mean_np,
            'z_pca': z_pca,
            'success': True
        }

    except Exception as e:
        print(f"Latent space analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}