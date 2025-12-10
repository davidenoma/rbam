import os
import subprocess
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, hp, tpe, space_eval
from matplotlib import pyplot as plt
import rbam_shap_analysis

import utils
from utils import save_summary, load_real_genotype_data_case_control, load_real_genotype_data


# Set up environment
# Find CUDA installation path using `whereis cuda` command
try:
    cuda_path = subprocess.run(["whereis", "cuda"], capture_output=True, text=True).stdout.split()[1]
    if cuda_path:
        os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_path}'
        print("CUDA path set for XLA:", cuda_path)
    else:
        print("CUDA not found; running without XLA GPU configuration.")
except IndexError:
    print("CUDA not found; running without XLA GPU configuration.")


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

@tf.keras.utils.register_keras_serializable(package="Custom", name="VAE")
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def get_config(self) -> dict:
        config = super(VAE, self).get_config()
        config.update({
            'encoder': tf.keras.utils.serialize_keras_object(self.encoder),
            'decoder': tf.keras.utils.serialize_keras_object(self.decoder),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'VAE':
        encoder = tf.keras.utils.deserialize_keras_object(config.pop('encoder'))
        decoder = tf.keras.utils.deserialize_keras_object(config.pop('decoder'))
        return cls(encoder=encoder, decoder=decoder)

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed

@tf.keras.utils.register_keras_serializable(package="Custom", name="vae_loss")
def vae_loss(encoder):
    def loss(x, x_reconstructed):
        z_mean, z_log_var = tf.split(encoder(x), num_or_size_splits=2, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = tf.maximum(reconstruction_loss + kl_loss, 0)  # Prevent negative loss
        return total_loss
    return loss

def create_vae_model(input_dim, num_hidden_layers_encoder, num_hidden_layers_decoder, encoding_dimensions,
                     decoding_dimensions, activation, batch_size, epochs, learning_rate, latent_dim):
    encoder_layers = [input_dim] + [encoding_dimensions] * num_hidden_layers_encoder + [2 * latent_dim]
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation) for layer in encoder_layers[1:]]
    ])

    decoder_layers = [latent_dim] + [decoding_dimensions] * num_hidden_layers_decoder + [input_dim]
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation) for layer in decoder_layers[1:]]
    ])

    vae = VAE(encoder=encoder, decoder=decoder)
    loss_function = vae_loss(vae.encoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_function)

    return vae

def save_model(model, snp_data_loc, directory, override=False):
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    print("Saving model to filepath", filepath)
    if os.path.exists(filepath) and not override:
        raise FileExistsError(f"The file {filename} already exists. Set override=True to overwrite.")
    model.save(filepath)

def load_model(snp_data_loc, directory_loc):
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    filepath = os.path.join(directory_loc, filename)
    if os.path.exists(filepath):
        print('Loading saved model: ', filepath)
        return tf.keras.models.load_model(filepath, custom_objects={"VAE": VAE, "vae_loss": vae_loss})
    else:
        print('Model not found: ', filepath)
        print('Now training...')
        return None

# Load data
directory = ""
snp_data_loc = sys.argv[1]

if sys.argv[3] == 'quantitative' or sys.argv[3] == 'cc_com':
    X_train, X_test, snp_data, phenotype, y_train, y_test = load_real_genotype_data(snp_data_loc)

    hopt = "hopt_cc_com_or_quant"
    directory = f"{os.getcwd()}/model_cc_com_qt"
else:
    X_train, X_test, snp_data = load_real_genotype_data_case_control(snp_data_loc)

    hopt = "hopt"
    directory = f"{os.getcwd()}/model"

def objective(params):
    model = create_vae_model(input_dim=X_train.shape[1], **params)
    history = model.fit(X_train, X_train,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        validation_split=0.25,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                        verbose=1)
    return {'loss': history.history['val_loss'][-1], 'status': 'ok'}


# Load or create VAE model
best_model = load_model(snp_data_loc, directory)

if not best_model:
    space = {
        'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', range(1, 17)),
        'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', range(1, 17)),
        'encoding_dimensions': hp.choice('encoding_dimensions', [128, 256, 512]),
        'decoding_dimensions': hp.choice('decoding_dimensions', [128, 256, 512]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'learning_rate': hp.choice('learning_rate', [0.000001, 0.00001, 0.0001, 0.001]),
        'epochs': hp.choice('epochs', [ 50, 100,150]),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'latent_dim': hp.choice('latent_dim', [4, 8, 16, 32, 64, 128, 512, 1024, int(X_train.shape[1] * 0.01),
                                               int(X_train.shape[1] * 0.05), int(X_train.shape[1] * 0.1),
                                               int(X_train.shape[1] * 0.5)])
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)
    best_hyperparameters = space_eval(space, best)
    print("Best hyperparameters for VAE:", best_hyperparameters)
    best_model = create_vae_model(input_dim=X_train.shape[1], **best_hyperparameters)
    best_history = best_model.fit(X_train, X_train, epochs=best_hyperparameters['epochs'],
                                  batch_size=best_hyperparameters['batch_size'], validation_split=0.25, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                             verbose=1)
    plt.figure(figsize=(10, 6))
    plt.plot(best_history.history['loss'], label='Training Loss')
    plt.plot(best_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    output_folder = "model_outputs"
    if hopt:
        output_folder = output_folder + "/" + hopt
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_loss_curve.png"))
    plt.show()
    save_model(best_model, snp_data_loc, directory)

# Reconstruct input data using the trained VAE
reconstructed_data_train = best_model.predict(X_train)
reconstructed_data_test = best_model.predict(X_test)
reconstructed_full_data = best_model.predict(snp_data)

mse_train = utils.compute_rmse(X_train, reconstructed_data_train)**2
mse_test = utils.compute_rmse(X_test, reconstructed_data_test)**2
mse_whole = utils.compute_rmse(snp_data, reconstructed_full_data)**2

# Calculate R²
r2_train = np.mean(utils.evaluate_r2(X_train, reconstructed_data_train))
r2_test = np.mean(utils.evaluate_r2(X_test, reconstructed_data_test))
r2_whole = np.mean(utils.evaluate_r2(snp_data, reconstructed_full_data))
utils.save_r2_scores(snp_data_loc, r2_train, r2_test, r2_whole, hopt=hopt)

# Calculate Adjusted R²
n_train, p_train = X_train.shape
n_test, p_test = X_test.shape
n_whole, p_whole = snp_data.shape

# Print results
print("MSE (Train):", mse_train)
print("MSE (Test):", mse_test)
print("MSE (Whole):", mse_whole)


print("R² (Train):", r2_train)
print("R² (Test):", r2_test)
print("R² (Whole):", r2_whole)


print("Cross Validation")

# Perform cross-validation
(
    avg_mse_train, avg_mse_test,
    avg_r2_train, avg_r2_test,
    avg_pearson_corr_train, avg_pearson_corr_test
) = utils.cross_validate_vae(snp_data, best_model)

# Save cross-validation metrics
utils.save_mse_values_cv(snp_data_loc, avg_mse_train, avg_mse_test, hopt=hopt)
utils.save_r2_scores_cv(snp_data_loc, avg_r2_train, avg_r2_test, hopt=hopt)

# Print cross-validation results
print("Average Training MSE:", avg_mse_train)
print("Average Testing MSE:", avg_mse_test)
print("Average Training R^2:", avg_r2_train)
print("Average Testing R^2:", avg_r2_test)

print("Average Training Pearson Correlation:", avg_pearson_corr_train)
print("Average Testing Pearson Correlation:", avg_pearson_corr_test)

# Replace the existing SHAP analysis section with this corrected version:
print("Starting SHAP analysis for VAE interpretability...")

# Create output directory for SHAP results
shap_output_dir = f"shap_analysis/{os.path.splitext(os.path.basename(snp_data_loc))[0]}"
os.makedirs(shap_output_dir, exist_ok=True)

try:
    # Extract SNP names from data if available
    snp_names = None
    if isinstance(X_train, pd.DataFrame):
        snp_names = X_train.columns.tolist()
    else:
        # Try to load SNP names from BIM file if available
        try:
            if len(sys.argv) > 2:
                bim_file = sys.argv[2]
                if os.path.exists(bim_file):
                    bim_df = pd.read_csv(bim_file, sep='\t', header=None)
                    snp_names = bim_df.iloc[:, 1].tolist()  # SNP names are in column 2
                    print(f"Loaded {len(snp_names)} SNP names from BIM file")
        except Exception as bim_error:
            print(f"Could not load SNP names from BIM file: {bim_error}")

    if snp_names is None:
        snp_names = [f"SNP_{i}" for i in range(X_train.shape[1])]
        print(f"Using generic SNP names for {len(snp_names)} SNPs")

    # Check if we have the required variables
    vae_model = best_model
    has_labels = 'y_train' in locals() and y_train is not None

    # Run individual analyses with proper error handling
    results = {}

    # 1. Try SHAP analysis with robust error handling
    print("\n=== Attempting SHAP Analysis ===")
    try:
        shap_results = rbam_shap_analysis.explain_vae_with_shap_robust(
            vae_model, X_train, snp_names=snp_names, sample_size=50
        )

        if shap_results and shap_results.get('success', False):
            print(f"SHAP analysis successful using {shap_results.get('method', 'unknown')} method!")
            results['shap'] = shap_results

            # Save SHAP scores to files
            print("Saving SHAP scores...")
            save_results = rbam_shap_analysis.save_shap_scores(shap_results, save_path=shap_output_dir)

            if save_results:
                print(f"✓ SHAP scores saved for {save_results['n_snps_saved']} SNPs")
                print(f"✓ Files saved: {len(save_results['files_saved'])}")
                for file_path in save_results['files_saved']:
                    print(f"  - {os.path.basename(file_path)}")

            # Plot SHAP results
            rbam_shap_analysis.plot_shap_analysis_robust(shap_results, save_path=shap_output_dir)

            # Print top 10 SNPs to console
            if 'shap_scores_per_snp' in shap_results and shap_results['shap_scores_per_snp'] is not None:
                print("\nTop 10 most important SNPs by SHAP analysis:")
                print("=" * 80)
                print(f"{'Rank':<6}{'SNP Name':<25}{'SHAP Score':<15}{'Effect':<12}{'Method':<15}")
                print("-" * 80)

                top_10 = shap_results['shap_scores_per_snp'].head(10)
                for i, (_, row) in enumerate(top_10.iterrows(), 1):
                    effect = "Positive" if row['Mean_SHAP'] > 0 else "Negative"
                    method = shap_results.get('method', 'Unknown')
                    print(f"{i:<6}{row['SNP_Name']:<25}{row['Mean_Absolute_SHAP']:<15.8f}{effect:<12}{method:<15}")

            # Save raw SHAP values
            if 'encoder_shap_values' in shap_results:
                raw_shap_file = os.path.join(shap_output_dir, "raw_shap_values.npz")
                np.savez(
                    raw_shap_file,
                    encoder_shap_values=shap_results['encoder_shap_values'],
                    explain_data=shap_results['explain_data'],
                    snp_names=np.array(snp_names) if snp_names else None
                )
                print(f"✓ Raw SHAP values saved to: {os.path.basename(raw_shap_file)}")

        else:
            print("SHAP analysis failed or returned invalid results")
            print(f"SHAP results: {shap_results}")

    except Exception as shap_error:
        print(f"SHAP analysis failed with error: {shap_error}")
        import traceback
        traceback.print_exc()

    # 2. Gradient-based feature importance (more reliable fallback)
    print("\n=== Gradient-based Feature Importance ===")
    try:
        importance_results = rbam_shap_analysis.analyze_feature_importance_alternative(
            vae_model, X_train, snp_names=snp_names, num_features=20
        )

        if importance_results and importance_results.get('success', False):
            print("Gradient analysis successful!")
            results['gradient_importance'] = importance_results
            rbam_shap_analysis.plot_feature_importance(importance_results, save_path=shap_output_dir)
        else:
            print("Gradient analysis failed")

    except Exception as grad_error:
        print(f"Gradient analysis failed with error: {grad_error}")

    # 3. Latent space analysis
    print("\n=== Latent Space Analysis ===")
    try:
        latent_results = rbam_shap_analysis.analyze_latent_space_interpretability(
            vae_model, X_train, y_data=y_train if has_labels else None, save_path=shap_output_dir
        )

        if latent_results and latent_results.get('success', False):
            print("Latent space analysis successful!")
            results['latent_analysis'] = latent_results
        else:
            print("Latent space analysis failed")

    except Exception as latent_error:
        print(f"Latent space analysis failed with error: {latent_error}")

    # 4. Reconstruction analysis
    print("\n=== Reconstruction Pattern Analysis ===")
    try:
        reconstruction_results = rbam_shap_analysis.analyze_reconstruction_patterns(
            vae_model, X_train, snp_names=snp_names, save_path=shap_output_dir
        )

        if reconstruction_results and reconstruction_results.get('success', False):
            print("Reconstruction analysis successful!")
            results['reconstruction'] = reconstruction_results
        else:
            print("Reconstruction analysis failed")

    except Exception as recon_error:
        print(f"Reconstruction analysis failed with error: {recon_error}")

    # Summary of completed analyses
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    successful_analyses = []
    failed_analyses = []

    analysis_types = ['shap', 'gradient_importance', 'latent_analysis', 'reconstruction']

    for analysis_name in analysis_types:
        if analysis_name in results and results[analysis_name].get('success', False):
            successful_analyses.append(analysis_name)
            print(f"✓ {analysis_name.replace('_', ' ').title()}: COMPLETED")
        else:
            failed_analyses.append(analysis_name)
            print(f"✗ {analysis_name.replace('_', ' ').title()}: FAILED")

    # Enhanced summary with SHAP score information
    summary_file = os.path.join(shap_output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("VAE Interpretability Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model file: {os.path.basename(snp_data_loc)}\n")
        f.write(f"Input dimensions: {X_train.shape[1]}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Has phenotype labels: {has_labels}\n")
        f.write(f"SNP names source: {'BIM file' if 'bim_file' in locals() else 'DataFrame columns' if isinstance(X_train, pd.DataFrame) else 'Generated'}\n\n")

        f.write("Completed Analyses:\n")
        for analysis in successful_analyses:
            f.write(f"  ✓ {analysis.replace('_', ' ').title()}\n")

        f.write("\nFailed Analyses:\n")
        for analysis in failed_analyses:
            f.write(f"  ✗ {analysis.replace('_', ' ').title()}\n")

        # Add SHAP-specific information
        if 'shap' in results and results['shap'].get('success', False):
            f.write(f"\nSHAP Analysis Details:\n")
            f.write(f"  Method used: {results['shap'].get('method', 'Unknown')}\n")
            if 'shap_scores_per_snp' in results['shap'] and results['shap']['shap_scores_per_snp'] is not None:
                shap_df = results['shap']['shap_scores_per_snp']
                f.write(f"  Total SNPs analyzed: {len(shap_df)}\n")
                f.write(f"  Top SNP: {shap_df.iloc[0]['SNP_Name']} (score: {shap_df.iloc[0]['Mean_Absolute_SHAP']:.8f})\n")
                f.write(f"  Mean SHAP score: {shap_df['Mean_Absolute_SHAP'].mean():.8f}\n")
                f.write(f"  95th percentile score: {shap_df['Mean_Absolute_SHAP'].quantile(0.95):.8f}\n")

        f.write(f"\nOutput Files Generated:\n")
        for root, dirs, files in os.walk(shap_output_dir):
            for file in files:
                if file.endswith(('.csv', '.png', '.txt', '.npz')):
                    f.write(f"  - {file}\n")

        f.write(f"\nResults saved to: {shap_output_dir}\n")

    print(f"\nAnalysis completed! Results saved to: {shap_output_dir}")
    print(f"Summary saved to: {summary_file}")

    # Print file summary
    csv_files = [f for f in os.listdir(shap_output_dir) if f.endswith('.csv')]
    if csv_files:
        print(f"\nSHAP Score Files Generated:")
        for csv_file in sorted(csv_files):
            file_path = os.path.join(shap_output_dir, csv_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"  - {csv_file}: {len(df)} SNPs")
                except:
                    print(f"  - {csv_file}: File created")

    if len(successful_analyses) == 0:
        print("WARNING: All analyses failed. Check your VAE model and data.")
    else:
        print(f"Successfully completed {len(successful_analyses)} out of 4 analyses.")

except Exception as e:
    print(f"Overall analysis failed: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing with standard analysis...")