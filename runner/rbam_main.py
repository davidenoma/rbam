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

# Extract feature importance
feature_importance_decoder = utils.extract_decoder_reconstruction_weights(best_model)
feature_importance_encoder = utils.extract_encoder_weights(best_model)

# Load BIM file
# bim = pd.read_csv(sys.argv[2], sep="\t")
#
# Obtain SNPs and weights for both decoder and encoder
# utils.obtain_snps_and_weights(X_train, feature_importance_decoder, bim, snp_data_loc, "decoder", hopt=hopt)
# utils.obtain_snps_and_weights(X_train, feature_importance_encoder, bim, snp_data_loc, "encoder", hopt=hopt)

# Add this to your main execution section after VAE training:

print("Starting SHAP analysis for VAE interpretability...")

# Create output directory for SHAP results
shap_output_dir = f"shap_analysis/{os.path.splitext(os.path.basename(snp_data_loc))[0]}"
os.makedirs(shap_output_dir, exist_ok=True)

try:
    # Perform SHAP analysis
    shap_results = rbam_shap_analysis.explain_vae_with_shap_robust(best_model, X_train, sample_size=100)

    # Create SHAP visualizations
    rbam_shap_analysis.plot_shap_analysis_robust(shap_results, save_path=shap_output_dir)

    # Analyze individual latent dimensions
    dimension_analysis = rbam_shap_analysis.analyze_feature_importance_alternative(best_model, X_train)

    # Comprehensive latent space analysis
    if 'y_train' in locals():
        latent_analysis = rbam_shap_analysis.analyze_latent_space_interpretability(
            best_model, X_train, y_train, save_path=shap_output_dir
        )
    else:
        latent_analysis = rbam_shap_analysis.analyze_latent_space_interpretability(
            best_model, X_train, save_path=shap_output_dir
        )

    # Save SHAP values for further analysis
    np.savez(
        os.path.join(shap_output_dir, "shap_values.npz"),
        encoder_shap_values=shap_results['encoder_shap_values'],
        reconstruction_shap_values=shap_results['reconstruction_shap_values'],
        explain_data=shap_results['explain_data']
    )

    print(f"SHAP analysis completed. Results saved to: {shap_output_dir}")

except Exception as e:
    print(f"SHAP analysis failed: {e}")
    print("Continuing with standard analysis...")


