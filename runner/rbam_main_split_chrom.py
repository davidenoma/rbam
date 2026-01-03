"""
RBAM Main Split Chromosome
Performs VAE reconstruction across multiple chromosomes and averages the metrics.
"""
import os
import sys
import argparse

# CRITICAL: Completely disable XLA JIT compilation to prevent GEMM autotuning crashes
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'  # -1 completely disables XLA JIT
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # Disable cuDNN autotuning

import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, hp, tpe, space_eval
from matplotlib import pyplot as plt

# Disable JIT compilation in TensorFlow
tf.config.optimizer.set_jit(False)

import utils
from utils import (load_genotype_data_by_chromosome, save_split_chrom_reconstruction_metrics,
                   compute_rmse, evaluate_r2, cross_validate_vae)

# Set up GPU memory growth
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
    def from_config(cls, config):
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
    @tf.keras.utils.register_keras_serializable(package="Custom", name="loss")
    def loss(x, x_reconstructed):
        z_mean, z_log_var = tf.split(encoder(x), num_or_size_splits=2, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = tf.maximum(reconstruction_loss + kl_loss, 0)
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
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_function, jit_compile=False)

    return vae


def save_model_chrom(model, snp_data_loc, chrom, directory, override=True):
    """Save a model for a specific chromosome."""
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_chr{chrom}.keras"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    print(f"Saving model for chromosome {chrom} to: {filepath}")
    if os.path.exists(filepath) and not override:
        raise FileExistsError(f"The file {filename} already exists. Set override=True to overwrite.")
    model.save(filepath)


def load_model_chrom(snp_data_loc, chrom, directory_loc):
    """Load a model for a specific chromosome."""
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_chr{chrom}.keras"
    filepath = os.path.join(directory_loc, filename)
    if os.path.exists(filepath):
        print(f'Loading saved model for chromosome {chrom}: ', filepath)
        try:
            return tf.keras.models.load_model(filepath, custom_objects={"VAE": VAE, "vae_loss": vae_loss})
        except Exception as e:
            print(f"Error loading model for chromosome {chrom}: {e}")
            return None
    else:
        print(f'Model not found for chromosome {chrom}: ', filepath)
        return None


def train_vae_for_chromosome(X_train, X_test, snp_data, snp_data_loc, chrom, directory, max_evals=10):
    """Train a VAE model for a specific chromosome."""
    input_dim = X_train.shape[1]

    if input_dim == 0:
        print(f"Skipping chromosome {chrom}: no SNPs found")
        return None

    # Calculate latent dim options based on input dimension
    latent_dim_options = [4, 8, 16, 32, 64]
    if input_dim > 100:
        latent_dim_options.extend([128, int(input_dim * 0.1), int(input_dim * 0.5)])

    space = {
        'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', range(1, 10)),
        'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', range(1, 10)),
        'encoding_dimensions': hp.choice('encoding_dimensions', [64, 128, 256]),
        'decoding_dimensions': hp.choice('decoding_dimensions', [64, 128, 256]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'learning_rate': hp.choice('learning_rate', [0.0001, 0.001]),
        'epochs': hp.choice('epochs', [50, 100]),
        'batch_size': hp.choice('batch_size', [32, 64]),
        'latent_dim': hp.choice('latent_dim', [d for d in latent_dim_options if d < input_dim])
    }

    def objective(params):
        model = create_vae_model(input_dim=input_dim, **params)
        history = model.fit(X_train, X_train,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            validation_split=0.25,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                            verbose=0)
        return {'loss': history.history['val_loss'][-1], 'status': 'ok'}

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    best_hyperparameters = space_eval(space, best)
    print(f"Best hyperparameters for chromosome {chrom}:", best_hyperparameters)

    best_model = create_vae_model(input_dim=input_dim, **best_hyperparameters)
    best_history = best_model.fit(X_train, X_train, epochs=best_hyperparameters['epochs'],
                                  batch_size=best_hyperparameters['batch_size'], validation_split=0.25,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                                  verbose=1)

    save_model_chrom(best_model, snp_data_loc, chrom, directory)
    return best_model


def compute_reconstruction_metrics(model, X_test, snp_data):
    """Compute reconstruction metrics for a model."""
    reconstructed_data_test = model.predict(X_test)
    reconstructed_full_data = model.predict(snp_data)

    mse_test = compute_rmse(X_test, reconstructed_data_test) ** 2
    mse_whole = compute_rmse(snp_data, reconstructed_full_data) ** 2
    r2_test = np.mean(evaluate_r2(X_test, reconstructed_data_test))
    r2_whole = np.mean(evaluate_r2(snp_data, reconstructed_full_data))

    return {
        'mse_test': mse_test,
        'mse_whole': mse_whole,
        'r2_test': r2_test,
        'r2_whole': r2_whole
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RBAM Split Chromosome Reconstruction')
    parser.add_argument('snp_data_loc', type=str, help='Path to SNP data file (.raw)')
    parser.add_argument('bim_file', type=str, help='Path to BIM file')
    parser.add_argument('--phenotype_type', type=str, default='case_control',
                        choices=['quantitative', 'cc_com', 'case_control'],
                        help='Type of phenotype')
    parser.add_argument('--max_evals', type=int, default=5,
                        help='Maximum hyperparameter evaluations per chromosome')

    args = parser.parse_args()

    snp_data_loc = args.snp_data_loc
    bim_file = args.bim_file
    phenotype_type = args.phenotype_type
    max_evals = args.max_evals

    hopt = "hopt_split_chrom"
    directory = f"{os.getcwd()}/model_split_chrom"

    print(f"\n{'=' * 60}")
    print("RBAM Split Chromosome Reconstruction")
    print(f"{'=' * 60}")
    print(f"SNP data: {snp_data_loc}")
    print(f"BIM file: {bim_file}")
    print(f"Phenotype type: {phenotype_type}")
    print(f"{'=' * 60}\n")

    # Load data split by chromosome
    chromosome_data = load_genotype_data_by_chromosome(snp_data_loc, bim_file)

    if not chromosome_data:
        print("Error: No chromosome data loaded")
        sys.exit(1)

    print(f"Loaded data for {len(chromosome_data)} chromosomes")

    # Store metrics for each chromosome
    all_metrics = {}

    # Process each chromosome
    for chrom, data in chromosome_data.items():
        X_train, X_test, snp_data, phenotype, y_train, y_test = data

        print(f"\n{'=' * 40}")
        print(f"Processing Chromosome {chrom}")
        print(f"Number of SNPs: {X_train.shape[1]}")
        print(f"Number of samples (train): {X_train.shape[0]}")
        print(f"{'=' * 40}")

        if X_train.shape[1] < 5:
            print(f"Skipping chromosome {chrom}: too few SNPs ({X_train.shape[1]})")
            continue

        # Try to load existing model
        model = load_model_chrom(snp_data_loc, chrom, directory)

        if model is None:
            # Train new model
            model = train_vae_for_chromosome(X_train, X_test, snp_data, snp_data_loc, chrom, directory, max_evals)

        if model is None:
            print(f"Failed to train model for chromosome {chrom}")
            continue

        # Compute metrics
        metrics = compute_reconstruction_metrics(model, X_test, snp_data)

        # Cross-validation
        try:
            avg_mse_cv, avg_r2_cv = cross_validate_vae(snp_data, model)
            metrics['mse_cv'] = avg_mse_cv
            metrics['r2_cv'] = avg_r2_cv
        except Exception as e:
            print(f"Cross-validation failed for chromosome {chrom}: {e}")

        all_metrics[chrom] = metrics

        print(f"Chromosome {chrom} Metrics:")
        print(f"  MSE (Test): {metrics['mse_test']:.6f}")
        print(f"  MSE (Whole): {metrics['mse_whole']:.6f}")
        print(f"  R² (Test): {metrics['r2_test']:.6f}")
        print(f"  R² (Whole): {metrics['r2_whole']:.6f}")

    # Save all metrics
    save_split_chrom_reconstruction_metrics(snp_data_loc, all_metrics, hopt=hopt)

    # Calculate and print average metrics
    if all_metrics:
        avg_mse_test = np.mean([m['mse_test'] for m in all_metrics.values()])
        avg_mse_whole = np.mean([m['mse_whole'] for m in all_metrics.values()])
        avg_r2_test = np.mean([m['r2_test'] for m in all_metrics.values()])
        avg_r2_whole = np.mean([m['r2_whole'] for m in all_metrics.values()])

        print(f"\n{'=' * 60}")
        print("Average Metrics Across All Chromosomes")
        print(f"{'=' * 60}")
        print(f"Average MSE (Test): {avg_mse_test:.6f}")
        print(f"Average MSE (Whole): {avg_mse_whole:.6f}")
        print(f"Average R² (Test): {avg_r2_test:.6f}")
        print(f"Average R² (Whole): {avg_r2_whole:.6f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
