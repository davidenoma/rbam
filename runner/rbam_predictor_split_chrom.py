"""
RBAM Predictor Split Chromosome
Performs VAE-based phenotype prediction across multiple chromosomes and averages the metrics.
Includes F1 score and AU-PRC metrics.
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
import keras
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Disable JIT compilation in TensorFlow
tf.config.optimizer.set_jit(False)

import utils
from utils import (load_genotype_data_by_chromosome, save_split_chrom_prediction_metrics,
                   cross_validate_classifier_extended, compute_rmse, evaluate_r2)

# Set up GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


@keras.saving.register_keras_serializable(package="Custom", name="VAE")
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop('encoder'))
        decoder = keras.saving.deserialize_keras_object(config.pop('decoder'))
        return cls(encoder=encoder, decoder=decoder)


@keras.saving.register_keras_serializable(package="Custom", name="vae_loss")
def vae_loss(encoder):
    @keras.saving.register_keras_serializable(package="Custom", name="loss")
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
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_chr{chrom}.keras"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath) and not override:
        raise FileExistsError(f"The file {filename} already exists.")
    model.save(filepath)


def load_model_chrom(snp_data_loc, chrom, directory_loc):
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_chr{chrom}.keras"
    filepath = os.path.join(directory_loc, filename)
    if os.path.exists(filepath):
        try:
            return keras.models.load_model(filepath, custom_objects={"VAE": VAE, "vae_loss": vae_loss}, safe_mode=False)
        except Exception as e:
            print(f"Error loading model for chromosome {chrom}: {e}")
            return None
    return None


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_pca_features(X_train, X_test, snp_data, n_components):
    """Extract PCA features from genotype data."""
    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    if n_components < 1:
        return None, None, None

    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(X_train)
    pca_test = pca.transform(X_test)
    pca_full = pca.transform(snp_data)

    scaler = StandardScaler()
    pca_train = scaler.fit_transform(pca_train)
    pca_test = scaler.transform(pca_test)
    pca_full = scaler.transform(pca_full)

    return pca_train, pca_test, pca_full


def extract_rbam_features(vae_model, X_train, X_test, snp_data):
    """Extract RBAM (VAE latent) features from genotype data."""
    encoder = vae_model.encoder
    z_mean_train, _ = tf.split(encoder.predict(X_train), num_or_size_splits=2, axis=1)
    z_mean_test, _ = tf.split(encoder.predict(X_test), num_or_size_splits=2, axis=1)
    z_mean_full, _ = tf.split(encoder.predict(snp_data), num_or_size_splits=2, axis=1)

    scaler = StandardScaler()
    z_mean_train = scaler.fit_transform(z_mean_train)
    z_mean_test = scaler.transform(z_mean_test)
    z_mean_full = scaler.transform(z_mean_full)

    return z_mean_train, z_mean_test, z_mean_full


def prepare_features(feature_mode, X_train, X_test, snp_data, vae_model=None, covariates=None,
                     n_pca_components=10, train_indices=None, test_indices=None):
    """Prepare features based on the selected mode."""
    features_train = None
    features_test = None
    features_full = None

    if feature_mode == 'pca_only':
        pca_train, pca_test, pca_full = extract_pca_features(X_train, X_test, snp_data, n_pca_components)
        if pca_train is None:
            return None, None, None
        features_train, features_test, features_full = pca_train, pca_test, pca_full

    elif feature_mode == 'rbam_only':
        if vae_model is None:
            return None, None, None
        rbam_train, rbam_test, rbam_full = extract_rbam_features(vae_model, X_train, X_test, snp_data)
        features_train, features_test, features_full = rbam_train, rbam_test, rbam_full

    elif feature_mode == 'rbam_covariates':
        if vae_model is None:
            return None, None, None
        rbam_train, rbam_test, rbam_full = extract_rbam_features(vae_model, X_train, X_test, snp_data)

        if covariates is not None:
            n_train = len(rbam_train)
            cov_train = covariates[:n_train] if train_indices is None else covariates[train_indices]
            cov_test = covariates[n_train:] if test_indices is None else covariates[test_indices]

            cov_scaler = StandardScaler()
            cov_train = cov_scaler.fit_transform(cov_train)
            cov_test = cov_scaler.transform(cov_test)
            cov_full = cov_scaler.transform(covariates)

            features_train = np.hstack([rbam_train, cov_train])
            features_test = np.hstack([rbam_test, cov_test])
            features_full = np.hstack([rbam_full, cov_full])
        else:
            features_train, features_test, features_full = rbam_train, rbam_test, rbam_full

    elif feature_mode == 'pca_rbam_covariates':
        pca_train, pca_test, pca_full = extract_pca_features(X_train, X_test, snp_data, n_pca_components)
        if pca_train is None or vae_model is None:
            return None, None, None

        rbam_train, rbam_test, rbam_full = extract_rbam_features(vae_model, X_train, X_test, snp_data)

        if covariates is not None:
            n_train = len(rbam_train)
            cov_train = covariates[:n_train] if train_indices is None else covariates[train_indices]
            cov_test = covariates[n_train:] if test_indices is None else covariates[test_indices]

            cov_scaler = StandardScaler()
            cov_train = cov_scaler.fit_transform(cov_train)
            cov_test = cov_scaler.transform(cov_test)
            cov_full = cov_scaler.transform(covariates)

            features_train = np.hstack([pca_train, rbam_train, cov_train])
            features_test = np.hstack([pca_test, rbam_test, cov_test])
            features_full = np.hstack([pca_full, rbam_full, cov_full])
        else:
            features_train = np.hstack([pca_train, rbam_train])
            features_test = np.hstack([pca_test, rbam_test])
            features_full = np.hstack([pca_full, rbam_full])

    return features_train, features_test, features_full


# ============================================================================
# Classifier Functions
# ============================================================================

def create_logistic_regression_model(C, penalty, class_weights_dict):
    return LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000, class_weight=class_weights_dict)


def create_random_forest_model(n_estimators, max_depth, class_weights_dict):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42,
                                  class_weight=class_weights_dict)


def create_xgboost_model(learning_rate, n_estimators, max_depth):
    return XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                         use_label_encoder=False, eval_metric='logloss')


def create_tf_classifier_model(input_dim, classifier_hidden_dim, activation, learning_rate):
    """Create a TensorFlow neural network classifier."""
    classifier = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(classifier_hidden_dim, activation=activation),
        tf.keras.layers.Dense(classifier_hidden_dim, activation=activation),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss='binary_crossentropy', metrics=['accuracy'], jit_compile=False)
    return classifier


def train_and_evaluate_classifier(features_train, features_test, y_train, y_test, classifier_type='logistic_regression'):
    """Train and evaluate a classifier with extended metrics."""

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    if classifier_type == 'logistic_regression':
        model = create_logistic_regression_model(C=1.0, penalty='l2', class_weights_dict=class_weights_dict)
        model.fit(features_train, y_train)
        y_pred_proba = model.predict_proba(features_test)[:, 1]
    elif classifier_type == 'random_forest':
        model = create_random_forest_model(n_estimators=100, max_depth=10, class_weights_dict=class_weights_dict)
        model.fit(features_train, y_train)
        y_pred_proba = model.predict_proba(features_test)[:, 1]
    elif classifier_type == 'xgboost':
        model = create_xgboost_model(learning_rate=0.1, n_estimators=100, max_depth=5)
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
        model.set_params(scale_pos_weight=scale_pos_weight)
        model.fit(features_train, y_train)
        y_pred_proba = model.predict_proba(features_test)[:, 1]
    elif classifier_type == 'tf_classifier':
        model = create_tf_classifier_model(
            input_dim=features_train.shape[1],
            classifier_hidden_dim=128,
            activation='relu',
            learning_rate=0.001
        )
        model.fit(
            features_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weights_dict,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        y_pred_proba = model.predict(features_test, verbose=0).ravel()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred_proba)

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'auprc': auprc
    }


def train_and_evaluate_all_classifiers(features_train, features_test, y_train, y_test):
    """Train and evaluate all four classifiers and return metrics for each."""
    classifier_types = ['logistic_regression', 'random_forest', 'xgboost', 'tf_classifier']
    all_metrics = {}

    for classifier_type in classifier_types:
        try:
            metrics = train_and_evaluate_classifier(
                features_train, features_test, y_train, y_test, classifier_type
            )
            all_metrics[classifier_type] = metrics
            print(f"    {classifier_type}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, "
                  f"F1={metrics['f1']:.4f}, AUC-PR={metrics['auprc']:.4f}")
        except Exception as e:
            print(f"    Error with {classifier_type}: {e}")
            all_metrics[classifier_type] = None

    return all_metrics


def train_vae_for_chromosome(X_train, snp_data_loc, chrom, directory, max_evals=5):
    """Train a VAE model for a specific chromosome."""
    input_dim = X_train.shape[1]

    if input_dim < 5:
        return None

    latent_dim_options = [4, 8, 16]
    if input_dim > 50:
        latent_dim_options.extend([32, int(input_dim * 0.2)])

    space = {
        'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', range(1, 5)),
        'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', range(1, 5)),
        'encoding_dimensions': hp.choice('encoding_dimensions', [64, 128]),
        'decoding_dimensions': hp.choice('decoding_dimensions', [64, 128]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'learning_rate': hp.choice('learning_rate', [0.001]),
        'epochs': hp.choice('epochs', [50]),
        'batch_size': hp.choice('batch_size', [32]),
        'latent_dim': hp.choice('latent_dim', [d for d in latent_dim_options if d < input_dim])
    }

    def objective(params):
        model = create_vae_model(input_dim=input_dim, **params)
        history = model.fit(X_train, X_train, epochs=params['epochs'], batch_size=params['batch_size'],
                            validation_split=0.25,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                            verbose=0)
        return {'loss': history.history['val_loss'][-1], 'status': STATUS_OK}

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    best_hyperparameters = space_eval(space, best)

    best_model = create_vae_model(input_dim=input_dim, **best_hyperparameters)
    best_model.fit(X_train, X_train, epochs=best_hyperparameters['epochs'],
                   batch_size=best_hyperparameters['batch_size'], validation_split=0.25,
                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                   verbose=0)

    save_model_chrom(best_model, snp_data_loc, chrom, directory)
    return best_model


def main():
    parser = argparse.ArgumentParser(description='RBAM Split Chromosome Predictor')
    parser.add_argument('snp_data_loc', type=str, help='Path to SNP data file (.raw)')
    parser.add_argument('bim_file', type=str, help='Path to BIM file')
    parser.add_argument('--feature_mode', type=str, default='rbam_only',
                        choices=['pca_only', 'rbam_only', 'rbam_covariates', 'pca_rbam_covariates'],
                        help='Feature mode')
    parser.add_argument('--covariate_file', type=str, default=None,
                        help='Path to covariate file')
    parser.add_argument('--n_pca_components', type=int, default=10,
                        help='Number of PCA components')
    parser.add_argument('--classifier', type=str, default='all',
                        choices=['all', 'logistic_regression', 'random_forest', 'xgboost', 'tf_classifier'],
                        help='Classifier type (default: all - uses all four classifiers)')
    parser.add_argument('--max_evals', type=int, default=5,
                        help='Maximum hyperparameter evaluations per chromosome')

    args = parser.parse_args()

    snp_data_loc = args.snp_data_loc
    bim_file = args.bim_file
    feature_mode = args.feature_mode
    covariate_file = args.covariate_file
    n_pca_components = args.n_pca_components
    classifier_type = args.classifier
    max_evals = args.max_evals

    hopt = f"hopt_split_chrom/{feature_mode}/{classifier_type}"
    directory = f"{os.getcwd()}/model_split_chrom"

    print(f"\n{'=' * 60}")
    print("RBAM Split Chromosome Predictor")
    print(f"{'=' * 60}")
    print(f"SNP data: {snp_data_loc}")
    print(f"BIM file: {bim_file}")
    print(f"Feature mode: {feature_mode}")
    print(f"Classifier: {classifier_type}")
    print(f"{'=' * 60}\n")

    # Load covariates if provided
    covariates = None
    if covariate_file and os.path.exists(covariate_file):
        cov_df = pd.read_csv(covariate_file, sep='\t')
        if 'FID' in cov_df.columns:
            cov_df = cov_df.drop(columns=['FID'])
        if 'IID' in cov_df.columns:
            cov_df = cov_df.drop(columns=['IID'])
        covariates = cov_df.values

    # Load data split by chromosome
    chromosome_data = load_genotype_data_by_chromosome(snp_data_loc, bim_file)

    if not chromosome_data:
        print("Error: No chromosome data loaded")
        sys.exit(1)

    print(f"Loaded data for {len(chromosome_data)} chromosomes")

    # Store metrics for each chromosome (organized by classifier if using 'all')
    all_prediction_metrics = {}  # {chrom: {classifier_type: metrics}} or {chrom: metrics}
    all_reconstruction_metrics = {}

    # Process each chromosome
    for chrom, data in chromosome_data.items():
        X_train, X_test, snp_data, phenotype, y_train, y_test = data

        # Convert labels: 1 to 0 and 2 to 1
        y_train = np.where(y_train == 1, 0, 1)
        y_test = np.where(y_test == 1, 0, 1)

        print(f"\n{'=' * 40}")
        print(f"Processing Chromosome {chrom}")
        print(f"Number of SNPs: {X_train.shape[1]}")
        print(f"Number of samples (train): {X_train.shape[0]}")
        print(f"{'=' * 40}")

        if X_train.shape[1] < 5:
            print(f"Skipping chromosome {chrom}: too few SNPs ({X_train.shape[1]})")
            continue

        # Get or train VAE model (if needed)
        vae_model = None
        if feature_mode in ['rbam_only', 'rbam_covariates', 'pca_rbam_covariates']:
            vae_model = load_model_chrom(snp_data_loc, chrom, directory)
            if vae_model is None:
                vae_model = train_vae_for_chromosome(X_train, snp_data_loc, chrom, directory, max_evals)

            if vae_model is not None:
                # Compute reconstruction metrics
                reconstructed_test = vae_model.predict(X_test)
                reconstructed_full = vae_model.predict(snp_data)

                mse_test = compute_rmse(X_test, reconstructed_test) ** 2
                mse_whole = compute_rmse(snp_data, reconstructed_full) ** 2
                r2_test = np.mean(evaluate_r2(X_test, reconstructed_test))
                r2_whole = np.mean(evaluate_r2(snp_data, reconstructed_full))

                all_reconstruction_metrics[chrom] = {
                    'mse_test': mse_test,
                    'mse_whole': mse_whole,
                    'r2_test': r2_test,
                    'r2_whole': r2_whole
                }

        # Prepare features
        try:
            features_train, features_test, features_full = prepare_features(
                feature_mode, X_train, X_test, snp_data, vae_model, covariates, n_pca_components
            )
        except Exception as e:
            print(f"Error preparing features for chromosome {chrom}: {e}")
            continue

        if features_train is None:
            print(f"Skipping chromosome {chrom}: could not prepare features")
            continue

        # Train and evaluate classifier(s)
        try:
            if classifier_type == 'all':
                # Use all four classifiers
                print(f"  Training all classifiers...")
                chrom_metrics = train_and_evaluate_all_classifiers(
                    features_train, features_test, y_train, y_test
                )
                all_prediction_metrics[chrom] = chrom_metrics
            else:
                # Use single classifier
                metrics = train_and_evaluate_classifier(
                    features_train, features_test, y_train, y_test, classifier_type
                )
                all_prediction_metrics[chrom] = {classifier_type: metrics}

                print(f"Chromosome {chrom} Prediction Metrics ({classifier_type}):")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  AUC-PR: {metrics['auprc']:.4f}")
        except Exception as e:
            print(f"Error training classifier for chromosome {chrom}: {e}")
            continue

    # Save and print prediction metrics
    if all_prediction_metrics:
        # Determine classifiers used
        classifiers_used = list(next(iter(all_prediction_metrics.values())).keys())

        print(f"\n{'=' * 60}")
        print("Average Prediction Metrics Across All Chromosomes")
        print(f"{'=' * 60}")

        for clf_type in classifiers_used:
            # Get metrics for this classifier across all chromosomes
            clf_metrics = [m[clf_type] for m in all_prediction_metrics.values()
                          if clf_type in m and m[clf_type] is not None]

            if clf_metrics:
                avg_accuracy = np.mean([m['accuracy'] for m in clf_metrics])
                avg_auc = np.mean([m['auc'] for m in clf_metrics])
                avg_f1 = np.mean([m['f1'] for m in clf_metrics])
                avg_auprc = np.mean([m['auprc'] for m in clf_metrics])

                print(f"\n{clf_type}:")
                print(f"  Average Accuracy: {avg_accuracy:.4f}")
                print(f"  Average AUC: {avg_auc:.4f}")
                print(f"  Average F1 Score: {avg_f1:.4f}")
                print(f"  Average AUC-PR: {avg_auprc:.4f}")

                # Save metrics for each classifier
                save_split_chrom_prediction_metrics(
                    snp_data_loc,
                    {chrom: m[clf_type] for chrom, m in all_prediction_metrics.items()
                     if clf_type in m and m[clf_type] is not None},
                    hopt=f"{hopt}/{clf_type}"
                )

        print(f"{'=' * 60}")

    if all_reconstruction_metrics:
        avg_mse = np.mean([m['mse_whole'] for m in all_reconstruction_metrics.values()])
        avg_r2 = np.mean([m['r2_whole'] for m in all_reconstruction_metrics.values()])

        print(f"\nAverage Reconstruction Metrics Across All Chromosomes")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average R²: {avg_r2:.6f}")


if __name__ == "__main__":
    main()

