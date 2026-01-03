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
import json
import matplotlib.pyplot as plt
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, balanced_accuracy_score
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


def save_learning_curve(history, snp_data_loc, chrom, output_dir):
    """Save learning curve plot for a chromosome model."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve - Chromosome {chrom}')
    plt.legend()
    plt.grid(True)

    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_chr{chrom}_learning_curve.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved learning curve: {filepath}")


def save_hyperparameters(hyperparameters, snp_data_loc, chrom, output_dir):
    """Save best hyperparameters to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_chr{chrom}_hyperparameters.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(hyperparameters, f, indent=2)
    print(f"  Saved hyperparameters: {filepath}")


def load_hyperparameters(snp_data_loc, chrom, output_dir):
    """Load hyperparameters from JSON file if exists."""
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_chr{chrom}_hyperparameters.json"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def extract_latent_features_for_fusion(vae_model, X_data):
    """Extract latent features from a VAE model for late fusion."""
    encoder = vae_model.encoder
    z_mean, _ = tf.split(encoder.predict(X_data, verbose=0), num_or_size_splits=2, axis=1)
    return np.array(z_mean)


def combine_latent_spaces(latent_features_dict, sample_indices=None):
    """
    Combine latent spaces from multiple chromosomes (late fusion).

    Args:
        latent_features_dict: Dict with chromosome as key and latent features as value
        sample_indices: Optional indices to select specific samples

    Returns:
        Combined latent features array
    """
    # Sort chromosomes for consistent ordering
    sorted_chroms = sorted(latent_features_dict.keys())

    combined_features = []
    for chrom in sorted_chroms:
        features = latent_features_dict[chrom]
        if sample_indices is not None:
            features = features[sample_indices]
        combined_features.append(features)

    # Concatenate along feature dimension
    combined = np.hstack(combined_features)
    return combined


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
    """Train and evaluate a classifier with extended metrics including independent test and cross-validation."""

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

    # Independent Test Metrics
    ind_test_accuracy = accuracy_score(y_test, y_pred)
    ind_test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
    ind_test_auc = roc_auc_score(y_test, y_pred_proba)
    ind_test_f1 = f1_score(y_test, y_pred)
    ind_test_auprc = average_precision_score(y_test, y_pred_proba)

    # Cross-validation metrics
    try:
        cv_accuracy, cv_auc, cv_f1, cv_auprc = cross_validate_classifier_extended(
            features_train, y_train, model
        )
    except Exception as e:
        print(f"    Cross-validation failed: {e}")
        cv_accuracy, cv_auc, cv_f1, cv_auprc = None, None, None, None

    return {
        # Independent test metrics
        'ind_test_accuracy': ind_test_accuracy,
        'ind_test_balanced_accuracy': ind_test_balanced_acc,
        'ind_test_auc': ind_test_auc,
        'ind_test_f1': ind_test_f1,
        'ind_test_auprc': ind_test_auprc,
        # Cross-validation metrics
        'cv_accuracy': cv_accuracy,
        'cv_auc': cv_auc,
        'cv_f1': cv_f1,
        'cv_auprc': cv_auprc,
        # Keep backward compatibility
        'accuracy': ind_test_accuracy,
        'balanced_accuracy': ind_test_balanced_acc,
        'auc': ind_test_auc,
        'f1': ind_test_f1,
        'auprc': ind_test_auprc
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
            print(f"    {classifier_type}:")
            print(f"      Ind.Test: Acc={metrics['ind_test_accuracy']:.4f}, BalAcc={metrics['ind_test_balanced_accuracy']:.4f}, "
                  f"AUC={metrics['ind_test_auc']:.4f}, F1={metrics['ind_test_f1']:.4f}, AUC-PR={metrics['ind_test_auprc']:.4f}")
            if metrics['cv_accuracy'] is not None:
                print(f"      CV:       Acc={metrics['cv_accuracy']:.4f}, AUC={metrics['cv_auc']:.4f}, "
                      f"F1={metrics['cv_f1']:.4f}, AUC-PR={metrics['cv_auprc']:.4f}")
        except Exception as e:
            print(f"    Error with {classifier_type}: {e}")
            all_metrics[classifier_type] = None

    return all_metrics


def train_vae_for_chromosome(X_train, snp_data_loc, chrom, directory, max_evals=5, output_dir=None):
    """Train a VAE model for a specific chromosome with learning curve and hyperparameter saving."""
    input_dim = X_train.shape[1]

    if input_dim < 5:
        return None, None

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "model_outputs", "split_chrom_training")

    # Calculate latent dim options based on input dimension
    latent_dim_options = [256, 512, 1024,
                          int(input_dim * 0.01),
                          int(input_dim * 0.05),
                          int(input_dim * 0.1),
                          int(input_dim * 0.5)]
    # Filter out latent dims that are >= input_dim or <= 0
    latent_dim_options = [d for d in latent_dim_options if 0 < d < input_dim]

    if not latent_dim_options:
        latent_dim_options = [min(32, input_dim - 1)]

    space = {
        'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', range(1, 5)),
        'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', range(1, 5)),
        'encoding_dimensions': hp.choice('encoding_dimensions', [128, 256, 512]),
        'decoding_dimensions': hp.choice('decoding_dimensions', [128, 256, 512]),
        'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
        'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001]),
        'epochs': hp.choice('epochs', [50, 100]),
        'batch_size': hp.choice('batch_size', [8, 16, 32, 64]),
        'latent_dim': hp.choice('latent_dim', latent_dim_options)
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

    print(f"  Best hyperparameters for chromosome {chrom}: {best_hyperparameters}")

    # Train final model with best hyperparameters
    best_model = create_vae_model(input_dim=input_dim, **best_hyperparameters)
    history = best_model.fit(X_train, X_train, epochs=best_hyperparameters['epochs'],
                             batch_size=best_hyperparameters['batch_size'], validation_split=0.25,
                             callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                             verbose=1)

    # Save model, learning curve, and hyperparameters
    save_model_chrom(best_model, snp_data_loc, chrom, directory)
    save_learning_curve(history, snp_data_loc, chrom, output_dir)
    save_hyperparameters(best_hyperparameters, snp_data_loc, chrom, output_dir)

    return best_model, best_hyperparameters


def main():
    parser = argparse.ArgumentParser(description='RBAM Split Chromosome Predictor')
    parser.add_argument('snp_data_loc', type=str, help='Path to SNP data file (.raw)')
    parser.add_argument('bim_file', type=str, help='Path to BIM file')
    parser.add_argument('--feature_mode', type=str, default='rbam_only',
                        choices=['pca_only', 'rbam_only', 'rbam_covariates', 'pca_rbam_covariates', 'late_fusion'],
                        help='Feature mode (late_fusion combines all chromosome latent spaces)')
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
    output_dir = os.path.join(os.getcwd(), "model_outputs", "split_chrom_training")

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

    # =========================================================================
    # Late Fusion Mode: Load/train all chromosome models, combine latent spaces
    # =========================================================================
    if feature_mode == 'late_fusion':
        print(f"\n{'=' * 60}")
        print("LATE FUSION MODE: Combining latent spaces from all chromosomes")
        print(f"{'=' * 60}\n")

        # Store latent features for each chromosome
        latent_features_train = {}
        latent_features_test = {}
        latent_features_full = {}
        all_hyperparameters = {}
        all_reconstruction_metrics = {}

        # Get consistent train/test indices from first chromosome
        first_chrom = list(chromosome_data.keys())[0]
        first_data = chromosome_data[first_chrom]
        n_samples = first_data[2].shape[0]  # snp_data shape

        # Process each chromosome to get latent features
        for chrom, data in chromosome_data.items():
            X_train, X_test, snp_data, phenotype, y_train, y_test = data

            print(f"\n{'=' * 40}")
            print(f"Processing Chromosome {chrom} for Late Fusion")
            print(f"Number of SNPs: {X_train.shape[1]}")
            print(f"{'=' * 40}")

            if X_train.shape[1] < 5:
                print(f"Skipping chromosome {chrom}: too few SNPs ({X_train.shape[1]})")
                continue

            # Load or train VAE model
            vae_model = load_model_chrom(snp_data_loc, chrom, directory)
            hyperparams = load_hyperparameters(snp_data_loc, chrom, output_dir)

            if vae_model is None:
                print(f"  Training new VAE model for chromosome {chrom}...")
                result = train_vae_for_chromosome(X_train, snp_data_loc, chrom, directory, max_evals, output_dir)
                if result[0] is not None:
                    vae_model, hyperparams = result
                else:
                    print(f"  Failed to train model for chromosome {chrom}")
                    continue
            else:
                print(f"  Loaded existing model for chromosome {chrom}")

            if hyperparams:
                all_hyperparameters[chrom] = hyperparams

            # Extract latent features
            z_train = extract_latent_features_for_fusion(vae_model, X_train)
            z_test = extract_latent_features_for_fusion(vae_model, X_test)
            z_full = extract_latent_features_for_fusion(vae_model, snp_data)

            # Scale latent features
            scaler = StandardScaler()
            z_train = scaler.fit_transform(z_train)
            z_test = scaler.transform(z_test)
            z_full = scaler.transform(z_full)

            latent_features_train[chrom] = z_train
            latent_features_test[chrom] = z_test
            latent_features_full[chrom] = z_full

            # Compute reconstruction metrics
            reconstructed_test = vae_model.predict(X_test, verbose=0)
            reconstructed_full = vae_model.predict(snp_data, verbose=0)

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

            print(f"  Latent dim: {z_train.shape[1]}, MSE: {mse_whole:.6f}, R²: {r2_whole:.6f}")

        if len(latent_features_train) == 0:
            print("Error: No chromosome latent features extracted")
            sys.exit(1)

        # Combine latent spaces (late fusion)
        print(f"\n{'=' * 60}")
        print(f"Combining {len(latent_features_train)} chromosome latent spaces...")
        print(f"{'=' * 60}")

        combined_train = combine_latent_spaces(latent_features_train)
        combined_test = combine_latent_spaces(latent_features_test)
        combined_full = combine_latent_spaces(latent_features_full)

        print(f"Combined latent space dimensions: {combined_train.shape[1]}")

        # Add covariates if provided
        if covariates is not None:
            n_train = combined_train.shape[0]
            cov_train = covariates[:n_train]
            cov_test = covariates[n_train:n_train + combined_test.shape[0]]

            cov_scaler = StandardScaler()
            cov_train = cov_scaler.fit_transform(cov_train)
            cov_test = cov_scaler.transform(cov_test)

            combined_train = np.hstack([combined_train, cov_train])
            combined_test = np.hstack([combined_test, cov_test])
            print(f"Added covariates. Final feature dimensions: {combined_train.shape[1]}")

        # Get labels from first chromosome (they should be the same across all)
        _, _, _, _, y_train, y_test = chromosome_data[first_chrom]
        y_train = np.where(y_train == 1, 0, 1)
        y_test = np.where(y_test == 1, 0, 1)

        # Train and evaluate classifiers on combined latent space
        print(f"\n{'=' * 60}")
        print("Training classifiers on combined (fused) latent space")
        print(f"{'=' * 60}")

        if classifier_type == 'all':
            fusion_metrics = train_and_evaluate_all_classifiers(
                combined_train, combined_test, y_train, y_test
            )
        else:
            fusion_metrics = {classifier_type: train_and_evaluate_classifier(
                combined_train, combined_test, y_train, y_test, classifier_type
            )}

        # Save late fusion results
        print(f"\n{'=' * 60}")
        print("LATE FUSION RESULTS")
        print(f"{'=' * 60}")

        for clf_type, metrics in fusion_metrics.items():
            if metrics is not None:
                print(f"\n{clf_type}:")
                print(f"  Independent Test Metrics:")
                print(f"    Accuracy: {metrics['ind_test_accuracy']:.4f}")
                print(f"    Balanced Accuracy: {metrics['ind_test_balanced_accuracy']:.4f}")
                print(f"    AUC: {metrics['ind_test_auc']:.4f}")
                print(f"    F1 Score: {metrics['ind_test_f1']:.4f}")
                print(f"    AUC-PR: {metrics['ind_test_auprc']:.4f}")
                if metrics['cv_accuracy'] is not None:
                    print(f"  Cross-Validation Metrics:")
                    print(f"    Accuracy: {metrics['cv_accuracy']:.4f}")
                    print(f"    AUC: {metrics['cv_auc']:.4f}")
                    print(f"    F1 Score: {metrics['cv_f1']:.4f}")
                    print(f"    AUC-PR: {metrics['cv_auprc']:.4f}")

        # Save fusion metrics to file
        fusion_output_dir = os.path.join(os.getcwd(), "model_outputs", "late_fusion")
        os.makedirs(fusion_output_dir, exist_ok=True)

        fusion_results_file = os.path.join(fusion_output_dir,
                                           f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_late_fusion_results.txt")
        with open(fusion_results_file, 'w') as f:
            f.write("LATE FUSION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of chromosomes used: {len(latent_features_train)}\n")
            f.write(f"Combined latent space dimensions: {combined_train.shape[1]}\n")
            f.write(f"Training samples: {combined_train.shape[0]}\n")
            f.write(f"Test samples: {combined_test.shape[0]}\n\n")

            for clf_type, metrics in fusion_metrics.items():
                if metrics is not None:
                    f.write(f"\n{clf_type}:\n")
                    f.write(f"  Independent Test Metrics:\n")
                    f.write(f"    Accuracy: {metrics['ind_test_accuracy']:.4f}\n")
                    f.write(f"    Balanced Accuracy: {metrics['ind_test_balanced_accuracy']:.4f}\n")
                    f.write(f"    AUC: {metrics['ind_test_auc']:.4f}\n")
                    f.write(f"    F1 Score: {metrics['ind_test_f1']:.4f}\n")
                    f.write(f"    AUC-PR: {metrics['ind_test_auprc']:.4f}\n")
                    if metrics['cv_accuracy'] is not None:
                        f.write(f"  Cross-Validation Metrics:\n")
                        f.write(f"    Accuracy: {metrics['cv_accuracy']:.4f}\n")
                        f.write(f"    AUC: {metrics['cv_auc']:.4f}\n")
                        f.write(f"    F1 Score: {metrics['cv_f1']:.4f}\n")
                        f.write(f"    AUC-PR: {metrics['cv_auprc']:.4f}\n")

        print(f"\nResults saved to: {fusion_results_file}")

        # Save all hyperparameters summary
        all_hyperparams_file = os.path.join(fusion_output_dir,
                                            f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_all_hyperparameters.json")
        with open(all_hyperparams_file, 'w') as f:
            json.dump(all_hyperparameters, f, indent=2)
        print(f"All hyperparameters saved to: {all_hyperparams_file}")

        # Print reconstruction metrics summary
        if all_reconstruction_metrics:
            avg_mse = np.mean([m['mse_whole'] for m in all_reconstruction_metrics.values()])
            avg_r2 = np.mean([m['r2_whole'] for m in all_reconstruction_metrics.values()])
            print(f"\nAverage Reconstruction Metrics Across All Chromosomes:")
            print(f"  Average MSE: {avg_mse:.6f}")
            print(f"  Average R²: {avg_r2:.6f}")

        return  # Exit after late fusion mode

    # =========================================================================
    # Standard Mode: Process each chromosome independently
    # =========================================================================
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
    all_hyperparameters = {}

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
            hyperparams = load_hyperparameters(snp_data_loc, chrom, output_dir)

            if vae_model is None:
                result = train_vae_for_chromosome(X_train, snp_data_loc, chrom, directory, max_evals, output_dir)
                if result[0] is not None:
                    vae_model, hyperparams = result
                    all_hyperparameters[chrom] = hyperparams
            else:
                print(f"  Loaded existing model for chromosome {chrom}")
                if hyperparams:
                    all_hyperparameters[chrom] = hyperparams

            if vae_model is not None:
                # Compute reconstruction metrics
                reconstructed_test = vae_model.predict(X_test, verbose=0)
                reconstructed_full = vae_model.predict(snp_data, verbose=0)

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
                print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
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
                # Independent test metrics averages
                avg_ind_accuracy = np.mean([m['ind_test_accuracy'] for m in clf_metrics])
                avg_ind_balanced_acc = np.mean([m['ind_test_balanced_accuracy'] for m in clf_metrics])
                avg_ind_auc = np.mean([m['ind_test_auc'] for m in clf_metrics])
                avg_ind_f1 = np.mean([m['ind_test_f1'] for m in clf_metrics])
                avg_ind_auprc = np.mean([m['ind_test_auprc'] for m in clf_metrics])

                # Cross-validation metrics averages (filter out None values)
                cv_metrics = [m for m in clf_metrics if m['cv_accuracy'] is not None]
                if cv_metrics:
                    avg_cv_accuracy = np.mean([m['cv_accuracy'] for m in cv_metrics])
                    avg_cv_auc = np.mean([m['cv_auc'] for m in cv_metrics])
                    avg_cv_f1 = np.mean([m['cv_f1'] for m in cv_metrics])
                    avg_cv_auprc = np.mean([m['cv_auprc'] for m in cv_metrics])
                else:
                    avg_cv_accuracy = avg_cv_auc = avg_cv_f1 = avg_cv_auprc = None

                print(f"\n{clf_type}:")
                print(f"  Independent Test Metrics:")
                print(f"    Average Accuracy: {avg_ind_accuracy:.4f}")
                print(f"    Average Balanced Accuracy: {avg_ind_balanced_acc:.4f}")
                print(f"    Average AUC: {avg_ind_auc:.4f}")
                print(f"    Average F1 Score: {avg_ind_f1:.4f}")
                print(f"    Average AUC-PR: {avg_ind_auprc:.4f}")
                if avg_cv_accuracy is not None:
                    print(f"  Cross-Validation Metrics:")
                    print(f"    Average Accuracy: {avg_cv_accuracy:.4f}")
                    print(f"    Average AUC: {avg_cv_auc:.4f}")
                    print(f"    Average F1 Score: {avg_cv_f1:.4f}")
                    print(f"    Average AUC-PR: {avg_cv_auprc:.4f}")

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

    # Save all hyperparameters summary
    if all_hyperparameters:
        all_hyperparams_file = os.path.join(output_dir,
                                            f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_all_hyperparameters.json")
        with open(all_hyperparams_file, 'w') as f:
            json.dump(all_hyperparameters, f, indent=2)
        print(f"\nAll hyperparameters saved to: {all_hyperparams_file}")

    # =========================================================================
    # Late Fusion (Default): Combine all chromosome latent spaces for prediction
    # =========================================================================
    if feature_mode in ['rbam_only', 'rbam_covariates', 'pca_rbam_covariates']:
        print(f"\n{'=' * 60}")
        print("LATE FUSION: Combining latent spaces from all chromosomes")
        print(f"{'=' * 60}\n")

        # Store latent features for each chromosome
        latent_features_train = {}
        latent_features_test = {}

        # Get first chromosome for reference labels
        first_chrom = list(chromosome_data.keys())[0]
        _, _, _, _, y_train_ref, y_test_ref = chromosome_data[first_chrom]
        y_train_ref = np.where(y_train_ref == 1, 0, 1)
        y_test_ref = np.where(y_test_ref == 1, 0, 1)

        # Extract latent features from all chromosome models
        for chrom, data in chromosome_data.items():
            X_train, X_test, snp_data, phenotype, y_train, y_test = data

            if X_train.shape[1] < 5:
                continue

            # Load VAE model
            vae_model = load_model_chrom(snp_data_loc, chrom, directory)

            if vae_model is not None:
                # Extract latent features
                z_train = extract_latent_features_for_fusion(vae_model, X_train)
                z_test = extract_latent_features_for_fusion(vae_model, X_test)

                # Scale latent features
                scaler = StandardScaler()
                z_train = scaler.fit_transform(z_train)
                z_test = scaler.transform(z_test)

                latent_features_train[chrom] = z_train
                latent_features_test[chrom] = z_test

        if len(latent_features_train) > 0:
            # Combine latent spaces (late fusion)
            print(f"Combining {len(latent_features_train)} chromosome latent spaces...")

            combined_train = combine_latent_spaces(latent_features_train)
            combined_test = combine_latent_spaces(latent_features_test)

            print(f"Combined latent space dimensions: {combined_train.shape[1]}")

            # Add covariates if provided
            if covariates is not None:
                n_train = combined_train.shape[0]
                cov_train = covariates[:n_train]
                cov_test = covariates[n_train:n_train + combined_test.shape[0]]

                cov_scaler = StandardScaler()
                cov_train = cov_scaler.fit_transform(cov_train)
                cov_test = cov_scaler.transform(cov_test)

                combined_train = np.hstack([combined_train, cov_train])
                combined_test = np.hstack([combined_test, cov_test])
                print(f"Added covariates. Final feature dimensions: {combined_train.shape[1]}")

            # Train and evaluate classifiers on combined latent space
            print(f"\nTraining classifiers on combined (fused) latent space...")

            if classifier_type == 'all':
                fusion_metrics = train_and_evaluate_all_classifiers(
                    combined_train, combined_test, y_train_ref, y_test_ref
                )
            else:
                fusion_metrics = {classifier_type: train_and_evaluate_classifier(
                    combined_train, combined_test, y_train_ref, y_test_ref, classifier_type
                )}

            # Print late fusion results
            print(f"\n{'=' * 60}")
            print("LATE FUSION RESULTS")
            print(f"{'=' * 60}")

            for clf_type, metrics in fusion_metrics.items():
                if metrics is not None:
                    print(f"\n{clf_type}:")
                    print(f"  Independent Test Metrics:")
                    print(f"    Accuracy: {metrics['ind_test_accuracy']:.4f}")
                    print(f"    Balanced Accuracy: {metrics['ind_test_balanced_accuracy']:.4f}")
                    print(f"    AUC: {metrics['ind_test_auc']:.4f}")
                    print(f"    F1 Score: {metrics['ind_test_f1']:.4f}")
                    print(f"    AUC-PR: {metrics['ind_test_auprc']:.4f}")
                    if metrics['cv_accuracy'] is not None:
                        print(f"  Cross-Validation Metrics:")
                        print(f"    Accuracy: {metrics['cv_accuracy']:.4f}")
                        print(f"    AUC: {metrics['cv_auc']:.4f}")
                        print(f"    F1 Score: {metrics['cv_f1']:.4f}")
                        print(f"    AUC-PR: {metrics['cv_auprc']:.4f}")

            # Save fusion metrics to file
            fusion_output_dir = os.path.join(os.getcwd(), "model_outputs", "late_fusion")
            os.makedirs(fusion_output_dir, exist_ok=True)

            fusion_results_file = os.path.join(fusion_output_dir,
                                               f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_late_fusion_results.txt")
            with open(fusion_results_file, 'w') as f:
                f.write("LATE FUSION RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Number of chromosomes used: {len(latent_features_train)}\n")
                f.write(f"Combined latent space dimensions: {combined_train.shape[1]}\n")
                f.write(f"Training samples: {combined_train.shape[0]}\n")
                f.write(f"Test samples: {combined_test.shape[0]}\n\n")

                for clf_type, metrics in fusion_metrics.items():
                    if metrics is not None:
                        f.write(f"\n{clf_type}:\n")
                        f.write(f"  Independent Test Metrics:\n")
                        f.write(f"    Accuracy: {metrics['ind_test_accuracy']:.4f}\n")
                        f.write(f"    Balanced Accuracy: {metrics['ind_test_balanced_accuracy']:.4f}\n")
                        f.write(f"    AUC: {metrics['ind_test_auc']:.4f}\n")
                        f.write(f"    F1 Score: {metrics['ind_test_f1']:.4f}\n")
                        f.write(f"    AUC-PR: {metrics['ind_test_auprc']:.4f}\n")
                        if metrics['cv_accuracy'] is not None:
                            f.write(f"  Cross-Validation Metrics:\n")
                            f.write(f"    Accuracy: {metrics['cv_accuracy']:.4f}\n")
                            f.write(f"    AUC: {metrics['cv_auc']:.4f}\n")
                            f.write(f"    F1 Score: {metrics['cv_f1']:.4f}\n")
                            f.write(f"    AUC-PR: {metrics['cv_auprc']:.4f}\n")

            print(f"\nLate fusion results saved to: {fusion_results_file}")
            print(f"{'=' * 60}")
        else:
            print("No chromosome models available for late fusion.")


if __name__ == "__main__":
    main()

