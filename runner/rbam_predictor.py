import os
import sys
import argparse

import keras
import numpy as np
import tensorflow as tf
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score, average_precision_score
from sklearn.utils import class_weight, resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Set up environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import utils
from utils import (load_real_genotype_data, cross_validate_classifier, save_classifier_metrics,
                   cross_validate_classifier_extended, save_classifier_metrics_extended)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='RBAM Classifier with multiple feature options')
parser.add_argument('snp_data_loc', type=str, help='Path to SNP data file')
parser.add_argument('--feature_mode', type=str, default='rbam_only',
                    choices=['pca_only', 'rbam_only', 'rbam_covariates', 'pca_rbam_covariates'],
                    help='Feature mode: pca_only, rbam_only, rbam_covariates, or pca_rbam_covariates')
parser.add_argument('--covariate_file', type=str, default=None,
                    help='Path to covariate file (required for rbam_covariates and pca_rbam_covariates modes)')
parser.add_argument('--n_pca_components', type=int, default=10,
                    help='Number of PCA components (default: 10)')

args = parser.parse_args()

snp_data_loc = args.snp_data_loc
feature_mode = args.feature_mode
covariate_file = args.covariate_file
n_pca_components = args.n_pca_components

# Load data
X_train, X_test, snp_data, phenotype, y_train, y_test = load_real_genotype_data(snp_data_loc)
scaler = StandardScaler()

# Convert labels: 1 to 0 and 2 to 1
y_train = np.where(y_train == 1, 0, 1)
y_test = np.where(y_test == 1, 0, 1)
phenotype = np.where(phenotype == 1, 0, 1)

# Extract SNP file name from path
snp_file_name = os.path.basename(snp_data_loc)
hopt = f"rbam_classifier/{feature_mode}"


# Load covariates if provided
def load_covariates(covariate_file, n_samples):
    """Load covariates from file."""
    if covariate_file and os.path.exists(covariate_file):
        import pandas as pd
        covariates = pd.read_csv(covariate_file, sep='\t')
        # Remove FID and IID columns if present
        if 'FID' in covariates.columns:
            covariates = covariates.drop(columns=['FID'])
        if 'IID' in covariates.columns:
            covariates = covariates.drop(columns=['IID'])
        return covariates.values
    return None


covariates = load_covariates(covariate_file, len(snp_data)) if covariate_file else None


# Define the VAE class
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
    """Return a loss function that captures the VAE loss."""
    @keras.saving.register_keras_serializable(package="Custom", name="loss")
    def loss(x, x_reconstructed):
        z_mean, z_log_var = tf.split(encoder(x), num_or_size_splits=2, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = tf.maximum(reconstruction_loss + kl_loss, 0)  # Prevent negative loss
        return total_loss

    return loss


# Custom EarlyStopping callback to prevent loss from going below zero
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if train_loss is not None and train_loss < 0:
            print(f"\nStopping training early at epoch {epoch + 1} as training loss has gone below zero.")
            self.model.stop_training = True
        elif val_loss is not None and val_loss < 0:
            print(f"\nStopping training early at epoch {epoch + 1} as validation loss has gone below zero.")
            self.model.stop_training = True


# Function to create the VAE model
def create_vae_model(input_dim, num_hidden_layers_encoder, num_hidden_layers_decoder, encoding_dimensions,
                     decoding_dimensions, activation, batch_size, epochs,
                     learning_rate, latent_dim):
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
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_function)

    return vae


def save_model(model: tf.keras.Model, snp_data_loc: str, override: bool = True):
    """Save a TensorFlow model to a specified location."""
    directory = f"{os.getcwd()}/model_cc_com_qt"
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    print(f"Saving model to filepath: {filepath}")
    if os.path.exists(filepath) and not override:
        raise FileExistsError(f"The file {filename} already exists. Set override=True to overwrite.")
    model.save(filepath)


def load_model(snp_data_loc):
    """Load a TensorFlow model if it exists."""
    directory = f"{os.getcwd()}/model_cc_com_qt"
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        try:
            return keras.models.load_model(filepath, custom_objects={"VAE": VAE, "vae_loss": vae_loss}, safe_mode=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        return None


# Objective function for VAE
def objective(params):
    model = create_vae_model(input_dim=X_train.shape[1], **params)
    history = model.fit(X_train, X_train, epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_split=0.25,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                            CustomEarlyStopping()
                        ],
                        verbose=1)
    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK}


# Define the search space for VAE hyperparameters
vae_space = {
    'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', range(1, 17)),
    'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', range(1, 17)),
    'encoding_dimensions': hp.choice('encoding_dimensions', [128, 256, 512]),
    'decoding_dimensions': hp.choice('decoding_dimensions', [128, 256, 512]),
    'activation': hp.choice('activation', ['relu', 'sigmoid']),
    'learning_rate': hp.choice('learning_rate', [0.000001, 0.00001, 0.0001, 0.001]),
    'epochs': hp.choice('epochs', [50, 100, 150]),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'latent_dim': hp.choice('latent_dim', [4, 8, 16, 32, 64, 128, 512, 1024, int(X_train.shape[1] * 0.01),
                                           int(X_train.shape[1] * 0.05), int(X_train.shape[1] * 0.1),
                                           int(X_train.shape[1] * 0.5)])
}


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_pca_features(X_train, X_test, snp_data, n_components):
    """Extract PCA features from genotype data."""
    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(X_train)
    pca_test = pca.transform(X_test)
    pca_full = pca.transform(snp_data)

    # Scale PCA features
    scaler = StandardScaler()
    pca_train = scaler.fit_transform(pca_train)
    pca_test = scaler.transform(pca_test)
    pca_full = scaler.transform(pca_full)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    return pca_train, pca_test, pca_full


def extract_rbam_features(vae_model, X_train, X_test, snp_data):
    """Extract RBAM (VAE latent) features from genotype data."""
    encoder = vae_model.encoder
    z_mean_train, _ = tf.split(encoder.predict(X_train), num_or_size_splits=2, axis=1)
    z_mean_test, _ = tf.split(encoder.predict(X_test), num_or_size_splits=2, axis=1)
    z_mean_full, _ = tf.split(encoder.predict(snp_data), num_or_size_splits=2, axis=1)

    # Scale latent space
    scaler = StandardScaler()
    z_mean_train = scaler.fit_transform(z_mean_train)
    z_mean_test = scaler.transform(z_mean_test)
    z_mean_full = scaler.transform(z_mean_full)

    return z_mean_train, z_mean_test, z_mean_full


def prepare_features(feature_mode, X_train, X_test, snp_data, vae_model=None, covariates=None,
                     n_pca_components=10, train_indices=None, test_indices=None):
    """
    Prepare features based on the selected mode.

    Modes:
    - pca_only: Use only PCA components
    - rbam_only: Use only RBAM latent features
    - rbam_covariates: Use RBAM latent features + covariates
    - pca_rbam_covariates: Use PCA + RBAM latent features + covariates
    """
    features_train = None
    features_test = None
    features_full = None

    if feature_mode == 'pca_only':
        pca_train, pca_test, pca_full = extract_pca_features(X_train, X_test, snp_data, n_pca_components)
        features_train = pca_train
        features_test = pca_test
        features_full = pca_full

    elif feature_mode == 'rbam_only':
        if vae_model is None:
            raise ValueError("VAE model is required for rbam_only mode")
        rbam_train, rbam_test, rbam_full = extract_rbam_features(vae_model, X_train, X_test, snp_data)
        features_train = rbam_train
        features_test = rbam_test
        features_full = rbam_full

    elif feature_mode == 'rbam_covariates':
        if vae_model is None:
            raise ValueError("VAE model is required for rbam_covariates mode")
        rbam_train, rbam_test, rbam_full = extract_rbam_features(vae_model, X_train, X_test, snp_data)

        if covariates is not None:
            cov_train = covariates[train_indices] if train_indices is not None else covariates[:len(rbam_train)]
            cov_test = covariates[test_indices] if test_indices is not None else covariates[len(rbam_train):]

            # Scale covariates
            cov_scaler = StandardScaler()
            cov_train = cov_scaler.fit_transform(cov_train)
            cov_test = cov_scaler.transform(cov_test)
            cov_full = cov_scaler.transform(covariates)

            features_train = np.hstack([rbam_train, cov_train])
            features_test = np.hstack([rbam_test, cov_test])
            features_full = np.hstack([rbam_full, cov_full])
        else:
            print("Warning: No covariates provided, using rbam_only mode")
            features_train = rbam_train
            features_test = rbam_test
            features_full = rbam_full

    elif feature_mode == 'pca_rbam_covariates':
        pca_train, pca_test, pca_full = extract_pca_features(X_train, X_test, snp_data, n_pca_components)

        if vae_model is None:
            raise ValueError("VAE model is required for pca_rbam_covariates mode")
        rbam_train, rbam_test, rbam_full = extract_rbam_features(vae_model, X_train, X_test, snp_data)

        if covariates is not None:
            cov_train = covariates[train_indices] if train_indices is not None else covariates[:len(rbam_train)]
            cov_test = covariates[test_indices] if test_indices is not None else covariates[len(rbam_train):]

            # Scale covariates
            cov_scaler = StandardScaler()
            cov_train = cov_scaler.fit_transform(cov_train)
            cov_test = cov_scaler.transform(cov_test)
            cov_full = cov_scaler.transform(covariates)

            features_train = np.hstack([pca_train, rbam_train, cov_train])
            features_test = np.hstack([pca_test, rbam_test, cov_test])
            features_full = np.hstack([pca_full, rbam_full, cov_full])
        else:
            print("Warning: No covariates provided, using pca + rbam features only")
            features_train = np.hstack([pca_train, rbam_train])
            features_test = np.hstack([pca_test, rbam_test])
            features_full = np.hstack([pca_full, rbam_full])

    return features_train, features_test, features_full


# ============================================================================
# Main Execution
# ============================================================================

# Skip VAE training if using PCA only mode
if feature_mode == 'pca_only':
    best_vae_model = None
    print("Using PCA only mode - skipping VAE training")
else:
    # Load or train the best VAE model with error handling
    try:
        best_vae_model = load_model(snp_data_loc)
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        best_vae_model = None

    if not best_vae_model:
        best_vae = fmin(fn=objective, space=vae_space, algo=tpe.suggest, max_evals=10)
        best_vae_hyperparameters = space_eval(vae_space, best_vae)
        print(f"Best hyperparameters for VAE ({snp_file_name}):", best_vae_hyperparameters)
        best_vae_model = create_vae_model(input_dim=X_train.shape[1], **best_vae_hyperparameters)
        best_vae_history = best_vae_model.fit(X_train, X_train, epochs=best_vae_hyperparameters['epochs'],
                                              batch_size=best_vae_hyperparameters['batch_size'],
                                              validation_split=0.25)
        save_model(best_vae_model, snp_data_loc)

# Calculate reconstruction metrics if VAE is used
if best_vae_model is not None:
    reconstructed_data_test = best_vae_model.predict(X_test)
    reconstructed_full_data = best_vae_model.predict(snp_data)

    mse_test = utils.compute_rmse(X_test, reconstructed_data_test) ** 2
    mse_whole = utils.compute_rmse(snp_data, reconstructed_full_data) ** 2
    utils.save_mse_values(snp_data_loc, mse_test, mse_whole, hopt=hopt)

    r2_test = np.mean(utils.evaluate_r2(X_test, reconstructed_data_test))
    r2_whole = np.mean(utils.evaluate_r2(snp_data, reconstructed_full_data))
    utils.save_r2_scores(snp_data_loc, r2_test, r2_whole, hopt=hopt)

    print("MSE (Whole):", mse_whole)
    print("R² (Whole):", r2_whole)

    avg_mse_test, avg_r2_test = utils.cross_validate_vae(snp_data, best_vae_model)
    utils.save_mse_values_cv(snp_data_loc, avg_mse_test, hopt=hopt)
    utils.save_r2_scores_cv(snp_data_loc, avg_r2_test, hopt=hopt)

# Prepare features based on selected mode
features_train, features_test, features_full = prepare_features(
    feature_mode, X_train, X_test, snp_data, best_vae_model, covariates, n_pca_components
)

print(f"\nFeature mode: {feature_mode}")
print(f"Training features shape: {features_train.shape}")
print(f"Test features shape: {features_test.shape}")

# Class weights calculation
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


# Define classifiers
def create_logistic_regression_model(C, penalty, class_weight=None):
    return LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000, class_weight=class_weights_dict)


def create_random_forest_model(n_estimators, max_depth, class_weight=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42,
                                  class_weight=class_weights_dict)


def create_xgboost_model(learning_rate, n_estimators, max_depth, class_weight=None):
    return XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                         use_label_encoder=False, eval_metric='logloss', class_weight=class_weights_dict)


def create_tf_classifier_model(input_dim, classifier_hidden_dim, activation, learning_rate, batch_size, epochs):
    classifier = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(classifier_hidden_dim, activation=activation),
        tf.keras.layers.Dense(classifier_hidden_dim, activation=activation),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# Hyperparameter spaces
classifier_space = {
    'tf_classifier': {
        'classifier_hidden_dim': hp.choice('classifier_hidden_dim', [128, 256, 512]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'learning_rate': hp.choice('learning_rate', [0.00001, 0.0001, 0.001]),
        'epochs': hp.choice('epochs', [50, 100, 150]),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128])
    },
    'logistic_regression': {
        'C': hp.uniform('C', 0.01, 10.0),
        'penalty': hp.choice('penalty', ['l1', 'l2'])
    },
    'random_forest': {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [5, 10, 15, 20, 25, 50])
    },
    'xgboost': {
        'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1, 0.2]),
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [3, 5, 7, 9])
    }
}


def objective_classifier(params, model_type, features_train, y_train, class_weights_dict):
    """Hyperopt objective for classifier optimization."""
    if model_type == 'tf_classifier':
        model = create_tf_classifier_model(input_dim=features_train.shape[1], **params)
        hist = model.fit(
            features_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.20,
            class_weight=class_weights_dict,
            verbose=0
        )
        val_loss = np.min(hist.history['val_loss'])
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            features_train, y_train,
            test_size=0.20, stratify=y_train, random_state=77)

        if model_type == 'logistic_regression':
            model = create_logistic_regression_model(**params)
        elif model_type == 'random_forest':
            model = create_random_forest_model(**params)
        elif model_type == 'xgboost':
            model = create_xgboost_model(**params)
            scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
            model.set_params(scale_pos_weight=scale_pos_weight)

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        val_loss = 1.0 - accuracy_score(y_val, preds)

    return {'loss': val_loss, 'status': STATUS_OK}


# Train and evaluate classifiers
for model_type, space in classifier_space.items():
    print(f"\n{'=' * 60}")
    print(f"Training {model_type} with {feature_mode} features")
    print(f"{'=' * 60}")

    best_classifier = fmin(
        fn=lambda params: objective_classifier(params, model_type, features_train, y_train, class_weights_dict),
        space=space, algo=tpe.suggest, max_evals=20
    )
    best_hyperparameters = space_eval(space, best_classifier)
    print(f"Best hyperparameters for {model_type} ({snp_file_name}): {best_hyperparameters}")

    # Train the model with the best hyperparameters
    if model_type == 'tf_classifier':
        best_model = create_tf_classifier_model(input_dim=features_train.shape[1], **best_hyperparameters)
        best_model.fit(
            features_train, y_train, epochs=best_hyperparameters['epochs'], validation_split=0.25,
            batch_size=best_hyperparameters['batch_size'], class_weight=class_weights_dict,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], verbose=1
        )
    elif model_type == 'logistic_regression':
        best_model = create_logistic_regression_model(**best_hyperparameters)
        best_model.fit(features_train, y_train)
    elif model_type == 'random_forest':
        best_model = create_random_forest_model(**best_hyperparameters)
        best_model.fit(features_train, y_train)
    elif model_type == 'xgboost':
        best_model = create_xgboost_model(**best_hyperparameters)
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        best_model.set_params(scale_pos_weight=scale_pos_weight)
        best_model.fit(features_train, y_train)

    # Evaluate classifier on an independent test set
    if model_type == 'tf_classifier':
        phenotype_predictions_proba = best_model.predict(features_test).ravel()
    else:
        phenotype_predictions_proba = best_model.predict_proba(features_test)[:, 1]

    phenotype_predictions_test = (phenotype_predictions_proba > 0.5).astype(int)

    ind_test_accuracy = accuracy_score(y_test, phenotype_predictions_test)
    ind_test_auc = roc_auc_score(y_test, phenotype_predictions_proba)
    ind_test_f1 = f1_score(y_test, phenotype_predictions_test)
    ind_test_auprc = average_precision_score(y_test, phenotype_predictions_proba)

    print(f"Independent Test Accuracy for {model_type} ({snp_file_name}): {ind_test_accuracy}")
    print(f"Independent Test AUC for {model_type} ({snp_file_name}): {ind_test_auc}")
    print(f"Independent Test F1 Score for {model_type} ({snp_file_name}): {ind_test_f1}")
    print(f"Independent Test AUC-PR for {model_type} ({snp_file_name}): {ind_test_auprc}")

    # Cross-validation with extended metrics
    avg_accuracy_val, avg_auc_val, avg_f1_val, avg_auprc_val = cross_validate_classifier_extended(
        features_train, y_train, best_model
    )

    # Save extended classifier metrics
    save_classifier_metrics_extended(
        snp_data_loc, avg_accuracy_val, avg_auc_val, avg_f1_val, avg_auprc_val,
        ind_test_accuracy, ind_test_auc, ind_test_f1, ind_test_auprc,
        hopt=f"{hopt}/{model_type}"
    )

    print(f"Cross-Validation Accuracy for {model_type} ({snp_file_name}): {avg_accuracy_val}")
    print(f"Cross-Validation AUC for {model_type} ({snp_file_name}): {avg_auc_val}")
    print(f"Cross-Validation F1 Score for {model_type} ({snp_file_name}): {avg_f1_val}")
    print(f"Cross-Validation AUC-PR for {model_type} ({snp_file_name}): {avg_auprc_val}")

print(f"\n{'=' * 60}")
print(f"RBAM Classifier completed with feature mode: {feature_mode}")
print(f"{'=' * 60}")

