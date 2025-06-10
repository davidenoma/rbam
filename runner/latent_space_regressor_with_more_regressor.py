import os
import sys

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import utils
from utils import save_summary, load_real_genotype_data, cross_validate_regressor, save_regressor_metrics, \
    save_mse_values, save_r2_scores, cross_validate_vae, evaluate_r2_ls_reg

# Set up environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load data
snp_data_loc = sys.argv[1]
X_train, X_test, snp_data, phenotype, y_train, y_test = load_real_genotype_data(snp_data_loc)
scaler = StandardScaler()


# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
hopt = "rbam_regressor"

# # Split the training data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
#

@tf.keras.utils.register_keras_serializable(package="Custom", name="VAE")
class VAE(tf.keras.Model):
    """
    Variational Autoencoder (VAE) class that includes an encoder and a decoder.

    Attributes:
        encoder (tf.keras.Sequential): The encoder part of the VAE.
        decoder (tf.keras.Sequential): The decoder part of the VAE.
    """

    def __init__(self, encoder: tf.keras.Sequential, decoder: tf.keras.Sequential, **kwargs):
        """
        Initialize the VAE with an encoder and a decoder.

        Args:
            encoder (tf.keras.Sequential): The encoder model.
            decoder (tf.keras.Sequential): The decoder model.
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        """
        Reparameterize the latent variables using the mean and log variance.

        Args:
            mean (tf.Tensor): Mean of the latent variables.
            log_var (tf.Tensor): Log variance of the latent variables.

        Returns:
            tf.Tensor: Reparameterized latent variables.
        """
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the VAE.

        Args:
            inputs (tf.Tensor): Input data.

        Returns:
            tf.Tensor: Reconstructed data after passing through the VAE.
        """
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed

    def get_config(self) -> dict:
        """
        Get the configuration of the VAE.

        Returns:
            dict: Configuration of the VAE.
        """
        config = super(VAE, self).get_config()
        config.update({
            'encoder': tf.keras.utils.serialize_keras_object(self.encoder),
            'decoder': tf.keras.utils.serialize_keras_object(self.decoder),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'VAE':
        """
        Create a VAE instance from a configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            VAE: A VAE instance.
        """
        encoder = tf.keras.utils.deserialize_keras_object(config.pop('encoder'))
        decoder = tf.keras.utils.deserialize_keras_object(config.pop('decoder'))
        return cls(encoder=encoder, decoder=decoder)


@tf.keras.utils.register_keras_serializable(package="Custom", name="vae_loss")
def vae_loss(encoder: tf.keras.Sequential) -> callable:
    """
    Return a loss function that captures the VAE loss (reconstruction + KL divergence).

    Args:
        encoder (tf.keras.Sequential): The encoder model.

    Returns:
        callable: A function that computes the VAE loss.
    """

    def loss(x: tf.Tensor, x_reconstructed: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var = tf.split(encoder(x), num_or_size_splits=2, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = tf.maximum(reconstruction_loss + kl_loss, 0)  # Prevent negative loss
        return total_loss

    return loss


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



def save_model(model: tf.keras.Model, snp_data_loc: str, override: bool = False):
    """
    Save a TensorFlow model to a specified location.

    Args:
        model (tf.keras.Model): The model to save.
        snp_data_loc (str): The path of the SNP data location.
        override (bool): If True, overwrite the existing file. Defaults to False.
    """
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
        print('Loading the model: ', filepath)
        return keras.models.load_model(filepath, custom_objects={"VAE": VAE, "vae_loss": vae_loss})

    else:
        return None

# Utility to save latent space predictions
def save_latent_space_predictions(z_mean, snp_data_loc, prefix="latent_space"):
    """
    Save latent space predictions to a file.

    Args:
        z_mean (numpy.ndarray): Latent space predictions.
        snp_data_loc (str): Path to the input SNP data file.
        prefix (str): Prefix for the output filename.
    """
    output_dir = "model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(snp_data_loc))[0]
    output_file = os.path.join(output_dir, f"{prefix}_{file_name}.csv")
    print(f"Saving latent space predictions to: {output_file}")
    np.savetxt(output_file, z_mean, delimiter=",", fmt="%.6f")

def create_vae_model(input_dim: int, num_hidden_layers_encoder: int, num_hidden_layers_decoder: int,
                     encoding_dimensions: int, decoding_dimensions: int, activation: str,
                     batch_size: int, epochs: int, learning_rate: float, latent_dim: int) -> VAE:
    """
    Create and return a Variational Autoencoder (VAE) model.

    Args:
        input_dim (int): Dimensionality of the input data.
        num_hidden_layers_encoder (int): Number of hidden layers in the encoder.
        num_hidden_layers_decoder (int): Number of hidden layers in the decoder.
        encoding_dimensions (int): Number of units in each hidden layer of the encoder.
        decoding_dimensions (int): Number of units in each hidden layer of the decoder.
        activation (str): Activation function to use.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        latent_dim (int): Dimensionality of the latent space.

    Returns:
        VAE: A compiled VAE model.
    """
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
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=vae_loss(vae.encoder))

    return vae


# Define the regressor model
def create_regressor_model(input_dim: int, regressor_hidden_dim: int, activation: str,
                           batch_size: int, epochs: int, learning_rate: float) -> tf.keras.Sequential:
    """
    Create and return a regressor model.

    Args:
        input_dim (int): Dimensionality of the input data.
        regressor_hidden_dim (int): Number of units in each hidden layer of the regressor.
        activation (str): Activation function to use.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Sequential: A compiled regressor model.
    """
    regressor = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(regressor_hidden_dim, activation=activation),
        tf.keras.layers.Dense(1)  # No activation for regression
    ])
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error', metrics=['mse'])
    return regressor


# Objective function for VAE
def objective(params):
    model = create_vae_model(input_dim=X_train.shape[1], **params)
    history = model.fit(X_train, X_train, epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_split=0.25,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                            CustomEarlyStopping()  # Custom early stopping to prevent train loss from going below zero
                        ],
                        verbose=1)

    # Return the minimum validation loss instead of the training loss
    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK}


vae_space = {
    # Number of hidden layers for the encoder network
    'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', range(1, 17)),  # Varies from 1 to 16 layers

    # Number of hidden layers for the decoder network
    'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', range(1, 17)),  # Varies from 1 to 16 layers

    # Size of the hidden layers in the encoder
    'encoding_dimensions': hp.choice('encoding_dimensions', [128, 256, 512]),  # Can be 128, 256, or 512 neurons

    # Size of the hidden layers in the decoder
    'decoding_dimensions': hp.choice('decoding_dimensions', [128, 256, 512]),  # Can be 128, 256, or 512 neurons

    # Activation function to be used in hidden layers
    'activation': hp.choice('activation', ['relu', 'sigmoid']),  # Either ReLU or Sigmoid activation function

    # Learning rate for the optimizer
    'learning_rate': hp.choice('learning_rate', [0.000001, 0.00001, 0.0001, 0.001]),  # Ranges from 1e-6 to 1e-3

    # Number of epochs to train the model
    'epochs': hp.choice('epochs', [50, 100, 150]),  # Can be 50, 100, or 150 epochs

    # Batch size for training
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),  # Can be 16, 32, 64, or 128 samples per batch

    # Latent dimension size of the bottleneck layer in the VAE
    'latent_dim': hp.choice('latent_dim', [
        4, 8, 16, 32, 64, 128, 512, 1024,  # Fixed sizes for the latent dimension
        int(X_train.shape[1] * 0.01),  # 1% of the input dimension
        int(X_train.shape[1] * 0.05),  # 5% of the input dimension
        int(X_train.shape[1] * 0.1),  # 10% of the input dimension
        int(X_train.shape[1] * 0.5)  # 50% of the input dimension
    ])
}

best_vae_model = load_model(snp_data_loc)

if not best_vae_model:  # Hyperparameter optimization for VAE
    best_vae = fmin(fn=objective, space=vae_space, algo=tpe.suggest, max_evals=10)

    # Extract the best hyperparameters for VAE
    best_vae_hyperparameters = space_eval(vae_space, best_vae)
    print("Best hyperparameters for VAE:", best_vae_hyperparameters)

    # Create and compile the best VAE model using the best hyperparameters
    best_vae_model = create_vae_model(input_dim=X_train.shape[1], **best_vae_hyperparameters)

    # Train the best VAE model on the entire dataset
    best_vae_history = best_vae_model.fit(X_train, X_train, epochs=best_vae_hyperparameters['epochs'],
                                          batch_size=best_vae_hyperparameters['batch_size'],
                                          validation_split=0.25)

    save_model(best_vae_model, snp_data_loc)

best_model = best_vae_model

# Reconstruct input data using the trained VAE
reconstructed_data_train = best_model.predict(X_train)
reconstructed_data_test = best_model.predict(X_test)
reconstructed_full_data = best_model.predict(snp_data)

# Calculate MSE
mse_train = utils.compute_rmse(X_train, reconstructed_data_train)**2
mse_test = utils.compute_rmse(X_test, reconstructed_data_test)**2
mse_whole = utils.compute_rmse(snp_data, reconstructed_full_data)**2
utils.save_mse_values(snp_data_loc, mse_train, mse_test, mse_whole, hopt=hopt)

# Calculate RMSE
rmse_train = utils.compute_rmse(X_train, reconstructed_data_train)
rmse_test = utils.compute_rmse(X_test, reconstructed_data_test)
rmse_whole = utils.compute_rmse(snp_data, reconstructed_full_data)

# Calculate R²
r2_train = np.mean(utils.evaluate_r2(X_train, reconstructed_data_train))
r2_test = np.mean(utils.evaluate_r2(X_test, reconstructed_data_test))
r2_whole = np.mean(utils.evaluate_r2(snp_data, reconstructed_full_data))
utils.save_r2_scores(snp_data_loc, r2_train, r2_test, r2_whole, hopt=hopt)

# Calculate Pearson Correlation
pearson_corr_train = utils.compute_pearson_correlation(X_train, reconstructed_data_train)
pearson_corr_test = utils.compute_pearson_correlation(X_test, reconstructed_data_test)
pearson_corr_whole = utils.compute_pearson_correlation(snp_data, reconstructed_full_data)

# Save Adjusted R², Pearson Correlation, and RMSE metrics

# Print results
print("MSE (Train):", mse_train)
print("MSE (Test):", mse_test)
print("MSE (Whole):", mse_whole)

print("RMSE (Train):", rmse_train)
print("RMSE (Test):", rmse_test)
print("RMSE (Whole):", rmse_whole)

print("R² (Train):", r2_train)
print("R² (Test):", r2_test)
print("R² (Whole):", r2_whole)


print("Pearson Correlation (Train):", pearson_corr_train)
print("Pearson Correlation (Test):", pearson_corr_test)
print("Pearson Correlation (Whole):", pearson_corr_whole)

print("Cross Validation")

# Perform cross-validation
(
    avg_mse_train, avg_mse_test,
    avg_r2_train, avg_r2_test,
    # avg_adj_r2_train, avg_adj_r2_test,
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
# print("Average Training Adjusted R^2:", avg_adj_r2_train)
# print("Average Testing Adjusted R^2:", avg_adj_r2_test)
print("Average Training Pearson Correlation:", avg_pearson_corr_train)
print("Average Testing Pearson Correlation:", avg_pearson_corr_test)
# Extract latent vectors
encoder = best_vae_model.encoder
z_mean_full, _ = tf.split(encoder.predict(snp_data), num_or_size_splits=2, axis=1)
z_mean_train, _ = tf.split(encoder.predict(X_train), num_or_size_splits=2, axis=1)
z_mean_test, _ = tf.split(encoder.predict(X_test), num_or_size_splits=2, axis=1)

# Save latent space predictions
save_latent_space_predictions(z_mean_full, snp_data_loc, prefix="latent_space_full")
save_latent_space_predictions(z_mean_train, snp_data_loc, prefix="latent_space_train")
save_latent_space_predictions(z_mean_test, snp_data_loc, prefix="latent_space_test")

# Scale latent space
z_mean_full = scaler.fit_transform(z_mean_full)
z_mean_train = scaler.fit_transform(z_mean_train)
z_mean_test = scaler.transform(z_mean_test)


# Define additional regressors
def create_linear_regression_model():
    return LinearRegression()


def create_elastic_net_model(alpha, l1_ratio):
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)


def create_random_forest_model(n_estimators, max_depth):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


def create_xgboost_model(learning_rate, n_estimators, max_depth):
    return XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, eval_metric='rmse')


def create_tf_regressor_model(input_dim, regressor_hidden_dim, activation, learning_rate, batch_size, epochs):
    regressor = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(regressor_hidden_dim, activation=activation),
        tf.keras.layers.Dense(1)  # No activation for regression
    ])
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error', metrics=['mse'])
    return regressor
# Remove 'linear_regression' from regressor_space
regressor_space = {
    'tf_regressor': {
        'regressor_hidden_dim': hp.choice('regressor_hidden_dim', [128, 256, 512]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'learning_rate': hp.choice('learning_rate', [0.00001, 0.0001, 0.001]),
        'epochs': hp.choice('epochs', [50, 100, 150]),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128])
    },
    'elastic_net': {
        'alpha': hp.uniform('alpha', 0.001, 1.0),
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0)
    },
    'random_forest': {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [5, 10, 15, 20])
    },
    'xgboost': {
        'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1, 0.2]),
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [3, 5, 7, 9])
    }
}

# Add linear regression evaluation outside of the fmin loop
if 'linear_regression' not in regressor_space:
    print("Evaluating Linear Regression separately")
    linear_regression_model = create_linear_regression_model()
    linear_regression_model.fit(z_mean_train, y_train)
    phenotype_predictions_test = linear_regression_model.predict(z_mean_test)
    mse_test = mean_squared_error(y_test, phenotype_predictions_test)
    r2_test = r2_score(y_test, phenotype_predictions_test)
    pearson_corr, _ = pearsonr(y_test.values.flatten(), phenotype_predictions_test.flatten())
    print(f"Independent Test MSE for Linear Regression ({snp_data_loc}): {mse_test}")
    print(f"Independent Test R2 for Linear Regression ({snp_data_loc}): {r2_test}")
    print(f"Independent Test Pearson Correlation for Linear Regression ({snp_data_loc}): {pearson_corr}")


# Objective function for hyperparameter optimization
def objective_regressor(params, model_type):
    if model_type == 'tf_regressor':
        model = create_tf_regressor_model(input_dim=z_mean_train.shape[1], **params)
        history = model.fit(z_mean_train, y_train, epochs=params['epochs'], validation_split=0.25,
                            batch_size=params['batch_size'], verbose=1)
        val_loss = np.mean(history.history['val_loss'])
    else:
        if model_type == 'linear_regression':
            model = create_linear_regression_model()
        elif model_type == 'elastic_net':
            model = create_elastic_net_model(**params)
        elif model_type == 'random_forest':
            model = create_random_forest_model(**params)
        elif model_type == 'xgboost':
            model = create_xgboost_model(**params)
        model.fit(z_mean_train, y_train)
        val_loss = mean_squared_error(y_test, model.predict(z_mean_test))

    return {'loss': val_loss, 'status': STATUS_OK}


# Hyperparameter optimization and evaluation loop
for model_type, space in regressor_space.items():
    best_regressor = fmin(fn=lambda params: objective_regressor(params, model_type), space=space, algo=tpe.suggest,
                          max_evals=10)
    best_hyperparameters = space_eval(space, best_regressor)
    print(f"Best hyperparameters for {model_type} ({snp_data_loc}): {best_hyperparameters}")

    # Train and evaluate the model with the best hyperparameters
    if model_type == 'tf_regressor':
        best_model = create_tf_regressor_model(input_dim=z_mean_train.shape[1], **best_hyperparameters)
        best_model.fit(z_mean_train, y_train, epochs=best_hyperparameters['epochs'], validation_split=0.25,
                       batch_size=best_hyperparameters['batch_size'],
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                       ], verbose=1)
    elif model_type == 'linear_regression':
        best_model = create_linear_regression_model()
        best_model.fit(z_mean_train, y_train)
    elif model_type == 'elastic_net':
        best_model = create_elastic_net_model(**best_hyperparameters)
        best_model.fit(z_mean_train, y_train)
    elif model_type == 'random_forest':
        best_model = create_random_forest_model(**best_hyperparameters)
        best_model.fit(z_mean_train, y_train)
    elif model_type == 'xgboost':
        best_model = create_xgboost_model(**best_hyperparameters)
        best_model.fit(z_mean_train, y_train)

    # Evaluate regressor
    phenotype_predictions_test = best_model.predict(z_mean_test)
    mse_test = mean_squared_error(y_test, phenotype_predictions_test)
    r2_test = r2_score(y_test, phenotype_predictions_test)
    pearson_corr, _ = pearsonr(y_test.values.flatten(), phenotype_predictions_test.flatten())

    print(f"Independent Test MSE for {model_type} ({snp_data_loc}): {mse_test}")
    print(f"Independent Test R2 for {model_type} ({snp_data_loc}): {r2_test}")
    print(f"Independent Test Pearson Correlation for {model_type} ({snp_data_loc}): {pearson_corr}")

    # Perform cross-validation for regressor
    # Perform cross-validation for regressor
    avg_mse_train, avg_mse_val, avg_r2_train, avg_r2_val, avg_pearson_train, avg_pearson_val = cross_validate_regressor(
        z_mean_full, phenotype, best_model)

    # Save metrics for regressor
    save_regressor_metrics(snp_data_loc, avg_mse_train, avg_r2_train, avg_mse_val, avg_r2_val, mse_test, r2_test,
                           avg_pearson_train, avg_pearson_val, pearson_corr, hopt=f"{hopt}/{model_type}")

