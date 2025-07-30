import os
import sys
from statistics import mean

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.utils import class_weight,resample
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
from utils import load_real_genotype_data, cross_validate_classifier, save_classifier_metrics
def save_model(model: tf.keras.Model, snp_data_loc: str,  override: bool = True):
    """
    Save a TensorFlow model to a specified location.

    Args:
        model (tf.keras.Model): The model to save.
        snp_data_loc (str): The path of the SNP data location.
        model_name (str): The name of the model to append to the filename.
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
        return keras.models.load_model(filepath, custom_objects={"VAE": VAE, "vae_loss": vae_loss})
    else:
        return None


# Load data
snp_data_loc = sys.argv[1]

X_train, X_test, snp_data, phenotype, y_train, y_test = load_real_genotype_data(snp_data_loc)
scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# Convert labels: 1 to 0 and 2 to 1
y_train = np.where(y_train == 1, 0, 1)
y_test = np.where(y_test == 1, 0, 1)
phenotype = np.where(phenotype == 1, 0, 1)


# Extract SNP file name from path
snp_file_name = os.path.basename(snp_data_loc)
hopt = "rbam_classifier"

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

# Load or train the best VAE model
best_vae_model = load_model(snp_data_loc)

if not best_vae_model:  # Hyperparameter optimization for VAE
    best_vae = fmin(fn=objective, space=vae_space, algo=tpe.suggest, max_evals=10)

    # Extract the best hyperparameters for VAE
    best_vae_hyperparameters = space_eval(vae_space, best_vae)
    print(f"Best hyperparameters for VAE ({snp_file_name}):", best_vae_hyperparameters)

    # Create and compile the best VAE model using the best hyperparameters
    best_vae_model = create_vae_model(input_dim=X_train.shape[1], **best_vae_hyperparameters)

    # Train the best VAE model on the entire dataset
    best_vae_history = best_vae_model.fit(X_train, X_train, epochs=best_vae_hyperparameters['epochs'],
                                          batch_size=best_vae_hyperparameters['batch_size'],
                                          validation_split=0.25)

    utils.save_model(best_vae_model, snp_data_loc)
# Calculate and save MSE and R2 scores for the VAE
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


# Calculate R²
r2_train = np.mean(utils.evaluate_r2(X_train, reconstructed_data_train))
r2_test = np.mean(utils.evaluate_r2(X_test, reconstructed_data_test))
r2_whole = np.mean(utils.evaluate_r2(snp_data, reconstructed_full_data))
utils.save_r2_scores(snp_data_loc, r2_train, r2_test, r2_whole, hopt=hopt)


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


# Extract latent vectors
encoder = best_vae_model.encoder
z_mean_full, _ = tf.split(encoder.predict(snp_data), num_or_size_splits=2, axis=1)
print(_,z_mean_full)
z_mean_train, _ = tf.split(encoder.predict(X_train), num_or_size_splits=2, axis=1)
z_mean_test, _ = tf.split(encoder.predict(X_test), num_or_size_splits=2, axis=1)

# Scale latent space
scaler = StandardScaler().fit(z_mean_train)
z_mean_train = scaler.transform(z_mean_train)
z_mean_test = scaler.transform(z_mean_test)
z_mean_full = scaler.transform(z_mean_full)  # Will be used later in CV

# Class weights calculation
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define additional classifiers
def create_logistic_regression_model(C, penalty, class_weight=None):
    return LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000,class_weight=class_weights_dict)

def create_random_forest_model(n_estimators, max_depth, class_weight=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42,class_weight=class_weights_dict)

def create_xgboost_model(learning_rate, n_estimators, max_depth, class_weight=None):
    return XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                         use_label_encoder=False, eval_metric='logloss',class_weight=class_weights_dict)

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
        'max_depth': hp.choice('max_depth', [5, 10, 15, 20,25,50])
    },
    'xgboost': {
        'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1, 0.2]),
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [3, 5, 7, 9])
    }
}




# Objective function for hyperparameter optimization
def objective_classifier(params, model_type):
    """
    Hyperopt objective:  model is trained on 80 % of the *training*
    data and evaluated on the held-out 20 % validation split.
    The outer test set (z_mean_test / y_test) is never touched here.
    """
    if model_type == 'tf_classifier':
        # Keras can rely on its own validation_split argument
        model = create_tf_classifier_model(input_dim=z_mean_train.shape[1],
                                           **params)
        hist = model.fit(
            z_mean_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.20,            # inner-loop validation
            class_weight=class_weights_dict,
            verbose=0
        )
        val_loss = np.min(hist.history['val_loss'])
    else:
        # --- make an inner train/val split so we never peek at x_test ---
        X_tr, X_val, y_tr, y_val = train_test_split(
            z_mean_train, y_train,
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
        val_loss = 1.0 - accuracy_score(y_val, preds)   # minimise 1-accuracy

    return {'loss': val_loss, 'status': STATUS_OK}

# Usage
for model_type, space in classifier_space.items():
    best_classifier = fmin(fn=lambda params: objective_classifier(params, model_type), space=space, algo=tpe.suggest, max_evals=20)
    best_hyperparameters = space_eval(space, best_classifier)
    print(f"Best hyperparameters for {model_type} ({snp_file_name}): {best_hyperparameters}")

    # Train and evaluate the model with the best hyperparameters
    if model_type == 'tf_classifier':
        best_model = create_tf_classifier_model(input_dim=z_mean_train.shape[1], **best_hyperparameters)
        best_model.fit(
            z_mean_train, y_train, epochs=best_hyperparameters['epochs'], validation_split=0.25,
            batch_size=best_hyperparameters['batch_size'], class_weight=class_weights_dict,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], verbose=1
        )
    elif model_type == 'logistic_regression':
        best_model = create_logistic_regression_model(**best_hyperparameters)
        best_model.fit(z_mean_train, y_train )
    elif model_type == 'random_forest':
        best_model = create_random_forest_model(**best_hyperparameters)
        best_model.fit(z_mean_train, y_train)
    elif model_type == 'xgboost':
        best_model = create_xgboost_model(**best_hyperparameters)
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        best_model.set_params(scale_pos_weight=scale_pos_weight)
        best_model.fit(z_mean_train, y_train)


    # Evaluate classifier

    phenotype_predictions_test = best_model.predict(z_mean_test)
      # --- probabilities for AUC, thresholded classes for accuracy ---

    if isinstance(best_model, tf.keras.Model):
                proba_test = best_model.predict(z_mean_test, verbose=0).ravel()
    else:
            proba_test = best_model.predict_proba(z_mean_test)[:, 1]

    y_pred_test = (proba_test > 0.5).astype(int)


    # Cross-validation for each classifier
    avg_accuracy_train, avg_accuracy_val, avg_auc_train, avg_auc_val = cross_validate_classifier(
        z_mean_train, y_train, best_model)

    # Save classifier metrics, including R²
    save_classifier_metrics(snp_data_loc, avg_accuracy_train, avg_auc_train, avg_accuracy_val, avg_auc_val
                            ,hopt=f"{hopt}/{model_type}")

    print(
        f"Cross-Validation Accuracy for {model_type} ({snp_file_name}) - Train: {avg_accuracy_train}, Test: {avg_accuracy_val}")
    print(f"Cross-Validation AUC for {model_type} ({snp_file_name}) - Train: {avg_auc_train}, Test: {avg_auc_val}")

