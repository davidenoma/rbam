import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, hp, tpe, space_eval
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap  # SHAP for XAI
import matplotlib.pyplot as plt
import utils
from utils import save_summary, load_real_genotype_data_case_control, load_real_genotype_data

# Set up environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

snp_data_loc = sys.argv[1]
if sys.argv[3] == 'quantitative':
    X_train, X_test, snp_data, phenotype, y_train, y_test = load_real_genotype_data(snp_data_loc)
else:
    X_train, X_test, snp_data = load_real_genotype_data_case_control(snp_data_loc)

# Feature names
feature_names = X_train.columns


# Function to sample 100 individuals without replacement until the data is exhausted
def sample_until_exhausted(X, n_samples=100, num_samples=None):
    remaining_indices = np.arange(X.shape[0])
    samples = []
    if num_samples:
        while num_samples > 0:
            if len(remaining_indices) >= n_samples:
                sampled_indices = np.random.choice(remaining_indices, size=n_samples, replace=False)
                samples.append(X.iloc[sampled_indices])
                remaining_indices = np.setdiff1d(remaining_indices, sampled_indices)  # Remove used indices
            num_samples = num_samples - 1
    else:
        while len(remaining_indices) >= n_samples:
            sampled_indices = np.random.choice(remaining_indices, size=n_samples, replace=False)
            samples.append(X.iloc[sampled_indices])
            remaining_indices = np.setdiff1d(remaining_indices, sampled_indices)  # Remove used indices

    return samples


# Sample until the dataset is almost exhausted
samples = sample_until_exhausted(X_train, n_samples=100,num_samples=1)

# Convert sampled data to NumPy array for SHAP compatibility
# Initialize the cumulative sum array with the shape of the model's output
sampled_shap_values_sum = np.zeros(len(feature_names))  # To accumulate the SHAP values


# Save model function
def save_model(model: tf.keras.Model, snp_data_loc: str, override: bool = False):
    directory = os.path.join(os.getcwd(), 'model_AE')
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    print(f"Saving model to filepath: {filepath}")

    if os.path.exists(filepath) and not override:
        raise FileExistsError(f"The file {filename} already exists. Set override=True to overwrite.")

    model.save(filepath)


# Load model function
def load_model(snp_data_loc):
    directory = os.path.join(os.getcwd(), 'model_AE')
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    filepath = os.path.join(directory, filename)

    if os.path.exists(filepath):
        return tf.keras.models.load_model(filepath, custom_objects={"Autoencoder": Autoencoder})
    else:
        return None


# Define Autoencoder class
@tf.keras.utils.register_keras_serializable(package="Custom", name="Autoencoder")
class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        latent_space = self.encoder(inputs)
        reconstructed = self.decoder(latent_space)
        return reconstructed

    def get_latent_space(self, inputs):
        return self.encoder(inputs)

    def get_config(self):
        config = super(Autoencoder, self).get_config()
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


# Function to create Autoencoder
def create_autoencoder(input_dim, num_hidden_layers_encoder, num_hidden_layers_decoder, encoding_dimensions,
                       decoding_dimensions, activation, batch_size, epochs, learning_rate, latent_dim):
    encoder_layers = [input_dim] + [encoding_dimensions] * num_hidden_layers_encoder + [latent_dim]
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation) for layer in encoder_layers[1:]]
    ])

    decoder_layers = [latent_dim] + [decoding_dimensions] * num_hidden_layers_decoder + [input_dim]
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation) for layer in decoder_layers[1:]]
    ])

    autoencoder = Autoencoder(encoder=encoder, decoder=decoder)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return autoencoder


def objective(params):
    model = create_autoencoder(input_dim=X_train.shape[1], **params)
    history = model.fit(X_train, X_train, epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_split=0.25,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                        verbose=1)

    return {'loss': history.history['loss'][-1], 'status': 'ok'}


# Load or create Autoencoder model
best_model = load_model(snp_data_loc)

if not best_model:
    # Perform hyperparameter optimization
    space = {
        'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', range(1, 17)),
        'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', range(1, 17)),
        'encoding_dimensions': hp.choice('encoding_dimensions', [128, 256, 512]),
        'decoding_dimensions': hp.choice('decoding_dimensions', [128, 256, 512]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'learning_rate': hp.choice('learning_rate', [0.000001, 0.00001, 0.0001, 0.001]),
        'epochs': hp.choice('epochs', [50, 100, 150,200]),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'latent_dim': hp.choice('latent_dim', [4, 8, 16, 32, 64, 128])
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)
    best_hyperparameters = space_eval(space, best)
    print("Best hyperparameters for Autoencoder:", best_hyperparameters)
    best_model = create_autoencoder(input_dim=X_train.shape[1], **best_hyperparameters)

    best_history = best_model.fit(X_train, X_train, epochs=best_hyperparameters['epochs'],
                                  batch_size=best_hyperparameters['batch_size'], validation_split=0.25)

    save_model(best_model, snp_data_loc)

# Directory to save SHAP values
shap_save_dir = f"model_outputs/hopt_AE/shap_values_{os.path.basename(snp_data_loc)}"
os.makedirs(shap_save_dir, exist_ok=True)

#reconstruction of the data splits
# Reconstruction of the data splits
reconstructed_data_test = best_model.predict(X_test)
reconstructed_data_train = best_model.predict(X_train)
reconstructed_full_data = best_model.predict(snp_data)

# MSE of reconstruction
mse_train = np.mean(np.square(X_train - reconstructed_data_train))
mse_test = np.mean(np.square(X_test - reconstructed_data_test))
mse_whole = np.mean(np.square(snp_data - reconstructed_full_data))

# Save MSE values
utils.save_mse_values(snp_data_loc, mse_train, mse_test, mse_whole, hopt="hopt_AE")

# RMSE of reconstruction
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
rmse_whole = np.sqrt(mse_whole)

# R² of reconstruction
r2_train = np.mean(utils.evaluate_r2(X_train, reconstructed_data_train))
r2_test = np.mean(utils.evaluate_r2(X_test, reconstructed_data_test))
r2_whole = np.mean(utils.evaluate_r2(snp_data, reconstructed_full_data))

# Save R² scores
utils.save_r2_scores(snp_data_loc, r2_train, r2_test, r2_whole, hopt="hopt_AE")

# Adjusted R² of reconstruction
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


# print(samples)
# Loop through samples, calculate SHAP values, and accumulate
for i, X_sample in enumerate(samples):
    X_sample_np = X_sample.to_numpy()

    # SHAP for XAI using the sampled individuals
    explainer = shap.DeepExplainer(best_model.encoder, X_sample_np)
    shap_values = explainer.shap_values(X_sample_np,check_additivity=False)

    # Average across the latent space dimension (axis=-1) to get SHAP values per feature
    shap_values_mean = np.mean(np.abs(shap_values[0]), axis=-1)

    # Check if the SHAP values shape matches the expected shape
    if shap_values_mean.shape != sampled_shap_values_sum.shape:
        print(f"Skipping sample {i} due to shape mismatch: {shap_values_mean.shape} vs {sampled_shap_values_sum.shape}")
        continue

    # Accumulate SHAP values
    sampled_shap_values_sum += shap_values_mean

    # Save SHAP values for the current sample

    # Save SHAP values for the current sample
    cleaned_feature_names = feature_names
    shap_df = pd.DataFrame({'SNP_ID': cleaned_feature_names, 'Weight': shap_values_mean})
    shap_df.to_csv(os.path.join(shap_save_dir, f'shap_values_sample_{i}.csv'), index=False)

# Save cumulative SHAP values as CSV
shap_df_total = pd.DataFrame({'SNP_ID': feature_names, 'Weight': sampled_shap_values_sum})
# shap_df_total.to_csv(os.path.join(shap_save_dir, f'shap_values_total_sum_{os.path.basename(snp_data_loc)}.csv'), index=False)
bim_file_path = sys.argv[2]
bim = pd.read_csv(bim_file_path, sep="\t", header=None)
bim.columns = ['Chromosome', 'SNP_ID', 'Start', 'Position', 'Ref', 'Alt']
print(shap_df_total,shap_df_total.shape)
print(bim,bim.shape)
shap_df_total['SNP_ID'] = shap_df_total['SNP_ID'].str[:-2]
# Merge SHAP results with BIM file
merged_data = pd.merge(shap_df_total, bim, how="inner", on="SNP_ID")
merged_data = merged_data[['SNP_ID', 'Chromosome', 'Position', 'Weight']]
print(merged_data,merged_data.shape)
# Save merged datac
output_file = os.path.join(shap_save_dir, f'{os.path.splitext(os.path.basename(snp_data_loc))[0]}_merged_snp_and_weights.csv')
merged_data.to_csv(output_file, index=False)

# Create SHAP Explanation object for cumulative sum
explained_data = shap.Explanation(values=sampled_shap_values_sum, feature_names=feature_names)

# Save SHAP bar plot for cumulative sum
shap.plots.bar(explained_data)
plt.savefig(f"{shap_save_dir}/shap_bar_plot_total.png")

sys.exit()

