import os
import subprocess

import keras
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from scipy.stats import f, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score, pairwise_distances
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

import utils


# from latent_space_classifier import VAE
# from sklearn.preprocessing import StandardScaler
# import keras
def extract_phenotype(genotype_file):
    """
    Extracts the phenotype column from the .raw genotype file, skipping the header.

    Parameters:
        genotype_file (str): The path to the genotype file (without the .raw extension).

    Returns:
        pd.Series: A pandas Series containing the phenotype values.
    """
    # Construct the awk command to extract the phenotype column, skipping the header
    awk_command = f"awk -F' ' 'NR > 1 {{print $6}}' {genotype_file}"
    print(genotype_file)

    # Run the awk command and capture the output
    result = subprocess.run(awk_command, shell=True, capture_output=True, text=True)

    # Convert the result to a list of phenotype values
    phenotype = result.stdout.strip().split('\n')
    print(phenotype)

    # Convert to a Pandas Series
    phenotype = pd.Series(phenotype, dtype='float')

    return phenotype


def load_real_genotype_data(snp_data_loc):
    """
    Loads genotype data from a file, processes SNP IDs to remove suffixes,
    removes invalid SNPs, extracts the phenotype, and splits the data into
    training and testing sets.

    Parameters:
        snp_data_loc (str): The path to the SNP data file.

    Returns:
        X_train, X_test, snp_data, phenotype, y_train, y_test (tuple): The train-test split data and the full dataset.
    """
    # Extract the phenotype column using the extract_phenotype function
    print(snp_data_loc)
    # sys.exit()
    phenotype = extract_phenotype(snp_data_loc)

    chunk_size = 1000
    # Columns to skip during reading
    columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    # Initialize an empty DataFrame
    snp_data = pd.DataFrame()


    # Read data in chunks, skipping unwanted columns
    reader = pd.read_csv(f"{snp_data_loc}", sep=" ", usecols=lambda column: column not in columns_to_skip,
                         dtype='int8', chunksize=chunk_size, verbose=True)

    i = 0
    for chunk in reader:
        snp_data = pd.concat([snp_data, chunk])

        i += 1

    # Process SNP IDs to remove suffixes (e.g., "_A", "_T")
    # snp_ids = snp_data.columns.str.replace(r'_[ATGC]$', '', regex=True)
    #
    # # Replace old column names with cleaned SNP IDs
    # snp_data.columns = snp_ids

    # Since the phenotype is already extracted, we don't need to extract it again.
    # Remove the PHENOTYPE column if it is still in the DataFrame
    if 'PHENOTYPE' in snp_data.columns:
        snp_data = snp_data.drop(columns=['PHENOTYPE'])

    print(snp_data.shape)
    print("Done loading the recoded genotype")

    # Preprocess the SNP data (train-test split)
    X_train, X_test, y_train, y_test = train_test_split(snp_data, phenotype, test_size=0.2, random_state=42)

    return X_train, X_test, snp_data, phenotype, y_train, y_test


def load_real_genotype_data_case_control(snp_data_loc):
    chunk_size = 1000
    # Columns to skip during reading
    columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX']
    # Initialize an empty DataFrame
    snp_data = pd.DataFrame()
    # Read data in chunks
    reader = pd.read_csv(snp_data_loc, sep=" ", usecols=lambda column: column not in columns_to_skip,
                         dtype='int8', chunksize=chunk_size, verbose=True)
    i = 0
    # Identify columns containing at least one -9 and remove those SNPS
    for chunk in reader:
        snp_data = pd.concat([snp_data, chunk])

    # Filter rows based on 'PHENOTYPE'

    if 'PHENOTYPE' in snp_data.columns:
        if 2 in snp_data['PHENOTYPE'].unique():
            snp_data = snp_data[snp_data['PHENOTYPE'] == 2]
        snp_data = snp_data.drop(columns=['PHENOTYPE'])
    else:
        # Handle the case where 'PHENOTYPE' column is not present
        pass
    # Removing -9
    # columns_to_remove = snp_data.columns[snp_data.eq(-9).any()]
    # snp_data = snp_data.drop(columns=columns_to_remove)
    #
    # # Reset index
    snp_data.reset_index(inplace=True, drop=True)
    # snp_data = snp_data.dropna(axis=1)
    print("Snp data shape:", snp_data.shape)
    print("Done loading the recoded genotype")
    # Preprocess the SNP data (standardization and train-test split)
    X_train, X_test, = train_test_split(snp_data, test_size=0.2, random_state=42)

    return X_train, X_test, snp_data


def sim_data():
    # Generate synthetic data
    num_samples = 5000
    num_snps = 100
    # Define feature names based on your data (replace these with your actual feature names)
    feature_names = [f'feature_{i}' for i in range(num_snps)]
    synthetic_data = np.random.randint(0, 3, size=(num_samples, num_snps), dtype='int8')
    synthetic_df = pd.DataFrame(data=synthetic_data, columns=feature_names)
    X_train, X_test = train_test_split(synthetic_df, test_size=0.2, random_state=42)
    return X_train, X_test


def save_summary(snp_data_loc, vae, hopt=None):
    output_folder = "model_outputs"
    if hopt:
        output_folder = output_folder + "/" + hopt
    os.makedirs(output_folder, exist_ok=True)
    # Writing model to file
    # Write Encoder to summary
    with open(os.path.join(output_folder,
                           f'{os.path.splitext(os.path.basename(snp_data_loc))[0]}_encoder_model_summary.txt'),
              'w') as f:
        sys.stdout = f  # Redirect standard output to the file
        vae.encoder.summary()  # Print the model summary
        sys.stdout = sys.__stdout__  # Reset standard output

    # Write Decoder to summary
    with open(os.path.join(output_folder,
                           f'{os.path.splitext(os.path.basename(snp_data_loc))[0]}_decoder_model_summary.txt'),
              'w') as f:
        sys.stdout = f  # Redirect standard output to the file
        vae.decoder.summary()  # Print the model summary
        sys.stdout = sys.__stdout__  # Reset standard output


def extract_decoder_reconstruction_weights(vae):
    # Instead i extract the decoder weights because I
    # want the representation learning be close to the reconstructions
    decoder_weights = vae.decoder.get_weights()
    # print('Decoder weights',decoder_weights)
    decoder_weight_paths = vae.decoder.get_config

    decoder_layers = vae.decoder.layers
    # Print the layer configurations at runtime
    for layer in decoder_layers:
        print(f"Layer: {layer.name}")
        print(f"Type: {layer.__class__.__name__}")
        print("Config:")

    # print(decoder_weight_paths)
    kernel_weights_for_2nd_to_last_neuron = decoder_weights[-2]
    feature_importance = []
    print([w.shape for w in decoder_weights])  # Print shapes of all layers' weights
    print(len(decoder_weights))  # Print shapes of all layers' weights
    kernel_weights_for_2nd_to_last_layer = decoder_weights[-2]
    print(kernel_weights_for_2nd_to_last_layer.shape)  # Should be 2D

    for i in range(kernel_weights_for_2nd_to_last_neuron.shape[1]):
        # Get weights for the i-th latent dimension
        weights = kernel_weights_for_2nd_to_last_neuron[:, i]  # Weights for the i-th neuron in that layer
        # Calculate the absolute sum of weights for each feature
        sum_weights_for_layer = np.sum(weights)
        feature_importance.append(sum_weights_for_layer)
    return feature_importance


def extract_encoder_weights(vae):
    # Instead i extract the decoder weights because I
    # want the representation learning be close to the reconstructions
    encoder_weights = vae.encoder.get_weights()

    encoder_layers = vae.encoder.layers

    # Print the layer configurations at runtime
    # for layer in encoder_layers:
    #     print(f"Layer: {layer.name}")
    #     print(f"Type: {layer.__class__.__name__}")
    #     print("Config:")
    #     for key, value in layer.get_config().items():
    #         print(f"  {key}: {value}")
    #     print("-" * 60)

    # print(encoder_weights.shape)
    # encoder_weight_paths = vae.encoder.get_weight_paths()
    # print(encoder_weight_paths)
    # print([w.shape for w in encoder_weights])
    kernel_weights_for_first_neuron = encoder_weights[0]
    feature_importance = []
    print('Encoder Size', kernel_weights_for_first_neuron.shape)
    for i in range(kernel_weights_for_first_neuron.shape[0]):
        # Get weights for the i-th latent dimension
        weights = kernel_weights_for_first_neuron[i]  # Weights for the i-th neuron in that layer
        # Calculate the absolute sum of weights for each feature
        sum_weights_for_layer = np.sum(weights)
        feature_importance.append(sum_weights_for_layer)
    return feature_importance


def save_plots(history, snp_data_loc, hopt=None):
    output_folder = "model_outputs"
    if hopt:
        output_folder = output_folder + "/" + hopt

    os.makedirs(output_folder, exist_ok=True)
    plt.plot(pd.DataFrame(history.history))
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Loss over Epochs")
    # Add legend if necessary
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.savefig(
        os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_loss_and_epoch.png"))
    plt.show()


def save_mse_values(snp_data_loc, mse_train, mse_test, mse_whole, hopt=None):
    output_folder = "model_outputs"
    if hopt:
        output_folder = output_folder + "/" + hopt
    os.makedirs(output_folder, exist_ok=True)

    # Print the MSE values
    print("Mean Squared Error (MSE) between original and reconstructed train data:", mse_train)
    print("Mean Squared Error (MSE) between input and reconstructed test data:", mse_test)
    print("Mean Squared Error (MSE) between input and reconstructed whole data:", mse_whole)

    # Write the MSE values to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_mse_results.txt"),
              "w") as file:
        file.write(
            "Mean Squared Error (MSE) between original and reconstructed original train data: " + str(mse_train) + "\n")
        file.write("Mean Squared Error (MSE) between reconstructed test data: " + str(mse_test) + "\n")
        file.write("Mean Squared Error (MSE) between reconstructed whole data: " + str(mse_whole) + "\n")


def save_mse_values_cv(snp_data_loc, mse_train, mse_test, hopt=None):
    output_folder = "model_outputs"
    if hopt:
        output_folder = output_folder + "/" + hopt
    os.makedirs(output_folder, exist_ok=True)

    # Print the MSE values
    print("Mean Squared Error (MSE) between original and reconstructed train data:", mse_train)
    print("Mean Squared Error (MSE) between input and reconstructed test data:", mse_test)

    # Write the MSE values to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_mse_results.txt"),
              "w") as file:
        file.write(
            "Mean Squared Error (MSE) between original and reconstructed original train data: " + str(mse_train) + "\n")
        file.write("Mean Squared Error (MSE) between reconstructed test data: " + str(mse_test) + "\n")


def save_r2_scores(snp_data_loc, r2_train, r2test, r2whole, hopt=None):
    output_folder = "model_outputs"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    os.makedirs(output_folder, exist_ok=True)
    # Print the R2 scores
    print("R-squared (R2) for train data:", r2_train)
    print("R-squared (R2) for  test data:", r2test)
    print("R-squared (R2) for  whole data:", r2whole)

    # Write the R2 scores to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_r2_results.txt"),
              "w") as file:
        file.write("R-squared (R2) for  train data: " + str(r2_train) + "\n")
        file.write("R-squared (R2) for  test data: " + str(r2test) + "\n")
        file.write("R-squared (R2) for  whole data: " + str(r2whole) + "\n")


def save_r2_scores_cv(snp_data_loc, r2_train, r2test, hopt=None):
    output_folder = "model_outputs/cv"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    os.makedirs(output_folder, exist_ok=True)
    # Print the R2 scores
    print("R-squared (R2) for train data:", r2_train)
    print("R-squared (R2) for  test data:", r2test)

    # Write the R2 scores to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_r2_results.txt"),
              "w") as file:
        file.write("R-squared (R2) for  train data: " + str(r2_train) + "\n")
        file.write("R-squared (R2) for  test data: " + str(r2test) + "\n")


def save_classifier_metrics(snp_data_loc, train_accuracy, train_auc, test_accuracy, test_auc, ind_test_accuracy,
                            ind_test_auc, hopt=None):
    """
    Save the classifier metrics including accuracy, AUC, and R² to a file.

    Parameters:
    snp_data_loc (str): Location of the SNP data.
    train_accuracy (float): Training accuracy score.
    train_auc (float): Training AUC score.
    test_accuracy (float): Validation accuracy score.
    test_auc (float): Validation AUC score.
    ind_test_accuracy (float): Independent test accuracy score.
    ind_test_auc (float): Independent test AUC score.
    ind_test_r2 (float): Independent test R² score.
    train_r2 (float): Cross-validation train R² score.
    test_r2 (float): Cross-validation test R² score.
    hopt (str): Optional hyperparameter optimization identifier.
    """
    output_folder = "model_outputs"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    os.makedirs(output_folder, exist_ok=True)

    # Write the metrics to a file
    with open(os.path.join(output_folder,
                           f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_classifier_metrics.txt"),
              "w") as file:
        file.write(f"Train Accuracy: {train_accuracy}\n")
        file.write(f"Train AUC: {train_auc}\n")
        # file.write(f"Train R²: {train_r2}\n")
        file.write(f"Test Accuracy: {test_accuracy}\n")
        file.write(f"Test AUC: {test_auc}\n")
        # file.write(f"Test R²: {test_r2}\n")
        file.write(f"Independent Test Accuracy: {ind_test_accuracy}\n")
        file.write(f"Independent Test AUC: {ind_test_auc}\n")
        # file.write(f"Independent Test R²: {ind_test_r2}\n")


def obtain_snps_and_weights(X_train, feature_importance, bim, snp_data_loc, feature_type, hopt=None):
    original_feature_names = X_train.columns
    rep_weights_and_snps = dict(zip(original_feature_names, feature_importance))
    # print(rep_weights_and_snps)
    df_rep_weights_and_snps = pd.DataFrame(rep_weights_and_snps.items(), columns=['SNP_ID', 'Weight'])
    df_rep_weights_and_snps['SNP_ID'] = df_rep_weights_and_snps['SNP_ID'].str[:-2]
    # Read the bim file

    bim.columns = ['Chromosome', 'SNP_ID', 'Start', 'Position', 'Ref', 'Alt']
    # Merge the two dataframes based on SNP_ID
    merged_data = pd.merge(df_rep_weights_and_snps, bim, how="inner", on='SNP_ID')
    # Select the required columns and rename them
    merged_data = merged_data[['SNP_ID', 'Chromosome', 'Position', 'Weight']]
    # Save the merged data to snp_and_weights file
    output_folder = "output_weights"
    if hopt:
        output_folder = "output_weights" + "/" + hopt

    os.makedirs(output_folder, exist_ok=True)
    # Update output file name with feature importance variable name
    output_file = os.path.join(output_folder,
                               f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_{feature_type}_snp_and_weights.tsv")
    merged_data.to_csv(output_file, index=False)


def evaluate_r2(original, reconstructed):
    ss_res = np.sum((original - reconstructed) ** 2, axis=0)
    ss_total = np.sum((original - np.mean(original)) ** 2, axis=0)
    rsquared = 1 - (ss_res / ss_total)

    return rsquared


def evaluate_r2_ls_reg(y_true, y_pred):
    """
    Compute the coefficient of determination (R²) for the given true and predicted values.

    Args:
        y_true (np.ndarray): The true values (targets).
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The R² score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_total)

    return r_squared


def save_model(model, snp_data_loc, override=False):
    """
    Save a TensorFlow model.

    Args:
    - model: The TensorFlow model to be saved.
    - snp_data_loc: The path to the SNP data file used for naming the model.
    - override: If True, override the existing model file with the new one. If False, raise an error if the file exists.
    """
    directory = f"{os.getcwd()}/model"
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    print("Saving model to filepath", filepath)
    if os.path.exists(filepath) and not override:
        raise FileExistsError(f"The file {filename} already exists. Set override=True to overwrite.")
    model.save(filepath)
    # keras.models.save_model(model, filepath)


# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import get_custom_objects
#
# # Register custom objects
# get_custom_objects().update({'vae_loss': vae_loss})
def load_model(snp_data_loc):
    """
    Load a TensorFlow model if it exists.

    Args:
    - snp_data_loc: The path to the SNP data file used for naming the model.

    Returns:
    - model: The loaded TensorFlow model if it exists, otherwise returns None.
    """

    # def vae_loss(x, x_reconstructed):
    #     z_mean, z_log_var = tf.split(hyp_optimize_cc_VAE.VAE.(x), num_or_size_splits=2, axis=1)
    #     reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_reconstructed))
    #     kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    #     return reconstruction_loss + kl_loss
    #
    # Register the VAE loss function
    #
    # keras.utils.get_custom_objects().update({'vae_loss': loss})

    directory = f"{os.getcwd()}/model"
    filename = f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}.keras"
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        return keras.models.load_model(filepath, safe_mode=False)
        # return tf.saved_model.load(filepath)
    else:
        return None
def cross_validate_vae(snp_data, best_model, n_splits=5, random_state=11):
    # Initialize lists to store metrics for all folds
    mse_train_list = []
    mse_val_list = []
    r2_train_list = []
    r2_val_list = []
    adj_r2_train_list = []
    adj_r2_val_list = []
    pearson_corr_train_list = []
    pearson_corr_val_list = []

    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Iterate over cross-validation folds
    for train_index, val_index in kf.split(snp_data):
        # Split data into training and validation sets
        X_train, X_val = snp_data.iloc[train_index], snp_data.iloc[val_index]

        # Reconstruct the input data using the trained VAE model
        reconstructed_data_train = best_model.predict(X_train)
        reconstructed_data_val = best_model.predict(X_val)

        # Calculate MSE for both training and validation sets
        mse_train = mean_squared_error(X_train, reconstructed_data_train)
        mse_val = mean_squared_error(X_val, reconstructed_data_val)

        # Calculate standard R-squared
        r2_train = evaluate_r2(X_train, reconstructed_data_train)
        r2_val = evaluate_r2(X_val, reconstructed_data_val)

        # Calculate Adjusted R-squared
        n_train, p_train = X_train.shape
        n_val, p_val = X_val.shape
        adj_r2_train = adjusted_r2_score(X_train, reconstructed_data_train, n_train, p_train)
        adj_r2_val = adjusted_r2_score(X_val, reconstructed_data_val, n_val, p_val)

        # Calculate Pearson Correlation
        pearson_corr_train = compute_pearson_correlation(X_train, reconstructed_data_train)
        pearson_corr_val = compute_pearson_correlation(X_val, reconstructed_data_val)

        # Append metrics to lists
        mse_train_list.append(mse_train)
        mse_val_list.append(mse_val)
        r2_train_list.append(r2_train)
        r2_val_list.append(r2_val)
        adj_r2_train_list.append(adj_r2_train)
        adj_r2_val_list.append(adj_r2_val)
        pearson_corr_train_list.append(pearson_corr_train)
        pearson_corr_val_list.append(pearson_corr_val)

    # Calculate average metrics over all folds
    avg_mse_train = np.mean(mse_train_list)
    avg_mse_val = np.mean(mse_val_list)
    avg_r2_train = np.mean(r2_train_list)
    avg_r2_val = np.mean(r2_val_list)
    avg_adj_r2_train = np.mean(adj_r2_train_list)
    avg_adj_r2_val = np.mean(adj_r2_val_list)
    avg_pearson_corr_train = np.mean(pearson_corr_train_list)
    avg_pearson_corr_val = np.mean(pearson_corr_val_list)

    # Return all metrics
    return (
        avg_mse_train,
        avg_mse_val,
        avg_r2_train,
        avg_r2_val,
        # avg_adj_r2_train,
        # avg_adj_r2_val,
        avg_pearson_corr_train,
        avg_pearson_corr_val,
    )


# def cross_validate_vae(snp_data, best_model, n_splits=5, test_size=0.2, random_state=11):
#     # Initialize lists to store MSE and R-squared values for all folds
#     mse_train_list = []
#     mse_val_list = []
#     r2_train_list = []
#     r2_val_list = []
#     r2_train_list_py = []
#     r2_val_list_py = []
#
#     # Perform K-fold cross-validation
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#
#     # Iterate over cross-validation folds
#     for train_index, val_index in kf.split(snp_data):
#         # Split data into training and validation sets using indices from KFold
#         X_train, X_val = snp_data.iloc[train_index], snp_data.iloc[val_index]
#
#         # Train your VAE model using X_train
#         # (Make sure best_model is trained before this step)
#
#         # Reconstruct the input data using the trained VAE model
#         reconstructed_data_train = best_model.predict(X_train)
#         reconstructed_data_val = best_model.predict(X_val)
#
#         # Calculate MSE for both training and validation sets
#         mse_train_cv = np.mean(np.square(X_train - reconstructed_data_train))
#         mse_val = np.mean(np.square(X_val - reconstructed_data_val))
#
#         # Calculate R-squared for the original data
#         r2_train_py = r2_score(X_train, reconstructed_data_train)
#         # Calculate R-squared for the reconstructed data
#         r2_val_py = r2_score(X_val, reconstructed_data_val)
#
#         r2_train_cv = evaluate_r2(X_train, reconstructed_data_train)
#         # Calculate R-squared for the reconstructed data
#         r2_val = evaluate_r2(X_val, reconstructed_data_val)
#
#         # Append MSE and R-squared values to lists
#         mse_train_list.append(mse_train_cv)
#         mse_val_list.append(mse_val)
#
#         r2_train_list.append(r2_train_cv)
#         r2_val_list.append(r2_val)
#
#         r2_train_list_py.append(r2_train_py)
#         r2_val_list_py.append(r2_val_py)
#
#     # Calculate average MSE and R-squared values over all folds
#     avg_mse_train = np.mean(mse_train_list)
#     avg_mse_val = np.mean(mse_val_list)
#     avg_r2_train = np.mean(r2_train_list)
#     avg_r2_val = np.mean(r2_val_list)
#
#     avg_r2_train_py = np.mean(r2_train_list_py)
#     avg_r2_val_py = np.mean(r2_val_list_py)
#
#     return avg_mse_train, avg_mse_val, avg_r2_train, avg_r2_val, avg_r2_train_py, avg_r2_val_py


def cross_validate_classifier(X, y, model, n_splits=5, random_state=11):
    """
    Cross-validate the classifier and calculate average accuracy, AUC, and R² scores.

    Parameters:
    X (numpy array or tf.Tensor): Feature matrix.
    y (numpy array or tf.Tensor): Labels.
    model (tf.keras.Model or Scikit-Learn Model): The classifier model.
    n_splits (int): Number of cross-validation splits.
    random_state (int): Random state for reproducibility.

    Returns:
    tuple: Average accuracy, AUC, and R² scores for training and validation sets.
    """
    # Convert TensorFlow tensors to NumPy arrays if necessary
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    if isinstance(y, tf.Tensor):
        y = y.numpy()

    # Initialize lists to store accuracy, AUC, and R² scores for all folds
    accuracy_train_list = []
    accuracy_val_list = []
    auc_train_list = []
    auc_val_list = []
    r2_train_list = []
    r2_val_list = []

    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Iterate over cross-validation folds
    for train_index, val_index in kf.split(X):
        # Split data into training and validation sets using indices from KFold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Check if the model is a Keras/TensorFlow model or a Scikit-Learn model
        if isinstance(model, tf.keras.Model):
            # Train the TensorFlow/Keras model
            # model.fit(X_train, y_train)  # Adjust these parameters as needed

            # Predict probabilities for both training and validation sets
            y_train_pred_proba = model.predict(X_train)
            y_val_pred_proba = model.predict(X_val)

        else:
            # Train the Scikit-Learn model
            # model.fit(X_train, y_train)

            # Predict probabilities for both training and validation sets
            if hasattr(model, "predict_proba"):
                y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                y_val_pred_proba = model.predict_proba(X_val)[:, 1]
            else:
                # For models that do not support predict_proba, use predict
                y_train_pred_proba = model.predict(X_train)
                y_val_pred_proba = model.predict(X_val)

        # Convert predicted probabilities to binary predictions
        y_train_pred = (y_train_pred_proba > 0.5).astype(int).flatten()
        y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()

        # Calculate accuracy and AUC scores
        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_val = accuracy_score(y_val, y_val_pred)

        auc_train = roc_auc_score(y_train, y_train_pred_proba)
        auc_val = roc_auc_score(y_val, y_val_pred_proba)

        # Calculate R² scores
        r2_train = evaluate_r2(y_train, y_train_pred)
        r2_val = evaluate_r2(y_val, y_val_pred)

        # Append accuracy, AUC, and R² scores to lists
        accuracy_train_list.append(accuracy_train)
        accuracy_val_list.append(accuracy_val)
        auc_train_list.append(auc_train)
        auc_val_list.append(auc_val)
        # r2_train_list.append(r2_train)
        # r2_val_list.append(r2_val)

    # Calculate average accuracy, AUC, and R² scores over all folds
    avg_accuracy_train = np.mean(accuracy_train_list)
    avg_accuracy_val = np.mean(accuracy_val_list)

    avg_auc_train = np.mean(auc_train_list)
    avg_auc_val = np.mean(auc_val_list)

    # avg_r2_train = np.mean(r2_train_list)
    # avg_r2_val = np.mean(r2_val_list)

    return avg_accuracy_train, avg_accuracy_val, avg_auc_train, avg_auc_val


def cross_validate_regressor(X, y, model, n_splits=5, random_state=11, best_epochs=None):
    """
    Cross-validate the regressor and calculate average MSE, R-squared scores, and Pearson correlation.

    Parameters:
    X (numpy array or tf.Tensor): Feature matrix.
    y (numpy array or pandas Series or tf.Tensor): Labels.
    model (tf.keras.Model or Scikit-Learn Model): The regressor model.
    n_splits (int): Number of cross-validation splits.
    random_state (int): Random state for reproducibility.
    best_epochs (int, optional): Number of epochs to use for TensorFlow/Keras models. If None, defaults to 50.

    Returns:
    tuple: Average MSE, R-squared scores, and Pearson correlation for training and validation sets.
    """
    # Convert TensorFlow tensors to NumPy arrays if necessary
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    if isinstance(y, tf.Tensor):
        y = y.numpy()

    # Initialize lists to store metrics for all folds
    mse_train_list = []
    mse_val_list = []
    r2_train_list = []
    r2_val_list = []
    pearson_train_list = []
    pearson_val_list = []

    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Iterate over cross-validation folds
    for train_index, val_index in kf.split(X):
        # Split data into training and validation sets using indices from KFold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Check if the model is a Keras/TensorFlow model or a Scikit-Learn model
        if isinstance(model, tf.keras.Model):
            # Use the best number of epochs if provided; otherwise, default to 50
            epochs_to_use = best_epochs if best_epochs is not None else 50

            # Train the TensorFlow/Keras model
            # model.fit(X_train, y_train)

            # Predict values for both training and validation sets
            y_train_pred = model.predict(X_train).flatten()
            y_val_pred = model.predict(X_val).flatten()

        else:
            # Train the Scikit-Learn model
            model.fit(X_train, y_train)

            # Predict values for both training and validation sets
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

        # Calculate MSE for both training and validation sets
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)

        # Calculate R-squared for both training and validation sets
        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_val, y_val_pred)

        # Calculate Pearson correlation for both training and validation sets
        pearson_train, _ = pearsonr(y_train.values.flatten(), y_train_pred.flatten())
        pearson_val, _ = pearsonr(y_val.values.flatten(), y_val_pred.flatten())

        # Append metrics to lists
        mse_train_list.append(mse_train)
        mse_val_list.append(mse_val)
        r2_train_list.append(r2_train)
        r2_val_list.append(r2_val)
        pearson_train_list.append(pearson_train)
        pearson_val_list.append(pearson_val)

    # Calculate average metrics over all folds
    avg_mse_train = np.mean(mse_train_list)
    avg_mse_val = np.mean(mse_val_list)
    avg_r2_train = np.mean(r2_train_list)
    avg_r2_val = np.mean(r2_val_list)
    avg_pearson_train = np.mean(pearson_train_list)
    avg_pearson_val = np.mean(pearson_val_list)

    return avg_mse_train, avg_mse_val, avg_r2_train, avg_r2_val, avg_pearson_train, avg_pearson_val


def save_regressor_metrics(snp_data_loc, train_mse, train_r2, test_mse, test_r2, ind_test_mse, ind_test_r2,
                           avg_pearson_train, avg_pearson_val, ind_test_pearson, hopt=None):
    output_folder = "model_outputs"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    os.makedirs(output_folder, exist_ok=True)

    # Print the metrics
    print('Saving for :',hopt)
    # print("Train MSE:", train_mse)
    # print("Train R-squared:", train_r2)
    print("Test MSE:", test_mse)
    print("Test R-squared:", test_r2)
    print("Independent Test MSE:", ind_test_mse)
    print("Independent Test R-squared:", ind_test_r2)
    # print("Train Pearson Correlation:", avg_pearson_train)
    print("Validation Pearson Correlation:", avg_pearson_val)
    print("Independent Test Pearson Correlation:", ind_test_pearson)

    # Write the metrics to a file
    with open(os.path.join(output_folder,
                           f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_regressor_metrics.txt"),
              "w") as file:
        file.write("Train MSE: " + str(train_mse) + "\n")
        file.write("Train R-squared: " + str(train_r2) + "\n")
        file.write("Test MSE: " + str(test_mse) + "\n")
        file.write("Test R-squared: " + str(test_r2) + "\n")
        file.write("Independent Test MSE: " + str(ind_test_mse) + "\n")
        file.write("Independent Test R-squared: " + str(ind_test_r2) + "\n")
        file.write("Train Pearson Correlation: " + str(avg_pearson_train) + "\n")
        file.write("Validation Pearson Correlation: " + str(avg_pearson_val) + "\n")
        file.write("Independent Test Pearson Correlation: " + str(ind_test_pearson) + "\n")



import os
import pandas as pd
from scipy.stats import f


def compute_p_value(r2, n, k=1):
    f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
    p_value = 1 - f.cdf(f_stat, dfn=k, dfd=(n - k - 1))  # Degrees of freedom: dfn=k, dfd=(n - k - 1)
    return p_value, f_stat


def compute_snp_p_values(snp_data_loc, r2_values, n_samples, bim_file, hopt=None):
    output_folder = "model_outputs"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)
    output_file = os.path.join(output_folder,
                               f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_f_stats_p_values.txt")

    # Read the BIM file
    bim_data = pd.read_csv(bim_file, sep="\t", header=None)
    bim_data.columns = ['Chromosome', 'SNP', 'GeneticDist', 'Position', 'Allele1', 'Allele2']
    # Prepare lists to store results
    p_values = []
    f_stats = []
    effect_sizes = []
    matched_snps = []
    matched_chromosomes = []
    matched_positions = []
    r2s = []

    # Match R² SNPs to the BIM file
    for snp_id, r2 in r2_values.items():  # Use SNP IDs from r2_values as the index
        snp_id = str(snp_id)[:-2]  # Ensure SNP ID is in correct format
        snp_info = bim_data[bim_data['SNP'] == snp_id]  # Match SNP ID with the BIM file
        # print(snp_info)
        if not snp_info.empty:  # Ensure we have a match
            p_value, f_stat = compute_p_value(r2, n_samples)  # Compute p-value and F-statistic
            f2 = r2 / (1 - r2)  # Compute effect size (Cohen's f²)

            p_values.append(p_value)
            f_stats.append(f_stat)
            effect_sizes.append(f2)
            matched_snps.append(snp_id)  # Use the matched SNP ID
            r2s.append(r2)  # Store the R² value
            matched_chromosomes.append(snp_info['Chromosome'].values[0])  # Add chromosome info
            matched_positions.append(snp_info['Position'].values[0])  # Add position info
            # print()
    # Create a DataFrame with matched SNP information, R², p-values, F-statistics, and effect sizes
    p_value_df = pd.DataFrame({
        'SNP': matched_snps,  # Matched SNPs based on the index from r2_whole_scores
        'Chromosome': matched_chromosomes,  # Chromosome information
        'Position': matched_positions,  # Position information
        'R2': r2s,  # R² values (ensure the values are aligned with matched SNPs)
        'F-stat': f_stats,  # F-statistics
        'p-value': p_values,  # Computed p-values
        'Effect Size (f²)': effect_sizes  # Computed effect sizes
    })

    # Save the p-value DataFrame to a CSV file
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    p_value_df.to_csv(output_file, index=False)
    print(f"SNP p-values, effect sizes, and genomic positions saved to {output_file}")


import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_latent_space_clustering(snp_data_loc, vae_model, X_data, method='tsne', n_components=2, hopt=None):
    """
    Visualize the latent space clustering of the VAE model.

    Args:
        vae_model (tf.keras.Model): The trained VAE model.
        X_data (pd.DataFrame or np.ndarray): Input genotype data (individuals x SNPs).
        labels (pd.Series or np.ndarray): Labels for the individuals (e.g., case/control or phenotypes).
        method (str): Dimensionality reduction method ('tsne' or 'pca').
        n_components (int): Number of dimensions to reduce to (default is 2).
    """
    # Ensure data is in NumPy format
    # if isinstance(X_data, pd.DataFrame):
    #     X_data = X_data.values
    output_folder = "model_outputs"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    # Get the latent space representation from the encoder
    encoder = vae_model.encoder
    z_mean, _ = tf.split(encoder.predict(X_data), num_or_size_splits=2, axis=1)
    print(z_mean)
    # Perform dimensionality reduction
    if method == 'tsne':
        print("Applying t-SNE for dimensionality reduction...")
        reduced_latent = TSNE(n_components=n_components).fit_transform(z_mean)
    elif method == 'pca':
        print("Applying PCA for dimensionality reduction..")
        reduced_latent = PCA(n_components=n_components).fit_transform(z_mean)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")

    # Create a scatter plot of the reduced latent space
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_latent[:, 0], y=reduced_latent[:, 1], palette='Set1', s=60, alpha=0.8)
    plt.title(f'Latent Space Clustering using {method.upper()}', fontsize=16)
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    # plt.legend(title='Labels', loc='best')
    plt.savefig(
        os.path.join(output_folder,
                     f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_{method.upper()}_clustering.png"))
    plt.show()


def permutation_feature_importance(snp_data_loc, X_original, X_reconstructed, model, snp_ids, n_permutations=5,
                                   hopt=None):
    """
    Compute permutation feature importance for each SNP by evaluating the drop in R² score and save the results to a CSV file.

    Args:
        X_original (pd.DataFrame or np.ndarray): Original genotype matrix (individuals x SNPs).
        X_reconstructed (pd.DataFrame or np.ndarray): Reconstructed genotype matrix (individuals x SNPs).
        model (tf.keras.Model): The trained VAE model used for reconstruction.
        snp_ids (list or pd.Series): List or Series of SNP IDs corresponding to columns of the genotype matrix.
        n_permutations (int): Number of times to permute each SNP.
        output_file (str): Path to the file where the results will be saved (default: 'permutation_importance_scores.csv').

    Returns:
        importance_scores (pd.Series): The drop in R² score for each SNP, indexed by SNP ID.
    """
    output_folder = "model_outputs"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)
        output_file = os.path.join(output_folder,
                                   f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_{n_permutations}_permutation_scores.txt")
    if isinstance(X_original, pd.DataFrame):
        X_original = X_original.values

    # Initialize an array to store the R² drop for each SNP
    importance_scores = np.zeros(X_original.shape[1])

    # Compute the baseline R² for each SNP
    r2_baseline = evaluate_r2(X_original, X_reconstructed)

    for snp_idx in range(X_original.shape[1]):  # Loop through SNPs by index
        r2_permuted_list = []

        for _ in range(n_permutations):
            # Shuffle the SNP values for this particular SNP
            X_permuted = X_original.copy()
            np.random.shuffle(X_permuted[:, snp_idx])  # Shuffle SNP column

            # Reconstruct using the model with permuted SNP
            X_reconstructed_permuted = model.predict(X_permuted)

            # Compute R² after permutation for the specific SNP
            r2_permuted = evaluate_r2(X_original, X_reconstructed_permuted)
            r2_permuted_list.append(r2_permuted[snp_idx])

        # Compute the mean R² after permutation and compare it to the baseline
        mean_r2_permuted = np.mean(r2_permuted_list)
        importance_scores[snp_idx] = r2_baseline[snp_idx] - mean_r2_permuted

    # Convert the importance scores into a pandas Series, indexed by SNP IDs
    importance_series = pd.Series(importance_scores, index=snp_ids, name='Feature Importance')

    # Save the importance scores to a CSV file
    importance_series.to_csv(output_file, index=True)
    print(f"Permutation importance scores saved to {output_file}")

    return importance_series


def rank_distance_correlation(X, Z):
    """
    Compute Rank Distance Correlation (RdCorr) between each feature in X and the latent representation Z.

    Args:
        X (np.ndarray): Original input matrix (samples x features).
        Z (np.ndarray): Latent space representation (samples x latent dimensions).

    Returns:
        rdc_scores (np.ndarray): Rank distance correlation scores for each feature in X.
    """
    n_samples, n_features = X.shape
    rdc_scores = np.zeros(n_features)

    for i in range(n_features):
        # Compute pairwise distances for the i-th feature and the latent space
        d_X = pairwise_distances(X[:, [i]], metric='euclidean')
        d_Z = pairwise_distances(Z, metric='euclidean')

        # Compute rank transformation of distances
        rank_X = np.argsort(np.argsort(d_X, axis=0), axis=0)
        rank_Z = np.argsort(np.argsort(d_Z, axis=0), axis=0)

        # Compute the rank distance covariance
        cov_XZ = np.sum((rank_X - rank_X.mean()) * (rank_Z - rank_Z.mean())) / n_samples
        var_X = np.sum((rank_X - rank_X.mean()) ** 2) / n_samples
        var_Z = np.sum((rank_Z - rank_Z.mean()) ** 2) / n_samples

        # Calculate Rank Distance Correlation (RdCorr)
        rdc_scores[i] = cov_XZ / np.sqrt(var_X * var_Z)

    return rdc_scores


def compute_snp_rdc_scores(snp_data_loc,vae_model, X_data, snp_ids, hopt=None):
    """
    Compute Rank Distance Correlation (RdCorr) scores for each SNP feature.

    Args:
        snp_data_loc:
        vae_model (tf.keras.Model): Trained VAE model.
        X_data (pd.DataFrame or np.ndarray): Input genotype data (individuals x SNPs).
        snp_ids (pd.Series or list): List or Series of SNP IDs corresponding to columns of the genotype matrix.
        hopt (str): Optional string for hyperopt or other directory handling.

    Returns:
        pd.DataFrame: DataFrame with SNPs and their corresponding RdCorr scores.
    """
    output_folder = "model_outputs"
    if hopt:
        output_folder = os.path.join(output_folder, hopt)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder,
                               f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_rdcorr_scores.csv")

    if isinstance(X_data, pd.DataFrame):
        X_data = X_data.values

    # Get the latent space representation from the encoder
    encoder = vae_model.encoder
    Z_mean, _ = tf.split(encoder.predict(X_data), num_or_size_splits=2, axis=1)  # Use z_mean from VAE

    # Compute RdCorr for each SNP feature
    rdc_scores = rank_distance_correlation(X_data, Z_mean)

    # Create a DataFrame to store SNPs and their RdCorr scores
    rdc_scores_df = pd.DataFrame({
        'SNP': snp_ids,
        'RdCorr': rdc_scores
    })

    # Save RdCorr scores to a CSV file
    rdc_scores_df.to_csv(output_file, index=False)
    print(f"Rank Distance Correlation (RdCorr) scores saved to {output_file}")

    return rdc_scores_df

# Additional metrics functions
# Adjusted R^2 Calculation
def adjusted_r2_score(y_true, y_pred, n, p):
    """
    Compute Adjusted R² score.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        n (int): Number of observations.
        p (int): Number of predictors.

    Returns:
        float: Adjusted R² score.
    """
    r2 = np.mean(utils.evaluate_r2(y_true, y_pred))
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)


# Pearson Correlation




def compute_pearson_correlation(y_true, y_pred):
    """
    Compute Pearson correlation coefficient between true and predicted values.

    Parameters:
        y_true (np.ndarray or pd.DataFrame): True values.
        y_pred (np.ndarray or pd.DataFrame): Predicted values.

    Returns:
        float: Pearson correlation coefficient.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return corr


def save_adjusted_r2_and_pearson_corr(snp_data_loc, adj_r2_train, adj_r2_test, adj_r2_whole,
                                      pearson_corr_train, pearson_corr_test, pearson_corr_whole,
                                      rmse_train, rmse_test, rmse_whole, hopt=None):
    """
    Save Adjusted R², Pearson Correlation, and RMSE metrics for train, test, and whole datasets.

    Parameters:
        snp_data_loc (str): File path for SNP data.
        adj_r2_train (float): Adjusted R² for the training data.
        adj_r2_test (float): Adjusted R² for the testing data.
        adj_r2_whole (float): Adjusted R² for the entire dataset.
        pearson_corr_train (float): Pearson Correlation for the training data.
        pearson_corr_test (float): Pearson Correlation for the testing data.
        pearson_corr_whole (float): Pearson Correlation for the entire dataset.
        rmse_train (float): RMSE for the training data.
        rmse_test (float): RMSE for the testing data.
        rmse_whole (float): RMSE for the entire dataset.
        hopt (str): Optional subdirectory for saving outputs.
    """
    output_folder = "model_outputs"
    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    os.makedirs(output_folder, exist_ok=True)

    # Print the metrics
    print("Adjusted R² (Train):", adj_r2_train)
    print("Adjusted R² (Test):", adj_r2_test)
    print("Adjusted R² (Whole):", adj_r2_whole)
    print("Pearson Correlation (Train):", pearson_corr_train)
    print("Pearson Correlation (Test):", pearson_corr_test)
    print("Pearson Correlation (Whole):", pearson_corr_whole)
    print("RMSE (Train):", rmse_train)
    print("RMSE (Test):", rmse_test)
    print("RMSE (Whole):", rmse_whole)

    # Write the metrics to a file
    metrics_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_metrics.txt")
    with open(metrics_file_path, "w") as metrics_file:
        metrics_file.write(f"Adjusted R² (Train): {adj_r2_train}\n")
        metrics_file.write(f"Adjusted R² (Test): {adj_r2_test}\n")
        metrics_file.write(f"Adjusted R² (Whole): {adj_r2_whole}\n")
        metrics_file.write(f"Pearson Correlation (Train): {pearson_corr_train}\n")
        metrics_file.write(f"Pearson Correlation (Test): {pearson_corr_test}\n")
        metrics_file.write(f"Pearson Correlation (Whole): {pearson_corr_whole}\n")
        metrics_file.write(f"RMSE (Train): {rmse_train}\n")
        metrics_file.write(f"RMSE (Test): {rmse_test}\n")
        metrics_file.write(f"RMSE (Whole): {rmse_whole}\n")
    print(f"Metrics saved to {metrics_file_path}")


def compute_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
        y_true (np.ndarray or pd.DataFrame): True values.
        y_pred (np.ndarray or pd.DataFrame): Predicted values.

    Returns:
        float: RMSE score.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    return np.sqrt(mean_squared_error(y_true, y_pred))
