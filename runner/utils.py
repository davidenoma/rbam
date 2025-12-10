import os
import subprocess

import keras
import pandas as pd
import sys
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, pairwise_distances
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns


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

    # Run the awk command and capture the output
    result = subprocess.run(awk_command, shell=True, capture_output=True, text=True)

    # Convert the result to a list of phenotype values
    phenotype = result.stdout.strip().split('\n')

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
    # # Reset index
    snp_data.reset_index(inplace=True, drop=True)
    # snp_data = snp_data.dropna(axis=1)
    print("Snp data shape:", snp_data.shape)
    print("Done loading the recoded genotype")
    # Preprocess the SNP data (standardization and train-test split)
    X_train, X_test, = train_test_split(snp_data, test_size=0.2, random_state=42)

    return X_train, X_test, snp_data



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




def save_classifier_metrics(snp_data_loc, accuracy, auc, ind_test_accuracy,
                            ind_test_auc,
                         hopt=None):
    """
    Save the classifier metrics including accuracy, AUC, and R² to a file.

    Parameters:
    snp_data_loc (str): Location of the SNP data.

    accuracy (float): Validation accuracy score.
    auc (float): Validation AUC score.
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

        file.write(f"Model Accuracy: {accuracy}\n")
        file.write(f"Model AUC: {auc}\n")
        file.write(f"Independent Test Accuracy: {ind_test_accuracy}\n")
        file.write(f"Independent Test AUC: {ind_test_auc}\n")



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

        # Append metrics to lists
        mse_train_list.append(mse_train)
        mse_val_list.append(mse_val)
        r2_train_list.append(r2_train)
        r2_val_list.append(r2_val)


    # Calculate average metrics over all folds
    # avg_mse_train = np.mean(mse_train_list)
    avg_mse_val = np.mean(mse_val_list)
    # avg_r2_train = np.mean(r2_train_list)
    avg_r2_val = np.mean(r2_val_list)


    # Return all metrics
    return (

        avg_mse_val,
        avg_r2_val
    )


def cross_validate_classifier(X, y, model, n_splits=5, random_state=11):
    """
    Uses K-fold CV without re-using the previously fitted `best_model`.
    The estimator is *cloned* (untrained) at every fold to avoid leakage.
    """
    X, y = np.asarray(X), np.asarray(y)
    kf   = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state)

    acc_val, auc_val = [] , []

    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if isinstance(model, tf.keras.Model):
            fold_model = tf.keras.models.clone_model(model, clone_function=None)
            fold_model.compile(optimizer=tf.keras.optimizers.Adam(),
                               loss='binary_crossentropy')
            fold_model.fit(X_tr, y_tr,
                           epochs=10, batch_size=32, verbose=0)
            y_val_proba = fold_model.predict(X_val, verbose=0).ravel()
        else:
            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            y_val_proba = (fold_model.predict_proba(X_val)[:, 1]
                           if hasattr(fold_model, "predict_proba")
                           else fold_model.predict(X_val))


        y_val_pred = (y_val_proba > 0.5).astype(int)


        acc_val.append(accuracy_score(y_val, y_val_pred))
        auc_val.append(roc_auc_score(y_val, y_val_proba))

    return ( np.mean(acc_val),
            np.mean(auc_val))



def save_mse_values(snp_data_loc, mse_test, mse_whole, hopt=None):
    output_folder = "model_outputs"
    if hopt:
        output_folder = output_folder + "/" + hopt
    os.makedirs(output_folder, exist_ok=True)

    # Print the MSE values
    print("Mean Squared Error (MSE) between input and reconstructed test data:", mse_test)
    print("Mean Squared Error (MSE) between input and reconstructed whole data:", mse_whole)

    # Write the MSE values to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_mse_results.txt"),
              "w") as file:

        file.write("Mean Squared Error (MSE) between reconstructed test data: " + str(mse_test) + "\n")
        file.write("Mean Squared Error (MSE) between reconstructed whole data: " + str(mse_whole) + "\n")


def save_mse_values_cv(snp_data_loc, mse_test, hopt=None):
    output_folder = "model_outputs"
    if hopt:
        output_folder = output_folder + "/" + hopt
    os.makedirs(output_folder, exist_ok=True)

    # Print the MSE values

    print("Mean Squared Error (MSE) between input and reconstructed test data:", mse_test)

    # Write the MSE values to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_mse_results.txt"),
              "w") as file:
        file.write("Mean Squared Error (MSE) between reconstructed test data: " + str(mse_test) + "\n")


def save_r2_scores(snp_data_loc, r2test, r2whole, hopt=None):
    output_folder = "model_outputs"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    os.makedirs(output_folder, exist_ok=True)
    # Print the R2 scores
    print("R-squared (R2) for  test data:", r2test)
    print("R-squared (R2) for  whole data:", r2whole)

    # Write the R2 scores to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_r2_results.txt"),
              "w") as file:
        file.write("R-squared (R2) for  test data: " + str(r2test) + "\n")
        file.write("R-squared (R2) for  whole data: " + str(r2whole) + "\n")


def save_r2_scores_cv(snp_data_loc, r2test, hopt=None):
    output_folder = "model_outputs/cv"

    if hopt:
        output_folder = os.path.join(output_folder, hopt)

    os.makedirs(output_folder, exist_ok=True)
    # Print the R2 scores
    print("R-squared (R2) for  test data:", r2test)

    # Write the R2 scores to a file
    with open(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(snp_data_loc))[0]}_r2_results.txt"),
              "w") as file:
        file.write("R-squared (R2) for  test data: " + str(r2test) + "\n")

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