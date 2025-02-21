import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
import utils


def intersect_sum_stats_and_bim_file(bim_file_input, filtered_sum_stats_file, full_sum_stats_file):
    bim_data = {}
    with open(bim_file_input, 'r') as bim_file:
        for line in bim_file:
            fields = line.strip().split()
            rs_id = fields[1]
            bim_data[rs_id] = fields
    effect_data = {}
    with open(full_sum_stats_file, 'r') as effect_file:
        for line in effect_file:
            if not line.startswith("rsID"):
                fields = line.strip().split()
                rs_id = fields[0]
                effect_data[rs_id] = fields[1:]
    # Intersect the dictionaries based on rsID
    intersected_data = {}
    for rs_id in bim_data:
        if rs_id in effect_data:
            bim_row = bim_data[rs_id]
            effect_row = effect_data[rs_id]
            # if bim_row[4] == effect_row[0]:  # Column 5 matches column 7
            intersected_data[rs_id] = bim_row + effect_row
    # Write the intersected data to a new file
    with open(filtered_sum_stats_file, 'w') as output_file:
        for rs_id in intersected_data:
            output_file.write('\t'.join(intersected_data[rs_id]) + '\n')
    return filtered_sum_stats_file


# Define a function to read rsIDs from the second file
def read_rsids_from_filtered_sum_stats(filename):
    rsids = set()
    with open(filename, 'r') as file:
        for line in file:
            rsid = line.strip().split()[1]
            rsids.add(rsid)
    return rsids




def extract_phenotype(genotype_file):
    # Construct the awk command to extract the phenotype column, skipping the header
    awk_command = f"awk -F' ' 'NR > 1 {{print $6}}' {genotype_file}.raw"

    # Run the awk command and capture the output
    result = subprocess.run(awk_command, shell=True, capture_output=True, text=True)

    # Convert the result to a list of phenotype values
    phenotype = result.stdout.strip().split('\n')

    # Remove any empty strings that may have resulted from trailing newlines
    phenotype = [p for p in phenotype if p]

    # Convert to a Pandas Series
    phenotype = pd.Series(phenotype, dtype='float')

    return phenotype


def load_real_genotype_data_rename_cols(snp_data_loc):
    print('LOADING GENOTYPE', snp_data_loc)

    # Extract the phenotype column using awk
    phenotype = extract_phenotype(snp_data_loc)

    chunk_size = 10000
    # Columns to skip during reading, including 'PHENOTYPE'
    columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']

    # Initialize an empty DataFrame
    snp_data_ = pd.DataFrame()

    # Read data in chunks, skipping unwanted columns
    reader = pd.read_csv(f"{snp_data_loc}", sep=" ", usecols=lambda column: column not in columns_to_skip,
                         dtype='int8', chunksize=chunk_size)

    for chunk in reader:
        snp_data_ = pd.concat([snp_data_, chunk])

    # Reset index after reading all chunks
    snp_data_.reset_index(inplace=True, drop=True)

    # Rename columns (removing suffix)
    colnames = snp_data_.columns
    new_col_names = [col[:-2] for col in colnames]
    snp_data_.columns = new_col_names

    # Return the processed genotype data and the phenotype column
    return snp_data_, phenotype


def sum_stats_onlyweighted_real_genotype_data(snp_data, sum_stats, ss_beta_column):
    """
    Calculate the genotype scores using just summary statistics.

    Parameters:
    - genotype_file_filtered (str): Path to the filtered genotype file.
    - sum_stats (str): Path to the sum statistics file.
    - ss_beta_column (int): column in intersected file that has summary statistics betas

    Returns:
    None
    """
    sum_stat_weighted = {}
    sum_stats = pd.read_csv(sum_stats, sep="\t", header=None)
    if not sum_stats.empty:
        for index, row in sum_stats.iterrows():
            snp_id = row.iloc[1]  # Assuming the first column contains SNP IDs
            beta = np.float64(row.iloc[ss_beta_column])  # Assuming the tdtype=floathird column contains the multipliers
            print("Sum stat beta", beta)
            print(snp_id)
            if snp_id in snp_data.columns:
                sum_stat_weighted[snp_id] = snp_data[snp_id] * beta

    modified_data_ss_only = pd.DataFrame(sum_stat_weighted,dtype=np.float64)
    print("Sum Stats with only sum stats: ", modified_data_ss_only.shape)
    # print("Sum Stats with only sum stats: ", modified_data_ss_only.head())

    return modified_data_ss_only

    # def sum_stat_plus_bridged_real_genotype_data(snp_data_input, sum_stats_filtered, ss_beta_column, data_bridge_weights):
    #
    #     sum_stats = pd.read_csv(sum_stats_filtered, sep="\t", header=None)
    #     # Initialize a dictionary to store modified columns
    #     # Apply sum stats multipliers if provided
    #     modified_columns = {}
    #     # betas =
    #     snp_id_betas = {}
    #     #muluplying the SNPs by the betas from the summary statistics.
    #     if not sum_stats.empty:
    #         for index, row in sum_stats.iterrows():
    #             snp_id = row.iloc[1]  # Assuming the first column contains SNP IDs
    #             beta = float(row.iloc[ss_beta_column])
    #             # Assuming the eighth column contains the multipliers
    #             if snp_id in snp_data_input.columns:
    #                 snp_id_betas[snp_id] = beta
    #                 # modified_columns[snp_id] = snp_data_input[snp_id] * beta
    #
    #     modified_data = pd.DataFrame(modified_columns)
    #
    #     weighted_columns = {}
    #
    #     if not data_bridge_weights.empty:t
    #         for index, row in data_bridge_weights.iterrows():
    #             snp_id = row.iloc[0]  # Assuming the first column contains SNP IDs
    #             bridge_weight = float(row.iloc[3])
    #
    #             # Assuming the third column contains the multipliers
    #             if snp_id in modified_data.columns:
    #                 weighted_columns[snp_id] = modified_data[snp_id] * bridge_weight
    #                 # modified_data[snp_id] = modified_data[snp_id] * beta
    #
    #     # Create a file with modified columns to elimiate snps with zero weights
    #     modified_data_output = pd.DataFrame(weighted_columns)
    #     if modified_data_output.shape[0] < 1:
    #         print('No corresponding bridge weight found.')
    # print(modified_data_output.shape,modified_data_output.sum().sum())

    # return modified_data_output


# def sum_stat_plus_bridged_real_genotype_data(snp_data_input, sum_stats_filtered, ss_beta_column, data_bridge_weights):
#     sum_stats = pd.read_csv(sum_stats_filtered, sep="\t", header=None)
#
#     # Initialize a dictionary to store modified columns
#     modified_columns = {}
#     if not sum_stats.empty:
#         for index, row in sum_stats.iterrows():
#             snp_id = row.iloc[1]  # Assuming the first column contains SNP IDs
#             beta = float(row.iloc[ss_beta_column])
#             if snp_id in snp_data_input.columns:
#                 if snp_id in data_bridge_weights['SNP_ID'].values:
#                     bridge_weight = np.float16(data_bridge_weights[data_bridge_weights['SNP_ID'] == snp_id].iloc[0, 3])
#                     # print('Bridge weights:',bridge_weight)
#                     modified_columns[snp_id] = snp_data_input[snp_id] * (beta + bridge_weight)
#                     # print("mod snp_id: ",modified_columns[snp_id] )
#                 # else:
#                 #     modified_columns[snp_id] = snp_data_input[snp_id] * beta
#
#     modified_data = pd.DataFrame(modified_columns,dtype=np.float16)
#     print(modified_data.shape, '<- Modified data shape.')
#     print(modified_data.head, '<- Modified data head.')
#     return modified_data
def sum_stats_bridge_weights(genotype_file_filtered, sum_stats, data_bridge_weights, ss_beta_column):
    print('Using bridge + summary statistics : ', data_bridge_weights, sum_stats)
    snp_data, phenotype = load_real_genotype_data_rename_cols(f'{genotype_file_filtered}.raw')
    phenotype = phenotype.replace({1: 0, 2: 1})
    bridge_weights = pd.read_csv(data_bridge_weights, dtype={'Score': np.float16})
    sum_stats = pd.read_csv(sum_stats, sep="\t", header=None)

    modified_columns = {}
    if not sum_stats.empty:
        for index, row in sum_stats.iterrows():
            snp_id = row.iloc[1]
            beta = float(row.iloc[ss_beta_column])
            # print(f'Processing SNP ID: {snp_id}, Beta: {beta}')

            if snp_id in snp_data.columns:
                if snp_id in bridge_weights['SNP_ID'].values:
                    bridge_weight = np.float16(bridge_weights[bridge_weights['SNP_ID'] == snp_id].iloc[0, 3])
                    modified_columns[snp_id] = snp_data[snp_id] * (beta + bridge_weight)
                    # print(f'Modified {snp_id}: {modified_columns[snp_id].head()}')

    modified_data = pd.DataFrame(modified_columns, dtype=np.float16)
    print(modified_data.shape, '<- Modified data shape.')
    # print(modified_data.head(), '<- Modified data head.')

    snp_data_br_and_w = modified_data

    print('Evaluating Prediction')
    y = phenotype
    gprs_scores = []
    for index, row in snp_data_br_and_w.iterrows():
        # print(row)
        row_sum = row.astype(float).sum()
        # print(row_sum,'row_sum')
        # print(row.sum(axis=))
        gprs_scores.append(row_sum)
    # Convert gprs_scores to a NumPy array
    gprs_scores = np.array(gprs_scores).reshape(-1, 1)

    # Apply StandardScaler to gprs_scores
    scaler = StandardScaler()
    gprs_scores_scaled = scaler.fit_transform(gprs_scores)
    # Predict using scaled scores
    y_pred_sc = (gprs_scores_scaled > 0.5).astype(int).flatten()
    auroc = roc_auc_score(y, y_pred_sc)
    r_squared = utils.evaluate_r2(y, y_pred_sc)
    accuracy = accuracy_score(y, y_pred_sc)

    print("AUROC:", auroc)
    print("R-squared:", r_squared)
    print("Accuracy:", accuracy)

    print('Evaluating L2 Reg')
    # scaler = StandardScaler()
    # Scale input data
    # input_scaled = scaler.fit_transform(snp_data_br_and_w)

    # Set up GridSearchCV for Ridge regression
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
    grid_search.fit(snp_data_br_and_w, phenotype)

    # Get the best model from grid search
    model = grid_search.best_estimator_
    gprs_scores_reg = model.predict(snp_data_br_and_w).tolist()
    gprs_scores_reg = np.array(gprs_scores_reg).reshape(-1, 1)
    gprs_scores_reg  = scaler.fit_transform(gprs_scores_reg)

    y_pred_reg = (gprs_scores_reg > 0.5).astype(int).flatten()

    # y_pred_reg = (np.array(gprs_scores_reg) > 0.5).astype(int)
    auroc = roc_auc_score(y, y_pred_reg)
    r_squared = utils.evaluate_r2(y, y_pred_reg)
    accuracy = accuracy_score(y, y_pred_reg)

    print('Regularized metrics: ')
    print('AUROC: Regularized:', auroc)
    print('R Squared Regularized ', r_squared)

    print('Accuracy Regularized ', accuracy)
    # ml_classifiers(snp_data_br_and_w,y)


# def sum_stats_bridge_weights(genotype_file_filtered, sum_stats, data_bridge_weights, ss_beta_column):
#     # # Load real genotype data
#     print('Using bridge + summary statistics : ', data_bridge_weights, sum_stats)
#     snp_data, phenotype = load_real_genotype_data_rename_cols(genotype_file_filtered)
#     #
#     bridge_weights = pd.read_csv(data_bridge_weights,dtype={'Score':np.float16})
#     sum_stats = pd.read_csv(sum_stats, sep="\t", header=None)
#     # snp_data_br_and_w = sum_stat_plus_bridged_real_genotype_data(snp_data, sum_stats, ss_beta_column, bridge_weights)
#     # Initialize a dictionary to store modified columns
#     # print(snp_data.columns)
#     modified_columns = {}
#     if not sum_stats.empty:
#         for index, row in sum_stats.iterrows():
#             snp_id = row.iloc[1]
#             # print(snp_id)
#             # Assuming the first column contains SNP IDs
#             beta = float(row.iloc[ss_beta_column])
#             if snp_id in snp_data.columns:
#                 if snp_id in bridge_weights['SNP_ID'].values:
#
#                     bridge_weight = np.float16(bridge_weights[bridge_weights['SNP_ID'] == snp_id].iloc[0, 3])
#                     # print('Bridge weights:',bridge_weight)
#                     modified_columns[snp_id] = snp_data[snp_id] * (beta * bridge_weight)
#                     # print("mod snp_id: ",modified_columns[snp_id] )
#                 # else:
#                 #     modified_columns[snp_id] = snp_data_input[snp_id] * beta
#
#     modified_data = pd.DataFrame(modified_columns, dtype=np.float16)
#     print(modified_data.shape, '<- Modified data shape.')
#     print(modified_data.head, '<- Modified data head.')
#     # return modified_data
#
#     snp_data_br_and_w  = modified_data
#
#     # print(snp_data_br_and_w.isna().sum())
#     # #
#     # try:
#     # col_names = snp_data_br_and_w.columns
#     # scaler = StandardScaler()
#     # snp_data_br_and_w = scaler.fit_transform(snp_data_br_and_w)
#     # snp_data_br_and_w = pd.DataFrame(snp_data_br_and_w, columns=col_names)
#     # print(snp_data_br_and_w.head())
#     #
#     print('Evaluating Prediction')
#     y = phenotype
#     gprs_scores = []
#     for index, row in snp_data_br_and_w.iterrows():
#         row_sum = row.astype(float).sum()
#         # print(row_sum,"row sum")
#         gprs_scores.append(row_sum)
#     y_pred_sc = (np.array(gprs_scores) > 0.5).astype(int)
#     # Compute AUROC
#     auroc = roc_auc_score(y, y_pred_sc)
#     # Compute R-squared
#     r_squared = utils.evaluate_r2(y, y_pred_sc)
#     # Compute accuracy
#     accuracy = accuracy_score(y, y_pred_sc)
#     print("AUROC:", auroc)
#     print("R-squared:", r_squared)
#     print("Accuracy:", accuracy)
#
#     print('Evaluatin L2 Reg')
#     scaler = StandardScaler()
#     input_scaled = scaler.fit_transform(snp_data_br_and_w)
#     # Define the parameter grid for alpha (adjust the range as needed)
#     param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
#     # Create a GridSearchCV object
#     grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
#     # Fit the grid search
#     grid_search.fit(input_scaled, phenotype)
#     # Get the best model
#     model = grid_search.best_estimator_
#     # Use the model for prediction
#     gprs_scores_reg = model.predict(input_scaled).tolist()
#     y_pred_reg = (np.array(gprs_scores_reg) > 0.5).astype(int)
#
#     # Compute AUROC for L2 Regularized
#     auroc = roc_auc_score(y, y_pred_reg)
#     # Compute R-squared
#     r_squared = utils.evaluate_r2(y, y_pred_reg)
#     # Compute accuracy
#     accuracy = accuracy_score(y, y_pred_reg)
#     print('Regularized metrics: ')
#     print('AUROC: Regularized:', auroc)
#     print('R Squared Regularized ', r_squared)
#     print('Accuracy Regularized ', accuracy)
#     # execute_GPRS_and_plot(snp_data_br_and_w)
#     # evaluate_results(snp_data_br_and_w, phenotype)
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")


def using_only_sum_statistics(genotype_file_filt, sum_stats, ss_beta_column):
    print('Using only summary statistics : ', sum_stats)
    # # unweighted_genotype, pheno = utils.load_genotype_data(genotype_File_filt)
    snp_data_2, phenotype = load_real_genotype_data_rename_cols(f'{genotype_file_filt}.raw')
    phenotype = phenotype.replace({1: 0, 2: 1})
    sstat_weighted_real_geno = sum_stats_onlyweighted_real_genotype_data(snp_data_2, sum_stats, ss_beta_column)
    #
    # col_names = sstat_weighted_real_geno
    scaler = StandardScaler()
    sstat_weighted_real_geno_pd = scaler.fit_transform(sstat_weighted_real_geno)
    sstat_weighted_real_geno = pd.DataFrame(sstat_weighted_real_geno_pd, columns=sstat_weighted_real_geno.columns)
    print(sstat_weighted_real_geno.head())
    execute_GPRS_and_plot(sstat_weighted_real_geno)
    print(sstat_weighted_real_geno.isna().sum(),"na sum")
    # print(sstat_weighted_real_geno,phenotype
    #       )
    print('-------Evluating results-------')
    evaluate_results(sstat_weighted_real_geno, phenotype)
    y = phenotype
    gprs_scores = get_individual_BW_summed_GPRS(input_modified_geno=sstat_weighted_real_geno)

    gprs_scores = np.array(gprs_scores).reshape(-1,1).flatten()
    print(gprs_scores)
    # Apply StandardScaler to gprs_scores
    print(gprs_scores)
    # Predict using scaled scores
    y_pred_sc = (gprs_scores > 0.5).astype(int)
    print(y_pred_sc)
    # y_pred_sc = (np.array(y_pred) > 0.5).astype(int)
    # Compute AUROCx
    auroc = roc_auc_score(y, y_pred_sc)
    # Compute R-squared
    r_squared = utils.evaluate_r2(y, y_pred_sc)
    # Compute accuracy
    accuracy = accuracy_score(y, y_pred_sc)
    print("AUROC:", auroc)
    print("R-squared:", r_squared)
    print("Accuracy:", accuracy)
    # return snp_data_2,phenotype

def get_individual_BW_summed_GPRS_L2_REG(input_modified_geno, phenotype):
    # Normalize data (optional, adjust based on your data)
    # scaler = StandardScaler()
    # input_scaled = scaler.fit_transform(input_modified_geno)
    # Normalize data
    # scaler = StandardScaler()
    # input_scaled = scaler.fit_transform(input_modified_geno)

    # Define the parameter grid for alpha (adjust the range as needed)
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}

    # Create a GridSearchCV object
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5)

    # Fit the grid search
    grid_search.fit(input_modified_geno, phenotype)
    # Get the best model
    model = grid_search.best_estimator_
    # Use the model for prediction
    gprs_scores = model.predict(input_modified_geno).tolist()
    return gprs_scores
    # Define the parameter grid for alpha (adjust the range as needed)
    # param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
    # # Create a GridSearchCV object
    # grid_search = GridSearchCV(Ridge(), param_grid, cv=5)  # Adjust cv for number of folds
    # # Fit the grid search
    # grid_search.fit(input_modified_geno, phenotype)
    # # Get the best model (no need for best_alpha)
    # model = grid_search.best_estimator_
    # # Use the model for prediction
    # gprs_scores = []
    # for index, row in input_modified_geno.iterrows():
    #     predicted_gprs = model.predict(row.values.reshape(1, -1))[0]
    #     gprs_scores.append(predicted_gprs)
    # return gprs_scores


# Execution step
def get_individual_BW_summed_GPRS(input_modified_geno):
    gprs_scores = []
    for index, row in input_modified_geno.iterrows():
        row_sum = row.astype(float).sum()
        # print(row_sum,"row sum")
        gprs_scores.append(row_sum)
    return gprs_scores


def execute_GPRS_and_plot(genotype):
    # Assuming snp_data_br_and_w is your DataFrame
    gprs_scores = get_individual_BW_summed_GPRS(genotype)
    # Write gprs_scores to a file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('results/' + current_time + '_' + 'gprs_scores.txt', 'w') as f:
        for score in gprs_scores:
            f.write(str(score) + '\n')
    # Calculate the 10th and 90th percentiles
    p10 = np.percentile(gprs_scores, 10)
    p90 = np.percentile(gprs_scores, 90)
    # Plot the distribution
    plt.hist(gprs_scores, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('GPRS Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of GPRS Scores')
    # Highlight top 10% and bottom 10% with red bars
    plt.axvline(x=p10, color='red', linestyle='--', label='10th percentile')
    plt.axvline(x=p90, color='red', linestyle='--', label='90th percentile')
    # Save the plot to a file
    plt.savefig('results/' + current_time + '_' + 'gprs_scores_distribution_with_percentiles.png')
    # Show the plot
    plt.legend()
    plt.show()


def round_to_1_or_2(num, midpoint):
    if num < midpoint:
        return 1
    else:
        return 2


# def evaluate_results(genotype, pheno):
#     y = pheno
#     n_trials = 5  # Number of trials
#     auroc_scores = []
#     r_squared_scores = []
#     accuracy_scores = []
#     for _ in range(n_trials):
#         y_pred = get_individual_BW_summed_GPRS(genotype)
#         # average = np.array(y_pred).mean()
#
#         y_pred_sc = (np.array(y_pred) > 0.5).astype(int)
#         # y_pred_sc = [round_to_1_or_2(num, average) for num in y_pred]
#         # Compute AUROC
#         auroc = roc_auc_score(y, y_pred_sc)
#         # Compute R-squared
#         r_squared = utils.evaluate_r2(y, y_pred_sc)
#         # Compute accuracy
#         accuracy = accuracy_score(y, y_pred_sc)
#         auroc_scores.append(auroc)
#         r_squared_scores.append(r_squared)
#         accuracy_scores.append(accuracy)
#
#     # Calculate average scores across trials
#     avg_auroc = sum(auroc_scores) / len(auroc_scores)
#     avg_r_squared = sum(r_squared_scores) / len(r_squared_scores)
#     avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
#
#     print("Average AUROC across trials:", avg_auroc)
#     print("Average R-squared across trials:", avg_r_squared)
#     print("Average accuracy across trials:", avg_accuracy)

def evaluate_results(genotype, pheno):
    y = pheno
    print(y,'y')
    gprs_scores = get_individual_BW_summed_GPRS(input_modified_geno=genotype)

    gprs_scores = np.array(gprs_scores).reshape(-1, 1)
    # Apply StandardScaler to gprs_scores
    scaler = StandardScaler()
    gprs_scores_scaled = scaler.fit_transform(gprs_scores)
    # Predict using scaled scores
    y_pred_sc = (gprs_scores_scaled > 0.5).astype(int).flatten()

    # y_pred_sc = (np.array(y_pred) > 0.5).astype(int)
    # Compute AUROCx
    auroc = roc_auc_score(y, y_pred_sc)
    # Compute R-squared
    r_squared = utils.evaluate_r2(y, y_pred_sc)
    # Compute accuracy
    accuracy = accuracy_score(y, y_pred_sc)
    print("AUROC:", auroc)
    print("R-squared:", r_squared)
    print("Accuracy:", accuracy)

    gprs_scores_reg = get_individual_BW_summed_GPRS_L2_REG(input_modified_geno=genotype, phenotype=pheno)
    gprs_scores_reg = np.array(gprs_scores_reg).reshape(-1, 1)
    gprs_scores_reg  = scaler.fit_transform(gprs_scores_reg)

    y_pred_reg = (gprs_scores_reg > 0.5).astype(int).flatten()
    # y_pred_reg = (np.array(y_pred_reg) > 0.5).astype(int)

    # Compute AUROC for L2 Regularized
    auroc = roc_auc_score(y, y_pred_reg)
    # Compute R-squared
    r_squared = utils.evaluate_r2(y, y_pred_reg)
    # Compute accuracy
    accuracy = accuracy_score(y, y_pred_reg)
    print('Regularized metrics: ')
    print('AUROC: Regularized:', auroc)
    print('R Squared Regularized ', r_squared)
    print('Accuracy Regularized ', accuracy)

    # ml_classifiers(genotype, y)

# sys.exit()
# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
# import numpy as np
#
# def ml_classifiers(genotype, y):
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(genotype, y, test_size=0.2, random_state=42)
#
#     # Standardize the data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     # Evaluate ElasticNet Regression
#     print('Evaluating ElasticNet Regression')
#     param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
#     grid_search = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5)  # Increase max_iter for convergence
#     grid_search.fit(X_train_scaled, y_train)
#     model = grid_search.best_estimator_
#     y_pred_reg = model.predict(X_test_scaled)
#     y_pred_reg_binary = (y_pred_reg > 0.5).astype(int)
#
#     auroc = roc_auc_score(y_test, y_pred_reg)
#     r_squared = utils.evaluate_r2(y_test, y_pred_reg)
#     accuracy = accuracy_score(y_test, y_pred_reg_binary)
#
#     print('Regularized metrics Elastic Net: ')
#     print('AUROC: Regularized :', auroc)
#     print('R Squared Regularized:', r_squared)
#     print('Accuracy Regularized:', accuracy)
#
#     # Evaluate XGBoost Classifier with cross-validation
#     print('Evaluating XGBoost Classifier with cross-validation')
#     xgb = XGBClassifier()
#     y_pred_xgb_proba_cv = cross_val_predict(xgb, genotype, y, cv=5, method='predict_proba')[:, 1]
#     y_pred_xgb_binary_cv = (y_pred_xgb_proba_cv > 0.5).astype(int)
#     y_pred_xgb_cv = cross_val_predict(xgb, genotype, y, cv=5, method='predict')
#
#     auroc_cv = roc_auc_score(y, y_pred_xgb_proba_cv)
#     accuracy_cv = accuracy_score(y, y_pred_xgb_binary_cv)
#     r_squared_cv = utils.evaluate_r2(y, y_pred_xgb_cv)
#     print('XGBoost Classifier metrics with cross-validation: ')
#     print('AUROC:', auroc_cv)
#     print('R Squared:', r_squared_cv)
#     print('Accuracy:', accuracy_cv)
#
#     # Evaluate Random Forest Classifier with cross-validation
#     print('Evaluating Random Forest Classifier with cross-validation')
#     rf = RandomForestClassifier()
#     y_pred_rf_proba_cv = cross_val_predict(rf, genotype, y, cv=5, method='predict_proba')[:, 1]
#     y_pred_rf_binary_cv = (y_pred_rf_proba_cv > 0.5).astype(int)
#     y_pred_rf_cv = cross_val_predict(rf, genotype, y, cv=5, method='predict')
#
#     auroc_cv = roc_auc_score(y, y_pred_rf_proba_cv)
#     accuracy_cv = accuracy_score(y, y_pred_rf_binary_cv)
#     r_squared_cv = utils.evaluate_r2(y, y_pred_rf_cv)
#     print('Random Forest Classifier metrics with cross-validation: ')
#     print('AUROC:', auroc_cv)
#     print('R Squared:', r_squared_cv)
#     print('Accuracy:', accuracy_cv)

# Sample call to the function (assuming you have genotype and y loaded):
# ml_classifiers(genotype, y)

def sum_stats_combined_bridge_weights(genotype_file_filtered, sum_stats, data_bridge_weights, ss_beta_column):
    print('Using bridge + summary statistics : ')
    ss_beta_column = ss_beta_column - 1 + 5
    # Load genotype data
    snp_data, phenotype = load_real_genotype_data_rename_cols(f'{genotype_file_filtered}')

    # Scale the genotype data while preserving the DataFrame structure
    scaler = StandardScaler()
    snp_data_scaled = pd.DataFrame(scaler.fit_transform(snp_data), columns=snp_data.columns)

    # Replace phenotype labels


    # Load bridge weights and summary statistics
    bridge_weights = data_bridge_weights
    sum_stats = pd.read_csv(sum_stats, sep="\t")

    modified_columns = {}
    # print(sum_stats)
    # Loop through the summary statistics
    if not sum_stats.empty:
        for index, row in sum_stats.iterrows():
            snp_id = row.iloc[1]
            beta = float(row.iloc[ss_beta_column])

            # Check if SNP ID exists in the genotype data and bridge weights
            if snp_id in snp_data.columns:
                if snp_id in bridge_weights['SNP_ID'].values:
                    bridge_weight = bridge_weights[bridge_weights['SNP_ID'] == snp_id].iloc[0, 1]
                    # print("Bridge weight:" , bridge_weight);
                    modified_columns[snp_id] = snp_data_scaled[snp_id] * (beta + bridge_weight)
                    # print(f'Modified {snp_id}: {modified_columns[snp_id].head(1)}')

    modified_data = pd.DataFrame(modified_columns)
    # print(modified_data.shape, '<- Modified data shape.')

    return modified_data, phenotype


