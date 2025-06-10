import subprocess
import sys

import pandas as pd


def load_real_genotype_data_rename_cols(snp_data_loc):
    print('LOADING GENOTYPE', snp_data_loc)
    chunk_size = 10000
    # Columns to skip during reading
    columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX']

    # Initialize an empty DataFrame
    snp_data_ = pd.DataFrame()
    # Read data in chunks
    reader = pd.read_csv(snp_data_loc, sep=" ", usecols=lambda column: column not in columns_to_skip,
                         dtype='int8', chunksize=chunk_size)
    for chunk in reader:
        snp_data_ = pd.concat([snp_data_, chunk])
        # print("Chunk Number", i + 1)
    phenotype = snp_data_['PHENOTYPE']
    snp_data_ = snp_data_.drop(columns=['PHENOTYPE'])
    snp_data_.reset_index(inplace=True, drop=True)

    colnames = snp_data_.columns
    new_col_names = []
    for i in colnames:
        new_col_names.append(i[:-2])
    snp_data_.columns = new_col_names

    return snp_data_, phenotype


def bridge_weighted_real_genotype_data(snp_data_input, data_bridge_weights):
    weighted_columns = {}
    data_bridge_weights = pd.read_csv(data_bridge_weights)
    if not data_bridge_weights.empty:
        for index, row in data_bridge_weights.iterrows():
            snp_id = row.iloc[0]  # Assuming the first column contains SNP IDs
            bridge_weight = float(row.iloc[3])

            # Assuming the third column contains the multipliers
            if snp_id in snp_data_input.columns:
                weighted_columns[snp_id] = snp_data_input[snp_id] * bridge_weight

    # Create a file with modified columns to elimiate snps with zero weights
    modified_data_output = pd.DataFrame(weighted_columns)
    return modified_data_output

#
# input_genotype = sys.argv[1]
# bridge_weights = sys.argv[2]
#
# renamed_geno,phenotype = load_real_genotype_data_rename_cols(input_genotype)
#
# modified_geno = bridge_weighted_real_genotype_data(renamed_geno,bridge_weights)


if __name__ == "__main__":
    input_genotype = sys.argv[1]
    bridge_weights = sys.argv[2]
    bim_file = sys.argv[3]

    renamed_geno, phenotype = load_real_genotype_data_rename_cols(input_genotype)

    modified_geno = bridge_weighted_real_genotype_data(renamed_geno, bridge_weights)
    # Save modified genotype to a CSV file
    modified_geno.to_csv("modified_geno.csv", index=False,sep=" ")
    # Execute hyp_optimize_cc_VAE.py as a subprocess
    subprocess.call(["python", "hyp_optimize_cc_VAE.py",  "modified_geno.csv", bim_file])