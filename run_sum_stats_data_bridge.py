from bridge_target_genotype_utils import sum_stats_bridge_weights, using_only_sum_statistics
from bridge_target_genotype_utils2 import intersect_sum_stats_and_bim_file, extract_snps_and_create_genotype


# SCHIZOPHRENIA
# sum_stats_files = [
#     'top01_percent_PGS002785_hmPOS_GRCh38.txt',
#     'top1_percent_PGS002785_hmPOS_GRCh38.txt',
#     'top5_percent_PGS002785_hmPOS_GRCh38.txt',
#     'top10_percent_PGS002785_hmPOS_GRCh38.txt',
#     'top50_percent_PGS002785_hmPOS_GRCh38.txt',
#     '100_percent_PGS002785_hmPOS_GRCh38.txt'
#
# ]
# #ALZHEIMER'S
# sum_stats_files = [
#     'top01_percent_PGS002753.txt',
#     'top1_percent_PGS002753.txt',
#     'top5_percent_PGS002753.txt',
#     'top10_percent_PGS002753.txt',
#     'top50_percent_PGS002753.txt',
#     'PGS002753.csv'
# ]
#
#Autism
sum_stats_files = ['autism_beta3.txt']
sum_stats_files = ['sum_stats/' + file for file in sum_stats_files]
#
# print(sum_stats_files)


genotype_folder = 'genotype_data/'
genotype_prefix = 'merged_Ilmn_ASD_fma2_prune'
# genotype_prefix = 'lift_geno_EUR_final_fm_a2'
# genotype_prefix = 'alz_168'
# bridge_weights = 'bridge_weights/merged_ALz_eqtl_weights.csv'

# bridge_weights = 'bridge_weights/phast_alz_219_combined.csv'
# bridge_weights = 'bridge_weights/alz_219_phylo_combined.csv'
# bridge_weights = 'bridge_weights/combined_phylo_ng.csv'
# bridge_weights = 'bridge_weights/combined_inv_phylo.csv'

# bridge_weights = 'bridge_weights/weights_EUR_eqtl.csv'
# bridge_weights = 'bridge_weights/combined_phast_scaling_2.csv'
# bridge_weights = 'bridge_weights/combined_inv_phylo.csv'
# bridge_weights = 'bridge_weights/combined_inv_alz_168.csv'
# bridge_weights = 'bridge_weights/combined_alz_168_phast.csv'
# bridge_weights = 'bridge_weights/combined_PHYLO.csv'

filtered_genotypes = []
filtered_sum_stats = []
# Iterate over each sum stats file
for full_sum_stats_file in sum_stats_files:
    # Extracting the filtered sum stats file name based on the number of snps in sum statistics and the
    #input_modified_geno bim file from the genotype
    filtered_sum_stats_file = intersect_sum_stats_and_bim_file(genotype_folder, genotype_prefix, full_sum_stats_file)
    # filtered_sum_stats_file = intersect_sum_stats_weights_and_bim_file(genotype_folder, genotype_prefix, full_sum_stats_file,bridge_weights)
    filtered_sum_stats.append(filtered_sum_stats_file)
    geno_type_filt = extract_snps_and_create_genotype(genotype_folder, genotype_prefix, filtered_sum_stats_file)
    filtered_genotypes.append(geno_type_filt)
# WEIGHT AND GENOTYPE


# print(filtered_genotypes,filtered_sum_stats_file)
for i in range(len(filtered_genotypes)):
    print("\n")
    print('______________________ START _____________________________________________')
    # sum_stats_bridge_weights(filtered_genotypes[i], filtered_sum_stats[i], bridge_weights,ss_beta_column=7)

    using_only_sum_statistics(filtered_genotypes[i], filtered_sum_stats[i],ss_beta_column=3)
    print('_______________________ END ____________________________________________')
    print("\n")
