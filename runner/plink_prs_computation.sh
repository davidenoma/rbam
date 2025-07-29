# Calculate Polygenic Risk Scores (PRS) using PLINK
# Usage: plink --bfile <genotype_file> --score <sumstats_file> <snp_col> <allele_col> <weight_col> header --out <output_prefix>

plink --bfile ${GENOTYPE_PREFIX} --score ${SUMSTATS_FILE} 2 6 8 header --out ${OUTPUT_PREFIX}

# Column specifications for --score command:
# 2: SNP ID column (variant identifier)
# 6: Effect allele column (allele for which effect size is reported)
# 8: Effect weight column (beta coefficients or effect sizes)
# header: indicates first row contains column names

# Example with placeholder variables:
# GENOTYPE_PREFIX="your_genotype_data"
# SUMSTATS_FILE="your_summary_statistics.tsv"
# OUTPUT_PREFIX="prs_results"

# This generates a .profile file containing individual PRS scores
# Output file: ${OUTPUT_PREFIX}.profile