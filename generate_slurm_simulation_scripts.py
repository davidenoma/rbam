import os

# Function to run shell commands
def run_command(command):
    print(f"Running command: {' '.join(command)}")
    os.system(' '.join(command))


# Function to generate SLURM job script for phenotype simulation
def generate_simulation_script(model, heritability, num_causal, interaction_percent, output_dir, no_interaction=False, phenotype_type="quantitative"):
    job_name = f"h2_{heritability}_{model}_{num_causal}_{interaction_percent}"
    if no_interaction:
        job_name += "_no_interaction"
    script_path = os.path.join(output_dir, f"{job_name}.sh")

    # Conditional flags
    binary_flag = "--binary" if phenotype_type == "cc" else ""
    interaction_flag = "--no_interaction" if no_interaction else ""

    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=5:00:00

# Simulate phenotypes
python ~/Simulations_GWAS/simulate_phenotypes_with_epistatic_snp_snp_new.py \\
    --plink_prefix uk_bk_whb_for_sim_5000_QC \\
    --num_causal {num_causal} \\
    --interaction_percent {interaction_percent} \\
    --model_type {model} \\
    --heritability {heritability} \\
    --gene_region_file gene_regions.csv \\
    {binary_flag} \\
    {interaction_flag}
""")
    print(f"Generated simulation script: {script_path}")


# Function to generate SNP pruning and latent space embedding scripts
def generate_pruning_script(model, heritability, num_causal, interaction_percent, output_dir, no_interaction=False, phenotype_type="quantitative"):
    job_name = f"prune_h2_{heritability}_{model}_{num_causal}_{interaction_percent}"
    if no_interaction:
        job_name += "_no_interaction"
    geno_prefix = f"simulated_phenotype_h2_{heritability}_{model}_{num_causal}_int_{interaction_percent}"
    if no_interaction:
        geno_prefix += "_no_interaction"
    pruned_prefix = f"{geno_prefix}_prune"
    script_path = os.path.join(output_dir, f"{job_name}.sh")

    binary_flag = "cc" if phenotype_type == "cc" else "quantitative"

    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davidenoma@gmail.com

module load cuda/12.1.1
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/global/software/cuda/cuda-12.1.1'

# Change directory to the genotype file folder
cd {geno_prefix}

# PLINK SNP pruning
plink --bfile {geno_prefix} --indep-pairwise 50 5 0.2 --out {geno_prefix}
plink --bfile {geno_prefix} --extract {geno_prefix}.prune.in --make-bed --out {pruned_prefix}

# Convert to raw format
plink --bfile {pruned_prefix} --recodeA --out {pruned_prefix}

# Run latent space embedding and reconstruction
python ~/RL_GENO/AE_latent_space_embedding.py {pruned_prefix}.raw {pruned_prefix}.bim {binary_flag}

python ~/RL_GENO/VAE_get_recon_cc_quant.py {pruned_prefix}.raw {pruned_prefix}.bim {binary_flag}
""")
    print(f"Generated pruning script: {script_path}")


# Function to generate SKAT unweighted SLURM script
def generate_skat_script(model, heritability, num_causal, interaction_percent, output_dir, no_interaction=False, phenotype_type="quantitative"):
    job_name = f"skat_unweighted_{heritability}_{model}_{num_causal}_{interaction_percent}"
    if no_interaction:
        job_name += "_no_interaction"
    geno_prefix = f"simulated_phenotype_h2_{heritability}_{model}_{num_causal}_int_{interaction_percent}"
    if no_interaction:
        geno_prefix += "_no_interaction"
    script_path = os.path.join(output_dir, f"{job_name}.sh")

    binary_flag = "TRUE" if phenotype_type == "cc" else "FALSE"

    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=72G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=davidenoma@gmail.com
#SBATCH --array=1-22
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019,theia,bigmem

module load R/4.2.0

# Define the chromosome range for all chromosomes
chromosomes=("chr1" "chr2" "chr3" "chr4" "chr5" "chr6" "chr7" "chr8" "chr9" "chr10" "chr11" "chr12" \\
             "chr13" "chr14" "chr15" "chr16" "chr17" "chr18" "chr19" "chr20" "chr21" "chr22")

chromosome=${{chromosomes[$SLURM_ARRAY_TASK_ID - 1]}}

# Run SKAT unweighted R script
Rscript ~/bas_pipeline/scripts/skat_unweighted.R \\
    "{geno_prefix}" gene_regions.csv \\
    "{geno_prefix}/" "$chromosome" "skat_unweighted" \\
    "{binary_flag}"
""")
    print(f"Generated SKAT unweighted script: {script_path}")


# Main function to generate all scripts
def main():
    output_dir = "slurm_jobs/h280"
    os.makedirs(output_dir, exist_ok=True)

    # User-defined inputs
    heritability = 0.8  # Example heritability
    num_causal = 100  # Number of causal SNPs
    interaction_percent = 100.0  # Percentage of interactions
    models = ['none']  # Interaction models (including no interaction)
    phenotype_type = "quantitative"  # Use "cc" for binary or "quantitative" for continuous phenotypes
    # Generate scripts for each model with and without interactions
    for model in models:
        if model == "none":
            generate_simulation_script("none", heritability, num_causal, 0.0, output_dir, no_interaction=True, phenotype_type=phenotype_type)
            generate_pruning_script("none", heritability, num_causal, 0.0, output_dir, no_interaction=True, phenotype_type=phenotype_type)
            generate_skat_script("none", heritability, num_causal, 0.0, output_dir, no_interaction=True, phenotype_type=phenotype_type)
        else:
            generate_simulation_script(model, heritability, num_causal, interaction_percent, output_dir, phenotype_type=phenotype_type)
            generate_pruning_script(model, heritability, num_causal, interaction_percent, output_dir, phenotype_type=phenotype_type)
            generate_skat_script(model, heritability, num_causal, interaction_percent, output_dir, phenotype_type=phenotype_type)

    print(f"All scripts generated in {output_dir}.")


if __name__ == "__main__":
    main()
