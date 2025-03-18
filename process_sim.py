import os
import subprocess
import glob

# Define constants and paths
BASE_DIR = "/path/to/simulated/folders"  # Path to the parent directory containing simulated folders
PLINK_PATH = "plink"  # Update to the full path if PLINK is not in your $PATH
PYTHON_PATH = "python"  # Update to the full path if needed
GENO_UTILS_PATH = "~/Simulations_Genomics/utils/update_config_file.py"
SNPEMBED_PATH = "~/RL_GENO/AE_latent_space_embedding.py"
VAE_PATH = "~/RL_GENO/VAE_get_recon_cc_quant.py"
MERGE_WEIGHTS_PATH = "~/RLGEN/utils/merge_enc_and_dec_weights.py"

def run_command(command, cwd=None):
    """
    Run a shell command.
    """
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        raise e

def process_folder(folder):
    """
    Process a single folder.
    """
    # Folder and file prefixes
    geno_prefix = os.path.basename(folder)
    pruned_prefix = f"{geno_prefix}_prune"

    print(f"Processing folder: {folder}")

    # Step 1: Change to folder and prune SNPs
    os.chdir(folder)
    run_command([PLINK_PATH, "--bfile", geno_prefix, "--indep-pairwise", "50", "5", "0.2", "--out", geno_prefix])
    run_command([PLINK_PATH, "--bfile", geno_prefix, "--extract", f"{geno_prefix}.prune.in", "--make-bed", "--out", pruned_prefix])

    # Step 2: Convert pruned genotype data to raw format
    run_command([PLINK_PATH, "--bfile", pruned_prefix, "--recodeA", "--out", pruned_prefix])

    # Step 3: Latent space embedding and reconstruction
    # run_command([PYTHON_PATH, SNPEMBED_PATH, f"{pruned_prefix}.raw", f"{pruned_prefix}.bim"])
    run_command([PYTHON_PATH, VAE_PATH, f"{pruned_prefix}.raw", f"{pruned_prefix}.bim", "cc"])

    # Step 4: Merge encoder and decoder weights
    run_command([
        PYTHON_PATH,
        MERGE_WEIGHTS_PATH,
        f"output_weights/hopt/{pruned_prefix}_encoder_snp_and_weights.tsv",
        f"output_weights/hopt/{pruned_prefix}_decoder_snp_and_weights.tsv",
        f"output_weights/hopt/{pruned_prefix}_encoder_decoder_snp_and_weights.tsv"
    ])

    # Step 5: Clone `bas_pipeline` repository and update configuration file
    if not os.path.exists("bas_pipeline"):
        run_command(["git", "clone", "https://github.com/davidenoma/bas_pipeline/"])
    run_command(["cp", "output_weights/hopt/*", "bas_pipeline/weights"])
    os.chdir("bas_pipeline")

    # Define weight types
    weight_types = [
        ("enc", f"weights/{pruned_prefix}_encoder_snp_and_weights.tsv"),
        ("dec", f"weights/{pruned_prefix}_decoder_snp_and_weights.tsv"),
        ("enc_dec", f"weights/{pruned_prefix}_encoder_decoder_snp_and_weights.tsv"),
    ]

    # Step 6: Run Snakemake for each weight type
    for weight_type, weights_file in weight_types:
        run_command([PYTHON_PATH, GENO_UTILS_PATH, "update", "--weights-type", weight_type, "--weights-file", weights_file, "--genotype-prefix", geno_prefix])
        run_command(["snakemake", "-c", "22"])
        run_command(["snakemake", "-c", "1", "merge_skat_results"])

    # Step 7: Run Snakemake for unweighted SKAT
    run_command(["snakemake", "-c", "22", "unweighted_skat"])

    print(f"Completed processing for folder: {folder}")

def main():
    # Get list of folders in the base directory
    folders = glob.glob(os.path.join(BASE_DIR, "*"))
    folders = [folder for folder in folders if os.path.isdir(folder)]

    # Process each folder
    for folder in folders:
        try:
            process_folder(folder)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

if __name__ == "__main__":
    main()
