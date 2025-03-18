import os
import subprocess
import glob
import shutil

# Define constants and paths
BASE_DIR = "."  # Path to the parent directory containing simulated folders
PLINK_PATH = "plink"  # Update to the full path if PLINK is not in your $PATH
PYTHON_PATH = "python"  # Update to the full path if needed
GENO_UTILS_PATH = os.path.expanduser("~/Simulations_GWAS/utils/update_config_file.py")
SNPEMBED_PATH = os.path.expanduser("~/RL_GENO/AE_latent_space_embedding.py")
VAE_PATH = os.path.expanduser("~/RL_GENO/VAE_get_recon_cc_quant.py")
MERGE_WEIGHTS_PATH = os.path.expanduser("~/RL_GENO/utils/merge_enc_and_dec_weights.py")


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
    # Save the initial working directory
    initial_cwd = os.getcwd()

    # Folder and file prefixes
    geno_prefix = os.path.basename(folder)
    pruned_prefix = f"{geno_prefix}_prune"

    print(f"Processing folder: {folder}")

    try:
        # Step 1: Change to folder and prune SNPs
        os.chdir(folder)
        run_command([PLINK_PATH, "--bfile", geno_prefix, "--indep-pairwise", "50", "5", "0.2", "--out", geno_prefix])
        run_command([PLINK_PATH, "--bfile", geno_prefix, "--extract", f"{geno_prefix}.prune.in", "--make-bed", "--out", pruned_prefix])

        # Step 2: Convert pruned genotype data to raw format
        run_command([PLINK_PATH, "--bfile", pruned_prefix, "--recodeA", "--out", pruned_prefix])

        # Step 3: Latent space embedding and reconstruction
        run_command([PYTHON_PATH, VAE_PATH, f"{pruned_prefix}.raw", f"{pruned_prefix}.bim", "cc"])

        # Step 4: Merge encoder and decoder weights
        output_weights_folder = os.path.join( "output_weights", "hopt")
        os.makedirs(output_weights_folder, exist_ok=True)

        run_command(
            [
                PYTHON_PATH,
                MERGE_WEIGHTS_PATH,
                os.path.join(output_weights_folder, f"{pruned_prefix}_encoder_snp_and_weights.tsv"),
                os.path.join(output_weights_folder, f"{pruned_prefix}_decoder_snp_and_weights.tsv"),
                os.path.join(output_weights_folder, f"{pruned_prefix}_encoder_decoder_snp_and_weights.tsv"),
            ],

        )

        # Step 5: Clone `bas_pipeline` repository and update configuration file
        # if not os.path.exists(pipeline_path):
        # Forced cloning of `bas_pipeline`
        pipeline_path = os.path.join(folder, "bas_pipeline")
        if os.path.exists(pipeline_path):
            print(f"Removing existing directory: {pipeline_path}")
            shutil.rmtree(pipeline_path)

        run_command(["git", "clone", "https://github.com/davidenoma/bas_pipeline/"], cwd=folder)
        pipeline_path = os.path.join(folder, "bas_pipeline")

        # Copy genotype files to bas_pipeline/genotype_data/
        genotype_data_path = os.path.join(pipeline_path, "genotype_data")

        for ext in ["bed", "bim", "fam"]:
            src_file = os.path.join(".", f"{geno_prefix}.{ext}")
            dest_file = os.path.join(genotype_data_path, f"{geno_prefix}.{ext}")
            shutil.copy(src_file, dest_file)
        print(f"Copied genotype files to {genotype_data_path}")

        # Ensure all weight types are in the "weights/" folder inside "bas_pipeline"

        weight_types = [
            ("enc", f"{pruned_prefix}_encoder_snp_and_weights.tsv"),
            ("dec", f"{pruned_prefix}_decoder_snp_and_weights.tsv"),
            ("enc_dec", f"{pruned_prefix}_encoder_decoder_snp_and_weights.tsv"),
        ]

        for _, weight_file in weight_types:
            src_path = os.path.join(output_weights_folder, weight_file)
            dest_path = os.path.join(pipeline_path, 'weights', weight_file)
            # os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)

        os.chdir(pipeline_path)  # Move into "bas_pipeline"
        config_file = os.path.join( "config/config.yaml")
        template_file = os.path.join( "config/config-tmpl.yaml")
        shutil.copy(template_file, config_file)
        print(f"Copied template config file to {config_file}")



        # Step 6: Run Snakemake for each weight type
        for weight_type, weights_file in weight_types:
            run_command(
                [
                    PYTHON_PATH,
                    GENO_UTILS_PATH,
                    "--config_file",
                    "config/config.yaml",
                    "--updates",
                    f"weights_type={weight_type}",
                    f"weights_file=weights/{weights_file}",
                    f"genotype_prefix={geno_prefix}",
                ]
            )
            print("\n--- Config File Contents ---")
            with open(config_file, "r") as config:
                print(config.read())
            print("--- End of Config File ---\n")

            run_command(["snakemake", "-c", "22"])

            run_command(["snakemake", "-c", "1", "merge_skat_results"])

        # Step 7: Run Snakemake for unweighted SKAT
        run_command(["snakemake", "-c", "22", "unweighted_skat"])

        print(f"Completed processing for folder: {folder}")

    except Exception as e:
        print(f"Error processing folder {folder}: {e}")

    finally:
        # Return to the initial working directory
        os.chdir(initial_cwd)


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
