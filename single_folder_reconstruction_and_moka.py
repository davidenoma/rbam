import os
import subprocess
import shutil
import sys
import traceback
import argparse
import urllib.request
import zipfile

# Define constants and paths
PYTHON_PATH = "python3"  # Update to the full path if needed

GENO_UTILS_PATH = os.path.join(os.path.dirname(__file__), "utils/update_config_file.py")
AE_PATH = os.path.join(os.path.dirname(__file__), "runner/rbam_XAI_main.py")
VAE_PATH = os.path.join(os.path.dirname(__file__),"runner/rbam_main.py")
MERGE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "utils/merge_enc_and_dec_weights.py")
BAS_PIPELINE_PATH = os.path.expanduser("~/moka")  # Pre-cloned path of moka_pipeline


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

def process_folder(folder, is_binary=True,is_spectral_decorrelated=True,do_reconstruction=False):
    """
    Process a single folder.
    """
    # Save the initial working directory
    initial_cwd = os.getcwd()

    # Folder and file prefixes
    geno_prefix = os.path.basename(folder.strip("/"))
    pruned_prefix = f"{geno_prefix}_prune"

    phenotype_type = "cc" if is_binary else "quantitative"
    print(f"Processing folder: {folder} with phenotype_type={phenotype_type}")

    try:
        # Step 1: Change to folder and prune SNPs
        os.chdir(folder)
        run_command([PLINK_PATH, "--bfile", geno_prefix, "--indep-pairwise", "50", "5", "0.2", "--allow-no-sex","--out", geno_prefix])
        run_command([PLINK_PATH, "--bfile", geno_prefix, "--extract", f"{geno_prefix}.prune.in", "--make-bed", "--out",
                     pruned_prefix])

        # Step 2: Convert pruned genotype data to raw format
        run_command([PLINK_PATH, "--bfile", pruned_prefix, "--recodeA", "--out", pruned_prefix])
        raw_file = f"{pruned_prefix}.raw"

        if sys.platform == "darwin":
            # macOS requires an empty string for the backup extension
            replace_command = f"sed -i '' 's/NA/0/g' {raw_file}"
        else:
            replace_command = f"sed -i 's/NA/0/g' {raw_file}"

        subprocess.run(replace_command, shell=True, check=True)


        # Optional reconstruction and weight extraction block
        if do_reconstruction:
            run_command([PYTHON_PATH, VAE_PATH, f"{pruned_prefix}.raw", f"{pruned_prefix}.bim", phenotype_type])
            run_command([PYTHON_PATH, AE_PATH, f"{pruned_prefix}.raw", f"{pruned_prefix}.bim", phenotype_type])

        # Step 4: Merge encoder and decoder weights
        if is_binary:
            output_weights_folder = os.path.join(folder, "output_weights", "hopt")
        else:
            output_weights_folder = os.path.join(folder, "output_weights", "hopt_cc_com_or_quant")

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

        # Step 5: Add SHAP weights to the weight types
        shap_weights_file = os.path.join(
            folder,
            f"model_outputs/hopt_AE/shap_values_{pruned_prefix}.raw/{pruned_prefix}_merged_snp_and_weights.csv"
        )
        if not os.path.exists(shap_weights_file):
            raise FileNotFoundError(f"SHAP weights file not found at {shap_weights_file}")

        shap_dest_file = os.path.join(output_weights_folder, f"{pruned_prefix}_shap_weights.tsv")
        shutil.copy(shap_weights_file, shap_dest_file)
        print(f"Copied SHAP weights from {shap_weights_file} to {shap_dest_file}")

        # Step 6: Use pre-cloned `bas_pipeline`
        pipeline_path = os.path.join(folder, "bas_pipeline")
        #No need to remove existing directory, as we are copying from a pre-cloned version
        # if os.path.exists(pipeline_path):
        #     print(f"Removing existing directory: {pipeline_path}")
        #     shutil.rmtree(pipeline_path)

        print(f"Copying pre-cloned bas_pipeline from {BAS_PIPELINE_PATH} to {pipeline_path}")
        shutil.copytree(BAS_PIPELINE_PATH, pipeline_path,dirs_exist_ok=True)

        # Copy genotype files to bas_pipeline/genotype_data/
        genotype_data_path = os.path.join(pipeline_path, "genotype_data")
        os.makedirs(genotype_data_path, exist_ok=True)

        for ext in ["bed", "bim", "fam"]:
            src_file = os.path.join(folder, f"{geno_prefix}.{ext}")
            dest_file = os.path.join(genotype_data_path, f"{geno_prefix}.{ext}")
            shutil.copy(src_file, dest_file)
        print(f"Copied genotype files to {genotype_data_path}")
        if is_spectral_decorrelated:
        # Define weight types including SHAP weights
            weight_types = [
                ("enc_sd", f"{pruned_prefix}_encoder_snp_and_weights.tsv"),
                ("dec_sd", f"{pruned_prefix}_decoder_snp_and_weights.tsv"),
                ("enc_dec_sd", f"{pruned_prefix}_encoder_decoder_snp_and_weights.tsv"),
                ("shap_sd", f"{pruned_prefix}_shap_weights.tsv"),
            ]
        else:
            weight_types = [
                ("enc", f"{pruned_prefix}_encoder_snp_and_weights.tsv"),
                ("dec", f"{pruned_prefix}_decoder_snp_and_weights.tsv"),
                ("enc_dec", f"{pruned_prefix}_encoder_decoder_snp_and_weights.tsv"),
                ("shap", f"{pruned_prefix}_shap_weights.tsv"),
            ]

        # Copy weight files to bas_pipeline weights directory
        for weight_type, weight_file in weight_types:
            src_path = os.path.join(output_weights_folder, weight_file)
            dest_path = os.path.join(pipeline_path, "weights", weight_file)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)
            print(f"Copied {weight_type} weights to {dest_path}")

        # Step 7: Update configuration file and run Snakemake
        os.chdir(pipeline_path)
        config_file = os.path.join("config", "config.yaml")
        template_file = os.path.join("config", "config-tmpl.yaml")
        shutil.copy(template_file, config_file)
        print(f"Copied template config file to {config_file}")

        # Run Snakemake for each weight type
        # Run Snakemake for each weight type
        for weight_type, weights_file in weight_types:
            try:
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
                        f"is_binary={'TRUE' if is_binary else 'FALSE'}",
                        f"is_spectral_decorrelated={'TRUE' if is_spectral_decorrelated else 'FALSE'}",
                    ]
                )
                run_command(["snakemake", "-c", "22"])
                run_command(["snakemake", "-c", "1", "merge_moka_results"])
                run_command(["snakemake", "-c", "1", "manhattan_plots"])
            except Exception:
                print(f"An error occurred while processing weight type: {weight_type}")
                print(traceback.format_exc())

    finally:
        os.chdir(initial_cwd)

def get_plink_binary(provided_path=None):
    """
    Returns the path to the PLINK binary. If not provided and not found in PATH, downloads and unzips PLINK.
    """
    if provided_path:
        if os.path.isfile(provided_path) and os.access(provided_path, os.X_OK):
            return provided_path
        else:
            print(f"Provided PLINK path {provided_path} is not executable or does not exist. Trying PATH...")
    # Try to find plink in PATH
    plink_in_path = shutil.which("plink")
    if plink_in_path:
        return plink_in_path
    # Download PLINK
    print("PLINK not found. Downloading PLINK 1.9...")
    plink_url = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20210606.zip" if sys.platform.startswith("linux") else "https://s3.amazonaws.com/plink1-assets/plink_mac_20210606.zip"
    plink_zip = "plink_download.zip"
    plink_dir = os.path.join(os.path.dirname(__file__), "plink_bin")
    os.makedirs(plink_dir, exist_ok=True)
    plink_bin_path = os.path.join(plink_dir, "plink")
    try:
        urllib.request.urlretrieve(plink_url, plink_zip)
        with zipfile.ZipFile(plink_zip, 'r') as zip_ref:
            zip_ref.extractall(plink_dir)
        os.chmod(plink_bin_path, 0o755)
        os.remove(plink_zip)
        print(f"PLINK downloaded and extracted to {plink_bin_path}")
        return plink_bin_path
    except Exception as e:
        print("Failed to download or extract PLINK.")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single folder reconstruction and MOKA pipeline integration.")
    parser.add_argument("folder_name", help="Path to the genotype folder")
    parser.add_argument("--quantitative", action="store_true", help="Set for quantitative traits (default: binary/case-control)")
    parser.add_argument("--no-spectral", action="store_true", help="Disable spectral decorrelation")
    parser.add_argument("--reconstruction", action="store_true", help="Enable genotype reconstruction")
    parser.add_argument("--plink-path", type=str, default=None, help="Path to PLINK binary (default: auto-download if not found)")
    args = parser.parse_args()

    folder_name = args.folder_name
    is_binary = not args.quantitative
    is_spectral_decorrelated = not args.no_spectral
    do_reconstruction = args.reconstruction
    PLINK_PATH = get_plink_binary(args.plink_path)

    folder_path = os.path.abspath(folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        try:
            process_folder(folder_path, is_binary, is_spectral_decorrelated, do_reconstruction)
        except Exception as e:
            print("An error occurred:")
            print(traceback.format_exc())
    else:
        print(f"Error: Folder {folder_name} does not exist.")