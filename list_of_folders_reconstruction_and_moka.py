import os
import sys

# Define SLURM template
slurm_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --cpus-per-task=22
#SBATCH --mem=128g
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davidenoma@gmail.com

module load cuda/12.1.1
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/global/software/cuda/cuda-12.1.1'

source ~/.bashrc
conda activate snakemake
module load R/4.2.0

python {script_path} {folder_name} {study_flag} {is_spectral_decorrelated} {do_reconstruction}
"""

def generate_and_submit_scripts(base_dir, folder_prefix, study_type, is_spectral_decorrelated,reconstruction_flag):
    # Determine the flag based on study type
    study_flag = "--quantitative" if study_type == "quantitative" else ""

    # Get all folders starting with the given prefix
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(f) and f.startswith(folder_prefix)]

    if not folders:
        print(f"No folders found with prefix '{folder_prefix}' in {base_dir}.")
        return

    # Get the script path relative to the utils folder
    script_path = os.path.join(os.path.dirname(__file__), "single_folder_reconstruction_and_moka.py")

    # Generate SLURM scripts and submit jobs
    for folder in folders:
        job_name = folder.rstrip("/")  # Remove trailing slash for job naming
        job_name = f"rbam_{job_name}"
        slurm_script = slurm_template.format(
            job_name=job_name,
            script_path=script_path,
            folder_name=os.path.abspath(folder),
            study_flag=study_flag,
            is_spectral_decorrelated=is_spectral_decorrelated,
            do_reconstruction = reconstruction_flag
        )

        # Save SLURM script
        slurm_file = f"{job_name}.slurm"
        with open(slurm_file, "w") as f:
            f.write(slurm_script)

        # Submit the job
        os.system(f"sbatch {slurm_file}")
        print(f"Submitted job for folder: {folder}")

if __name__ == "__main__":
    # Updated usage: an optional reconstruction flag can be provided.
    if len(sys.argv) < 5:
        print("Usage: python script.py <base_dir> <folder_prefix> <study_type> <is_spectral_decorrelated> [--reconstruction]")
        print("<study_type> should be 'quantitative' or 'case_control'")
        sys.exit(1)

    base_dir = sys.argv[1]
    folder_prefix = sys.argv[2]
    study_type = sys.argv[3]
    is_spectral_decorrelated = sys.argv[4]
    do_reconstruction = "--reconstruction" in sys.argv

    if study_type not in ["quantitative", "case_control"]:
        print("Invalid study type. Use 'quantitative' or 'case_control'.")
        sys.exit(1)

    generate_and_submit_scripts(base_dir, folder_prefix, study_type, is_spectral_decorrelated, do_reconstruction)