# RBAM: Representation Learning-Based Genome-wide Association Mapping

 <img width="600" height="600" alt="rbam_method,rbam_logo" src="https://github.com/user-attachments/assets/150c58a8-8e1e-4824-bc44-11efbd2cd2c7" />



## Overview
RBAM  is a framework that leverages variational autoencoders (VAE) to learn latent genotype representations, facilitating representation-informed association mapping and phenotype classification. This approach addresses limitations of traditional GWAS by accounting for polygenicity, epistatic interactions, and linkage disequilibrium.


## Abstract

Genome-wide association studies (GWAS) have provided key insights into the genetic architecture of complex diseases. However, traditional approaches often fall short in accounting for polygenicity, epistatic interactions, and linkage diequilibrium, leading to reduced power. We present Representation Learning-Based Association Mapping (RBAM), a framework that leverages variational autoencoders (VAE) to learn latent genotype representations, facilitating representation-informed association mapping and phenotype classification. Using 17 complex disorders and traits spanning brain disorders, immunological traits, cancers, cardiometabolic, and quantitative phenotypes, GWAS samples from the UK Biobank, dbGaP, and WTCCC, RBAM demonstrates superior power to detect validated gene-disease associations, particularly validated via DisGeNET disease-specific databases. Simulation studies confirm that RBAM maintains a controlled Type I error rate. Functional analysis reveals overlapping genetic pathways among different diseases. Overall, RBAM provides a robust and interpretable framework, bridging the gap between unsupervised representation learning and association mapping.

**Keywords:** Representation learning, Variational auto-encoder, Genome-wide association study, Kernel association testing, Complex traits, Polygenic risk prediction
<div align="center">
    <img width="600" height="600" alt="rbam_final" src="https://github.com/user-attachments/assets/388e92ba-cfd5-42de-aeff-2ac9632b094f" />
</div>

## Installation

1. Clone the RBAM repository:
```bash
git clone https://github.com/davidenoma/rbam.git
```

2. Clone the MOKA pipeline (required for association mapping):
```bash
git clone https://github.com/davidenoma/moka.git  ~/moka
cd ~/moka
```

3. (Recommended) Create a new Python 3.9 environment:

#### Using Conda
```bash
conda create -n rbam_env python=3.9
conda activate rbam_env
```

#### Using venv (pip)
```bash
python3.9 -m venv rbam_env
source rbam_env/bin/activate
```

4. Install Python dependencies:

#### Using Conda
```bash
conda install --file requirements.txt
```

#### Using pip
```bash
pip install -r requirements.txt
```

### External Tools
- [PLINK](https://www.cog-genomics.org/plink/) - For genotype data processing
- [MOKA Pipeline](https://github.com/davidenoma/moka) - For association mapping

```commandline
export PATH=$PATH:/path/to/plink
```
## Data Input

Your input genotype data must be in PLINK binary format (.bed, .bim, .fam files). 

The framework supports both:
- **Case-control studies** (binary phenotypes)
- **Quantitative traits** (continuous phenotypes)

## Usage with test data provided

### 1. Genotype Reconstruction and Weight Extraction

#### For Variational Autoencoder (VAE)
```bash
python runner/rbam_main.py <genotype_file.raw> <genotype_file.bim> <phenotype_type>
```

**Parameters:**
- `<genotype_file.raw>`: Path to PLINK raw format genotype file
```bash
plink --recode A --bfile genotype_file --out genotype_file
```
- `<genotype_file.bim>`: Path to corresponding BIM file
- `<phenotype_type>`: Either `"cc"` (case-control) or `"quantitative"`

**Example:**
```bash
python runner/rbam_main.py data/diabetes.raw data/diabetes.bim cc
```

**Output:**
- Trained VAE model saved in `model/` or `model_cc_com_qt/`
- Reconstruction metrics (MSE, R², RMSE, Pearson correlation)
- Encoder and decoder weights extracted to `output_weights/`

#### For Vanilla Autoencoder with XAI
```bash
python runner/rbam_XAI_main.py <genotype_file.raw> <genotype_file.bim> <phenotype_type>
```

**Features:**
- Hyperparameter optimization using Hyperopt
- SHAP (SHapley Additive exPlanations) for explainable AI
- Feature importance extraction
- Reconstruction quality metrics

**Output:**
- SHAP values for feature importance: `model_outputs/hopt_AE/shap_values_*/`
- Merged SNP weights: `*_merged_snp_and_weights.csv`
- Visualization plots: SHAP bar plots

### 2. Latent Space Classification

```bash
python runner/latent_space_predictor.py <genotype_file.raw>
```

**Classifiers Implemented:**
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network (TensorFlow)

**Features:**
- Automated hyperparameter tuning
- Cross-validation
- Class imbalance handling
- Multiple performance metrics (Accuracy, AUC, R²)

**Output:**
- Classification results: `model_outputs/rbam_classifier/`
- Performance metrics for each classifier


### 3. Single Folder Processing with MOKA Pipeline

For comprehensive analysis including association mapping:

```bash
python single_folder_reconstruction_and_moka.py <folder_path> [options]
```

**Options:**
- `--quantitative`: For quantitative traits (default: binary/case-control)
- `--reconstruction`: Enable genotype reconstruction
- `--plink-path <path>`: Specify the path to the PLINK binary (default: `plink` in PATH)

**Example:**
```bash
# Binary trait analysis with default PLINK
python single_folder_reconstruction_and_moka.py /path/to/genotype_folder

# Quantitative trait analysis
python single_folder_reconstruction_and_moka.py /path/to/genotype_folder --quantitative

# With genotype reconstruction
python single_folder_reconstruction_and_moka.py /path/to/genotype_folder --reconstruction

# Specify custom PLINK binary location
python single_folder_reconstruction_and_moka.py /path/to/genotype_folder --plink-path /usr/local/bin/plink
```

### IMPORTANT: Folder and Genotype File Naming

- The folder name **must match** the genotype file prefix. For example, if your genotype files are named `test_geno.bed`, `test_geno.bim`, and `test_geno.fam`, they must be placed in a folder named `test_geno/`.
- Example structure:
  ```
  test_geno/
    ├── test_geno.bed
    ├── test_geno.bim
    └── test_geno.fam
  ```

### Automatic PLINK Download
- If you do **not** provide `--plink-path` and PLINK is not found in your system PATH, the pipeline will automatically download and unzip PLINK for you (Linux/macOS supported).
- You can still specify a custom PLINK binary using `--plink-path` if needed.

### Example: Using Provided test_geno Data

```bash
# Run the pipeline on the provided test_geno example (binary trait)
python single_folder_reconstruction_and_moka.py test_geno

# For quantitative trait
python single_folder_reconstruction_and_moka.py test_geno --quantitative

# With genotype reconstruction
python single_folder_reconstruction_and_moka.py test_geno --reconstruction

# Specify custom PLINK binary location
python single_folder_reconstruction_and_moka.py test_geno --plink-path /usr/local/bin/plink
```

**Note:** If you do not specify `--plink-path` and PLINK is not installed, the script will automatically download and use PLINK.

**Workflow:**
1. **Data Preprocessing:**
    - LD pruning using PLINK (`--indep-pairwise 50 5 0.2`)
   - Genotype folder must bear the same name as the genotype ( bed, bim and fam ) files e.g.
     - test_geno/
        - test_geno.bim
        - test_geno.fam
        - test_geno.bed
   - Conversion to raw format
   - Missing value imputation

3. **Model Training:**
   - VAE and Autoencoder training
   - Weight extraction (encoder, decoder, combined)
   - SHAP analysis for feature importance

4. **Association Mapping:**
   - Integration with MOKA pipeline
   - Multiple weight types analysis:
     - `enc`: Encoder weights
     - `dec`: Decoder weights  
     - `enc_dec`: Combined weights
     - `shap`: SHAP-based weights

5. **Results Generation:**
   - GWAS results for each weight type
   - Manhattan plots
   - Merged association results

## Output Structure

```
├── model/                          # Trained models (case-control)
├── model_cc_com_qt/               # Trained models (quantitative)
├── model_outputs/                 # Analysis results
│   ├── hopt_AE/                   # Autoencoder results
│   │   └── shap_values_*/         # SHAP analysis
│   └── rbam_classifier/           # Classification results
├── output_weights/                # Extracted weights
│   ├── hopt/                      # Binary trait weights
│   └── hopt_cc_com_or_quant/     # Quantitative trait weights
└── bas_pipeline/                  # MOKA pipeline results
    ├── results/                   # Association results
    └── plots/                     # Manhattan plots
```

## Key Features

### 1. Representation Learning
- **Variational Autoencoders (VAE)**: Learn latent representations accounting for genetic architecture
- **Hyperparameter Optimization**: Automated tuning using Hyperopt
- **Cross-validation**: Robust model evaluation

### 2. Explainable AI
- **SHAP Analysis**: Feature importance for individual SNPs
- **Multiple Weight Types**: Encoder, decoder, and combined weights
- **Visualization**: Bar plots and importance rankings

### 3. Classification Framework
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost, Neural Networks
- **Class Imbalance Handling**: Balanced class weights and downsampling
- **Performance Metrics**: Accuracy, AUC, R² scores

### 4. Association Mapping
- **Integration with MOKA**: Seamless pipeline for GWAS
- **Multiple Weight Strategies**: Different biological interpretations
- **Spectral Decorrelation**: Optional preprocessing for population structure

## Performance Metrics

The framework provides comprehensive evaluation:

- **Reconstruction Quality**: MSE, RMSE, R², Adjusted R², Pearson correlation
- **Classification Performance**: Accuracy, AUC, cross-validation scores
- **Association Power**: Manhattan plots, genomic inflation factors

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Reduce batch size or use CPU-only mode
2. **PLINK Errors**: Ensure PLINK is in PATH and data format is correct
3. **Missing Dependencies**: Install all required packages and external tools

### Memory Optimization
- Use smaller batch sizes for large datasets
- Enable TensorFlow memory growth: `tf.config.experimental.set_memory_growth()`
- Consider data chunking for very large genotype files

## Citation

If you use RBAM in your research, please cite: 
Representation learning-based genome-wide association mapping discovers genes underlying complex traits
https://doi.org/10.21203/rs.3.rs-7624342/v1  


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Open an issue on GitHub
- Contact: david.enoma@ucalgary.ca or quan.long@ucalgary.ca
## Acknowledgments

- UK Biobank, dbGaP, and WTCCC for providing genetic data
- DisGeNET for disease-gene association validation
- The MOKA pipeline for association mapping framework
