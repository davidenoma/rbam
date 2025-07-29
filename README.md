# RBAM: Representation Learning-Based Association Mapping
<img width="600" height="600" alt="rbam_methods" src="https://github.com/user-attachments/assets/8a5148fa-4da2-4904-a78e-45ff9f7e68fc" />

## Overview
RBAM (Representation Learning-Based Association Mapping) is a framework that leverages variational autoencoders (VAE) to learn latent genotype representations, facilitating representation-informed association mapping and phenotype classification. This approach addresses limitations of traditional GWAS by accounting for polygenicity, epistatic interactions, and linkage disequilibrium.

## Abstract

Genome-wide association studies (GWAS) have provided key insights into the genetic architecture of complex diseases. However, traditional approaches often fall short in accounting for polygenicity, epistatic interactions, and linkage diequilibrium, leading to reduced power. We present Representation Learning-Based Association Mapping (RBAM), a framework that leverages variational autoencoders (VAE) to learn latent genotype representations, facilitating representation-informed association mapping and phenotype classification. Using 17 complex disorders and traits spanning brain disorders, immunological traits, cancers, cardiometabolic, and quantitative phenotypes, GWAS samples from the UK Biobank, dbGaP, and WTCCC, RBAM demonstrates superior power to detect validated gene-disease associations, particularly validated via DisGeNET disease-specific databases. Simulation studies confirm that RBAM maintains a controlled Type I error rate. Functional analysis reveals overlapping genetic pathways among different diseases. Overall, RBAM provides a robust and interpretable framework, bridging the gap between unsupervised representation learning and association mapping.

**Keywords:** Representation learning, Variational auto-encoder, Genome-wide association study, Kernel association testing, Complex traits, Polygenic risk prediction

## Prerequisites

### Software Requirements
- Python 3.8+
- TensorFlow 2.x
- PLINK 1.9 or 2.0
- Snakemake
- CUDA-compatible GPU (recommended)

### Python Dependencies
```bash
pip install tensorflow numpy pandas scikit-learn hyperopt xgboost shap matplotlib seaborn
```

### External Tools
- [PLINK](https://www.cog-genomics.org/plink/) - For genotype data processing
- [MOKA Pipeline](https://github.com/davidenoma/moka) - For association mapping

## Installation

1. Clone the RBAM repository:
```bash
git clone https://github.com/your-username/rbam.git
cd rbam
```

2. Clone the MOKA pipeline (required for association mapping):
```bash
git clone https://github.com/davidenoma/moka.git ~/moka
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Your genotype data should be in PLINK binary format (.bed, .bim, .fam files). The framework supports both:
- **Case-control studies** (binary phenotypes)
- **Quantitative traits** (continuous phenotypes)

## Usage

### 1. Genotype Reconstruction and Weight Extraction

#### For Variational Autoencoder (VAE)
```bash
python runner/rbam_main.py <genotype_file.raw> <genotype_file.bim> <phenotype_type>
```

**Parameters:**
- `<genotype_file.raw>`: Path to PLINK raw format genotype file
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
- Cross-validation results

### 3. Single Folder Processing with MOKA Pipeline

For comprehensive analysis including association mapping:

```bash
python single_folder_reconstruction_and_moka.py <folder_path> [options]
```

**Options:**
- `--quantitative`: For quantitative traits (default: binary/case-control)
- `--no-spectral`: Disable spectral decorrelation
- `--reconstruction`: Enable genotype reconstruction

**Example:**
```bash
# Binary trait analysis
python single_folder_reconstruction_and_moka.py /path/to/genotype_folder

# Quantitative trait analysis
python single_folder_reconstruction_and_moka.py /path/to/genotype_folder --quantitative

# With genotype reconstruction
python single_folder_reconstruction_and_moka.py /path/to/genotype_folder --reconstruction
```

**Workflow:**
1. **Data Preprocessing:**
   - LD pruning using PLINK (`--indep-pairwise 50 5 0.2`)
   - Conversion to raw format
   - Missing value imputation

2. **Model Training:**
   - VAE and Autoencoder training
   - Weight extraction (encoder, decoder, combined)
   - SHAP analysis for feature importance

3. **Association Mapping:**
   - Integration with MOKA pipeline
   - Multiple weight types analysis:
     - `enc`: Encoder weights
     - `dec`: Decoder weights  
     - `enc_dec`: Combined weights
     - `shap`: SHAP-based weights
   - Spectral decorrelation options (`_sd` suffix)

4. **Results Generation:**
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

```bibtex
@article{rbam2024,
  title={Representation learning-based genome-wide association mapping discovers genes underlying complex traits},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

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
- Contact: [your-email]

## Acknowledgments

- UK Biobank, dbGaP, and WTCCC for providing genetic data
- DisGeNET for disease-gene association validation
- The MOKA pipeline for association mapping framework
