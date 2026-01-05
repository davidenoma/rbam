# Quick Start Guide - SSVAE & Joint VAE

## Installation & Setup

```bash
# Navigate to project
cd /Users/davidenoma/PycharmProjects/rbam

# Verify Python environment
python --version  # Should be 3.9+
```

## Quick Test (5 minutes)

```bash
# Test SSVAE training
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw \
  --model_type ssvae \
  --epochs 50 \
  --latent_dim 64 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --output_dir ./model_outputs/test_ssvae

# Expected output:
# Epoch 1/50: loss=0.956, val_loss=0.737
# Epoch 2/50: loss=0.747, val_loss=0.757
# ...
# Epoch N/50: Stopped - No improvement for 10 epochs
# Saved learning curves: ./model_outputs/test_ssvae/ssvae_learning_curves.png
# SSVAE Evaluation Metrics:
#   Accuracy: 0.498
#   AUC-ROC: 0.519
#   F1 Score: 0.665
#   AUC-PR: 0.518
```

## Full Training Commands

### Option 1: SSVAE Only
```bash
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw \
  --model_type ssvae \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --latent_dim 128 \
  --alpha 1.0 --beta 1.0 --gamma 1.0 \
  --output_dir ./model_outputs/ssvae_full
```

### Option 2: Joint VAE+Classifier Only
```bash
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw \
  --model_type joint \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --latent_dim 128 \
  --alpha 1.0 --beta 1.0 --gamma 1.0 \
  --output_dir ./model_outputs/joint_full
```

### Option 3: Train Both Models
```bash
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw \
  --model_type both \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --latent_dim 128 \
  --alpha 1.0 --beta 1.0 --gamma 1.0 \
  --output_dir ./model_outputs/both
```

## Output Files

After training, check these files:

```
model_outputs/
├── ssvae_model.keras              ← Trained SSVAE model
├── ssvae_learning_curves.png       ← 4-panel loss visualization
├── ssvae_metrics.json              ← Evaluation results
├── joint_model.keras               ← Trained Joint model
├── joint_learning_curves.png       ← 4-panel loss visualization
└── joint_metrics.json              ← Evaluation results
```

## Understanding the Output

### Learning Curves (PNG)
Shows 4 subplots:
1. **Total Loss** - Should decrease smoothly
2. **Reconstruction Loss** - How well inputs are reconstructed
3. **Classification Loss** - How well phenotype is predicted
4. **Accuracy** - Classification accuracy (higher is better)

### Metrics (JSON)
```json
{
  "accuracy": 0.498,           ← Overall accuracy
  "balanced_accuracy": 0.500,  ← Per-class accuracy (better for imbalanced)
  "auc": 0.519,                ← Area under ROC curve (0.5=random, 1.0=perfect)
  "f1": 0.665,                 ← Harmonic mean of precision/recall
  "auprc": 0.518               ← Area under precision-recall curve
}
```

## Interpreting Results

### Good Results ✅
- **Accuracy**: > 0.55
- **Balanced Accuracy**: > 0.55
- **AUC-ROC**: > 0.60
- **F1 Score**: > 0.60
- Training loss decreases smoothly
- No NaN values appear

### Warning Signs ⚠️
- **Accuracy**: < 0.55 (close to random)
- **Loss becoming negative**: Indicates divergence
- **NaN values**: Training automatically stops
- Training stops early (< 20 epochs): May need more data or different hyperparameters

## Adjusting Hyperparameters

If results are suboptimal:

### For High Loss
```bash
--learning_rate 0.0001   # Reduce learning rate
--batch_size 16          # Smaller batches
--beta 0.5               # Reduce KL weight if diverging
```

### For Poor Classification
```bash
--gamma 2.0              # Increase classification weight
--epochs 200             # More training epochs
--alpha 0.5              # Reduce reconstruction weight
```

### For Slow Training
```bash
--batch_size 64          # Larger batches
--learning_rate 0.01     # Increase learning rate (carefully)
```

## Data Requirements

- **Format**: PLINK .raw format (space-separated)
- **Columns**: FID, IID, PAT, MAT, SEX, PHENOTYPE, [SNP1, SNP2, ...]
- **PHENOTYPE**: 1 = control, 2 = case
- **Minimum samples**: 100+ (more is better)
- **Minimum SNPs**: 100+

## Troubleshooting

### Error: "Cannot find model at..."
→ Check data path is correct
→ Use absolute paths if possible

### Error: "Out of memory"
→ Reduce `--batch_size` (try 16 or 8)
→ Reduce `--latent_dim` (try 64)
→ Use smaller dataset

### Error: "NaN values"
→ Reduce `--learning_rate` (try 0.0001)
→ Reduce `--beta` (try 0.5)
→ Increase `--batch_size`

### Poor Accuracy
→ Check data quality
→ Try `--gamma 2.0` to weight classification more
→ Increase `--epochs` (try 200)

## Next Steps

1. **Explore Results**: Look at learning curves and metrics
2. **Tune Hyperparameters**: Based on guidance above
3. **Save Best Model**: Models are automatically saved
4. **Load Models**: Use `keras.models.load_model()`
5. **Generate Latent Embeddings**: Use encoder for visualization

## Advanced Usage

### Load Trained Model and Generate Embeddings

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('model_outputs/ssvae_model.keras')

# Generate latent embeddings
latent_embeddings = model.ssvae.encoder(X_data, training=False)
# Result: (num_samples, latent_dim)
```

### Extract Classification Predictions

```python
# Get predictions
_, y_pred, _, _ = model.ssvae((X_data, y_data), training=False)
# Result: (num_samples, 1)
```

## Documentation

- `SEMI_SUPERVISED_VAE_GUIDE.md` - Complete usage guide with theory
- `FIXES_APPLIED.md` - Technical details of fixes applied
- `runner/rbam_semi_supervised_joint_vae.py` - Source code with comments

## Support

If issues persist:
1. Check `FIXES_APPLIED.md` for technical details
2. Review `SEMI_SUPERVISED_VAE_GUIDE.md` for hyperparameter recommendations
3. Check console output for specific error messages
4. Ensure test data is properly formatted

---

**Ready to train!** Start with Option 1 for SSVAE test:
```bash
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw --model_type ssvae --epochs 50
```

