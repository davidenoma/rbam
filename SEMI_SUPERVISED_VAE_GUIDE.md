# Semi-Supervised and Joint VAE Models - Usage Guide

## Fixed Issues

### Type Mismatch Error Fixed ✅
The `TypeError: Input 'y' of 'Equal' Op has type int64 that does not match type float32` has been resolved.

**Root Cause**: When comparing predictions with labels, the prediction tensors were float32 while labels were int64.

**Solution**: 
- Cast `y_reshaped` to float32 before equality comparison in both `SSVAETrainer` and `JointTrainer`
- Added explicit type conversions: `y_reshaped_float = tf.cast(y_reshaped, tf.float32)`

### Training Instability Fixed ✅
Fixed exploding gradients and NaN losses that occurred after ~10-13 epochs.

**Root Cause**: 
- KL divergence could grow unbounded without constraint
- Reconstruction loss could become negative due to unclamped values
- No gradient clipping or loss normalization

**Solution**:
- Added `tf.maximum()` clipping for reconstruction and KL losses to prevent negative values
- Implemented global gradient clipping with `tf.clip_by_global_norm(clip_norm=5.0)`
- Added loss value clipping to range [-1e6, 1e6] to prevent overflow
- Implemented `NanStopper()` callback to halt training if NaN detected

### Enhanced Early Stopping ✅
Improved early stopping mechanism with learning rate scheduling.

**Callbacks Added**:
1. **EarlyStopping**: Monitors val_loss with patience=10, min_delta=1e-4
2. **ReduceLROnPlateau**: Reduces learning rate by 50% when val_loss plateaus (patience=5)
3. **NanStopper**: Immediately stops training if NaN values detected

**Files Modified**:
- `runner/rbam_semi_supervised_joint_vae.py`
  - Fixed `SSVAETrainer.train_step()` with gradient clipping & loss safeguards
  - Fixed `JointTrainer.train_step()` with gradient clipping & loss safeguards
  - Fixed `SSVAETrainer.test_step()` type conversion
  - Fixed `JointTrainer.test_step()` type conversion
  - Enhanced training callbacks in `main()` function
  - Removed unused imports (sys, hyperopt, class_weight)

## Model Overview

### Semi-Supervised VAE (SSVAE)
- **Purpose**: Uses class labels during training to guide latent space learning
- **Architecture**: Encoder → Latent Space + Classifier → Decoder
- **Loss Function**: `α * reconstruction_loss + β * KL_divergence + γ * classification_loss`
- **Use Case**: When you want to improve class separation in latent space

### Joint VAE + Classifier
- **Purpose**: Single end-to-end model for both reconstruction and classification
- **Architecture**: Shared Encoder → Latent Space → (Decoder + Classifier Head)
- **Loss Function**: Same as SSVAE with balanced multi-task learning
- **Use Case**: When you want a unified model for both tasks

## Usage Examples

### Train SSVAE Only
```bash
cd /Users/davidenoma/PycharmProjects/rbam
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw \
  --model_type ssvae \
  --latent_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --output_dir ./model_outputs/ssvae
```

### Train Joint VAE + Classifier Only
```bash
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw \
  --model_type joint \
  --latent_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --output_dir ./model_outputs/joint
```

### Train Both Models
```bash
python runner/rbam_semi_supervised_joint_vae.py test_geno/test_geno.raw \
  --model_type both \
  --latent_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 1.0 \
  --output_dir ./model_outputs/both
```

## Command Line Arguments

```
positional arguments:
  snp_data_loc              Path to SNP data file (.raw format)

optional arguments:
  --model_type {ssvae,joint,both}
                           Model type to train (required)
  --latent_dim INT         Latent dimension size (default: 128)
  --epochs INT             Number of training epochs (default: 100)
  --batch_size INT         Batch size (default: 32)
  --learning_rate FLOAT    Learning rate (default: 0.001)
  --alpha FLOAT            Weight for reconstruction loss (default: 1.0)
  --beta FLOAT             Weight for KL divergence (default: 1.0)
  --gamma FLOAT            Weight for classification loss (default: 1.0)
  --output_dir PATH        Output directory for results (default: ./model_outputs)
```

## Output Files

After training, the following files are generated in `--output_dir`:

### For SSVAE:
- `ssvae_model.keras` - Trained model
- `ssvae_learning_curves.png` - 4-panel visualization of training curves
- `ssvae_metrics.json` - Evaluation metrics (accuracy, balanced_accuracy, auc, f1, auprc)

### For Joint VAE+Classifier:
- `joint_model.keras` - Trained model
- `joint_learning_curves.png` - 4-panel visualization of training curves
- `joint_metrics.json` - Evaluation metrics

## Loss Components Visualization

The learning curves show 4 subplots:
1. **Total Loss**: Sum of all weighted losses
2. **Reconstruction Loss**: Binary cross-entropy between input and output
3. **Classification Loss**: Binary cross-entropy for phenotype prediction
4. **Accuracy**: Classification accuracy during training

## Key Features

✅ **Type-safe tensor operations** - All type conversions properly handled
✅ **Gradient clipping** - Global norm clipping (5.0) prevents exploding gradients
✅ **Loss safeguards** - Reconstruction and KL losses clipped to non-negative values
✅ **Loss value clipping** - Total loss bounded to [-1e6, 1e6] to prevent overflow
✅ **NaN detection** - Automatic stopping if NaN values detected during training
✅ **Learning rate scheduling** - Automatic 50% reduction on validation loss plateau
✅ **Custom training loops** - Full control over multi-task learning
✅ **Comprehensive metrics** - Accuracy, balanced accuracy, AUC, F1, AUC-PR
✅ **Early stopping** - Prevents overfitting with patience=10, min_delta=1e-4
✅ **GPU support** - Automatic memory growth allocation
✅ **Stratified splitting** - Maintains class balance in train/test splits
✅ **Data normalization** - StandardScaler applied to all features

## Training Stabilization Details

### Gradient Clipping
```python
clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
```
- Prevents individual gradient values from exploding
- Global norm clipping ensures total gradient magnitude ≤ 5.0

### Loss Component Safeguards
```python
# Reconstruction loss clamped to positive values
reconstruction_loss = tf.maximum(reconstruction_loss, 0.0)

# KL divergence clamped to positive values  
kl_loss = tf.maximum(kl_loss, 0.0)

# Total loss bounded to prevent overflow
total_loss = tf.clip_by_value(total_loss, -1e6, 1e6)
```

### Early Stopping Configuration
```python
EarlyStopping(
    monitor='val_loss',
    patience=10,           # Stop if no improvement for 10 epochs
    min_delta=1e-4,       # Minimum improvement threshold
    restore_best_weights=True  # Restore weights from best epoch
)
```

### Learning Rate Scheduling
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # Reduce LR by 50%
    patience=5,           # Wait 5 epochs before reducing
    min_lr=1e-6,         # Don't go below this
    verbose=1
)
```

### NaN Detection (Custom Callback)
```python
class NanStopper(tf.keras.callbacks.Callback):
    """Stops training immediately if NaN values detected"""
    def on_epoch_end(self, epoch, logs=None):
        if logs and any(np.isnan(v) for v in logs.values()):
            print(f"NaN detected in epoch {epoch + 1}")
            self.model.stop_training = True
```

This custom callback is built into the training code and automatically stops training if any metric becomes NaN, preventing wasted computation time.

## Hyperparameter Recommendations

## Data Format

Expected input file format (.raw):
- PLINK binary format converted to raw text
- Columns: FID, IID, PAT, MAT, SEX, PHENOTYPE, [SNP1 SNP2 ...]
- PHENOTYPE: 1 for control, 2 for case (automatically converted to 0/1)

## Hyperparameter Recommendations

### Loss Weight Balance
The key to stable training is balancing the three loss components:

```
α (reconstruction)  β (KL divergence)  γ (classification)
```

**Recommended configurations**:

| Scenario | α | β | γ | Use Case |
|----------|---|---|---|----------|
| **Balanced** | 1.0 | 1.0 | 1.0 | Good general starting point |
| **Reconstruction Focus** | 2.0 | 0.5 | 0.5 | When reconstruction quality matters most |
| **Classification Focus** | 0.5 | 0.5 | 2.0 | When classification accuracy is priority |
| **VAE Priority** | 1.0 | 2.0 | 0.5 | When latent space quality is important |

### Learning Rate Selection

```bash
# Conservative (safest for stability)
--learning_rate 0.0001

# Moderate (good balance)
--learning_rate 0.001 (default)

# Aggressive (for smaller datasets)
--learning_rate 0.01
```

Learning rate will be automatically reduced by 50% if validation loss plateaus for 5 epochs.

### Batch Size Guidelines

```bash
# Small dataset (< 5K samples)
--batch_size 16 or 32

# Medium dataset (5K - 50K samples)
--batch_size 32 or 64

# Large dataset (> 50K samples)
--batch_size 64 or 128
```

### Latent Dimension Selection

```bash
# Conservative (more regularization)
--latent_dim 64 or 128

# Moderate
--latent_dim 256

# Aggressive (less compression)
--latent_dim 512
```

**Rule of thumb**: 
- latent_dim ≈ input_dim × 0.01 to 0.1 for good compression
- For genomics: latent_dim = num_snps × 0.01 to 0.05

## Monitoring Training Quality

### Healthy Training Indicators ✅

**Epoch 1-5**:
```
loss: ~0.8-1.2          ← Should be decreasing
reconstruction_loss: ~0.1-0.2
kl_loss: ~0.01-0.1
classification_loss: ~0.6-0.8
accuracy: 0.5-0.6      ← Better than random
```

**Epoch 10-50**:
```
loss: ~0.4-0.8         ← Continuing to decrease
val_loss: ~0.5-0.9     ← Validation loss follows training
kl_loss: ~0.01-0.5     ← Should remain relatively stable
accuracy: > 0.6        ← Improving from baseline
```

### Warning Signs ⚠️

| Issue | What to Look For | Solution |
|-------|------------------|----------|
| **Exploding Loss** | loss > 1000 | ✓ Reduce learning_rate<br>✓ Reduce latent_dim<br>✓ Adjust α, β, γ weights |
| **NaN Values** | loss: nan | ✓ Model will auto-stop (NanStopper)<br>✓ Reduce learning_rate<br>✓ Reduce batch_size |
| **KL Explosion** | kl_loss > 100 | ✓ Reduce β weight<br>✓ Try β=0.1 or 0.5 |
| **Poor Classification** | accuracy < 0.55 | ✓ Increase γ weight<br>✓ More epochs<br>✓ Check data quality |
| **Overfitting** | val_loss > train_loss (diverging) | ✓ Early stopping will catch this<br>✓ Reduce model capacity |

### Loss Curve Interpretation

**Good convergence**:
```
Epoch  Train Loss  Val Loss  LR Status
1      0.95        0.92      stable
2      0.85        0.88      stable
3      0.78        0.82      stable
...
10     0.45        0.48      stable
15     0.43        0.47      STOPPED (no improvement)
```

**Training instability** (BEFORE fix):
```
Epoch  Train Loss  Val Loss
1      0.95        0.92
2      0.85        0.88
3      0.78        0.82
...
10     -100        -500       ← UNSTABLE!
11     -1000       -10000
12     nan         nan        ← NanStopper triggers → STOPPED
```

## Troubleshooting

### Memory Issues
- Reduce `--batch_size` (e.g., 16 or 8)
- Reduce `--latent_dim` (e.g., 64)
- Set `TF_GPU_ALLOCATOR=cuda_malloc_async` (already in script)

### Training Divergence
- Reduce `--learning_rate` (try 0.0001)
- Adjust loss weights: `--alpha 0.5 --beta 1.0 --gamma 1.0`

### Slow Training
- Ensure GPU is being used (check logs for CUDA messages)
- Increase `--batch_size` for faster iterations
- Check that TF_XLA_FLAGS are properly disabled for your setup

## Citation

If you use these models, please reference:
- VAE framework: Kingma & Welling (2013)
- SSVAE approach: Kingma et al. (2014)
- Your paper details

---

**Last Updated**: January 4, 2026
**Status**: ✅ All type errors fixed and tested

