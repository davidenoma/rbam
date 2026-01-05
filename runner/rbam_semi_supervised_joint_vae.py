"""
Semi-supervised and Joint VAE Models for Genomics
Implements:
1. Semi-supervised VAE (SSVAE) - uses labels during training
2. Joint VAE + Classifier - single model for reconstruction and classification
"""
import os
import argparse

# Disable XLA JIT compilation
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                            average_precision_score, balanced_accuracy_score)

tf.config.optimizer.set_jit(False)

# Set up GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# ============================================================================
# Custom Callbacks
# ============================================================================

class NanStopper(tf.keras.callbacks.Callback):
    """
    Custom callback to stop training immediately if NaN values are detected.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Check if any metric is NaN
        if any(np.isnan(v) if isinstance(v, (int, float)) else False for v in logs.values()):
            print(f"\n🛑 NaN detected in epoch {epoch + 1}")
            print(f"   Metrics: {logs}")
            print("   Stopping training immediately...")
            self.model.stop_training = True


# ============================================================================
# Semi-supervised VAE (SSVAE)
# ============================================================================

@keras.saving.register_keras_serializable(package="Custom", name="SSVAE")
class SemiSupervisedVAE(tf.keras.Model):
    """
    Semi-supervised VAE that incorporates labels during training.
    The latent space is conditioned on the labels to improve separation.
    """
    def __init__(self, encoder, decoder, classifier, alpha=1.0, beta=1.0, **kwargs):
        """
        Args:
            encoder: Encoder network
            decoder: Decoder network
            classifier: Classification network operating on latent space
            alpha: Weight for reconstruction loss
            beta: Weight for KL divergence loss
        """
        super(SemiSupervisedVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.alpha = alpha
        self.beta = beta

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs, training=False):
        """
        Args:
            inputs: Tuple of (x, y) where x is data and y is label
        """
        if isinstance(inputs, tuple):
            x, y = inputs
        else:
            x = inputs
            y = None

        # Encode
        encoder_output = self.encoder(x, training=training)
        z_mean, z_log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)

        # Reparameterize
        z = self.reparameterize(z_mean, z_log_var)

        # Decode
        reconstructed = self.decoder(z, training=training)

        # Classify
        y_pred = self.classifier(z, training=training)

        return reconstructed, y_pred, z_mean, z_log_var

    def get_config(self):
        config = super(SemiSupervisedVAE, self).get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
            'classifier': keras.saving.serialize_keras_object(self.classifier),
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop('encoder'))
        decoder = keras.saving.deserialize_keras_object(config.pop('decoder'))
        classifier = keras.saving.deserialize_keras_object(config.pop('classifier'))
        alpha = config.pop('alpha', 1.0)
        beta = config.pop('beta', 1.0)
        return cls(encoder=encoder, decoder=decoder, classifier=classifier,
                  alpha=alpha, beta=beta)


@keras.saving.register_keras_serializable(package="Custom", name="ssvae_loss")
def ssvae_loss(ssvae_model, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Loss function for semi-supervised VAE.

    Args:
        ssvae_model: SSVAE model instance
        alpha: Weight for reconstruction loss
        beta: Weight for KL divergence
        gamma: Weight for classification loss
    """
    @keras.saving.register_keras_serializable(package="Custom", name="loss")
    def loss(inputs, outputs):
        x, y_true = inputs
        x_reconstructed, y_pred, z_mean, z_log_var = outputs

        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x, x_reconstructed)
        )

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        # Classification loss (only for labeled data)
        y_true_reshaped = tf.reshape(y_true, [-1, 1])
        classification_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_true_reshaped, y_pred)
        )

        # Total loss
        total_loss = (alpha * reconstruction_loss +
                     beta * kl_loss +
                     gamma * classification_loss)

        return total_loss

    return loss


def create_ssvae_model(input_dim, latent_dim, num_hidden_layers_encoder,
                       num_hidden_layers_decoder, encoding_dimensions,
                       decoding_dimensions, classifier_hidden_dim,
                       activation, learning_rate, alpha=1.0, beta=1.0, gamma=1.0):
    """Create a semi-supervised VAE model."""

    # Encoder
    encoder_layers = [input_dim] + [encoding_dimensions] * num_hidden_layers_encoder + [2 * latent_dim]
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation) for layer in encoder_layers[1:]]
    ], name="encoder")

    # Decoder
    decoder_layers = [latent_dim] + [decoding_dimensions] * num_hidden_layers_decoder + [input_dim]
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation if i < len(decoder_layers) - 2 else 'sigmoid')
          for i, layer in enumerate(decoder_layers[1:])]
    ], name="decoder")

    # Classifier (operates on latent space)
    classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(classifier_hidden_dim, activation=activation),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(classifier_hidden_dim // 2, activation=activation),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="classifier")

    # Create SSVAE
    ssvae = SemiSupervisedVAE(encoder=encoder, decoder=decoder, classifier=classifier,
                              alpha=alpha, beta=beta)

    # Custom training step
    class SSVAETrainer(tf.keras.Model):
        def __init__(self, ssvae, alpha, beta, gamma):
            super().__init__()
            self.ssvae = ssvae
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        def call(self, inputs):
            return self.ssvae(inputs)

        def train_step(self, data):
            x, y = data

            with tf.GradientTape() as tape:
                reconstructed, y_pred, z_mean, z_log_var = self.ssvae((x, y), training=True)

                # Reconstruction loss - clipped to prevent negative values
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(x, reconstructed)
                )
                reconstruction_loss = tf.maximum(reconstruction_loss, 0.0)

                # KL divergence - with safeguard
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                kl_loss = tf.maximum(kl_loss, 0.0)

                # Classification loss
                y_reshaped = tf.reshape(y, [-1, 1])
                classification_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(y_reshaped, y_pred)
                )

                # Total loss with safeguards against NaN
                total_loss = (self.alpha * reconstruction_loss +
                            self.beta * kl_loss +
                            self.gamma * classification_loss)

                # Clip loss to prevent explosion
                total_loss = tf.clip_by_value(total_loss, -1e6, 1e6)

            # Compute gradients
            trainable_vars = self.ssvae.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)

            # Clip gradients to prevent explosion
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
            self.optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

            # Update metrics
            y_pred_binary = tf.cast(tf.round(y_pred), tf.float32)
            y_reshaped_float = tf.cast(y_reshaped, tf.float32)
            return {
                'loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'classification_loss': classification_loss,
                'accuracy': tf.reduce_mean(
                    tf.cast(tf.equal(y_pred_binary, y_reshaped_float), tf.float32)
                )
            }

        def test_step(self, data):
            x, y = data
            reconstructed, y_pred, z_mean, z_log_var = self.ssvae((x, y), training=False)

            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x, reconstructed)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            y_reshaped = tf.reshape(y, [-1, 1])
            classification_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_reshaped, y_pred)
            )
            total_loss = (self.alpha * reconstruction_loss +
                        self.beta * kl_loss +
                        self.gamma * classification_loss)

            y_pred_binary = tf.cast(tf.round(y_pred), tf.float32)
            y_reshaped_float = tf.cast(y_reshaped, tf.float32)
            return {
                'loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'classification_loss': classification_loss,
                'accuracy': tf.reduce_mean(
                    tf.cast(tf.equal(y_pred_binary, y_reshaped_float), tf.float32)
                )
            }

    trainer = SSVAETrainer(ssvae, alpha, beta, gamma)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return trainer


# ============================================================================
# Joint VAE + Classifier
# ============================================================================

@keras.saving.register_keras_serializable(package="Custom", name="JointVAEClassifier")
class JointVAEClassifier(tf.keras.Model):
    """
    Joint VAE and Classifier model that shares representations.
    Single model that performs both reconstruction and classification.
    """
    def __init__(self, encoder, decoder, classifier_head, alpha=1.0, beta=1.0, **kwargs):
        """
        Args:
            encoder: Shared encoder network
            decoder: Decoder for reconstruction
            classifier_head: Classification head attached to latent space
            alpha: Weight for reconstruction loss
            beta: Weight for KL divergence
        """
        super(JointVAEClassifier, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier_head = classifier_head
        self.alpha = alpha
        self.beta = beta

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def encode(self, x, training=False):
        """Encode input and return mean and log variance."""
        encoder_output = self.encoder(x, training=training)
        z_mean, z_log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def decode(self, z, training=False):
        """Decode latent representation."""
        return self.decoder(z, training=training)

    def classify(self, z, training=False):
        """Classify from latent representation."""
        return self.classifier_head(z, training=training)

    def call(self, inputs, training=False):
        """
        Forward pass through the model.
        Returns: (reconstructed, y_pred, z_mean, z_log_var)
        """
        # Encode
        z_mean, z_log_var = self.encode(inputs, training=training)

        # Reparameterize
        z = self.reparameterize(z_mean, z_log_var)

        # Decode
        reconstructed = self.decode(z, training=training)

        # Classify
        y_pred = self.classify(z, training=training)

        return reconstructed, y_pred, z_mean, z_log_var

    def get_config(self):
        config = super(JointVAEClassifier, self).get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
            'classifier_head': keras.saving.serialize_keras_object(self.classifier_head),
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop('encoder'))
        decoder = keras.saving.deserialize_keras_object(config.pop('decoder'))
        classifier_head = keras.saving.deserialize_keras_object(config.pop('classifier_head'))
        alpha = config.pop('alpha', 1.0)
        beta = config.pop('beta', 1.0)
        return cls(encoder=encoder, decoder=decoder, classifier_head=classifier_head,
                  alpha=alpha, beta=beta)


def create_joint_vae_classifier(input_dim, latent_dim, num_hidden_layers_encoder,
                                num_hidden_layers_decoder, encoding_dimensions,
                                decoding_dimensions, classifier_hidden_dims,
                                activation, learning_rate, alpha=1.0, beta=1.0, gamma=1.0):
    """Create a joint VAE + Classifier model."""

    # Shared encoder
    encoder_layers = [input_dim] + [encoding_dimensions] * num_hidden_layers_encoder + [2 * latent_dim]
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation) for layer in encoder_layers[1:]]
    ], name="shared_encoder")

    # Decoder
    decoder_layers = [latent_dim] + [decoding_dimensions] * num_hidden_layers_decoder + [input_dim]
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        *[tf.keras.layers.Dense(layer, activation=activation if i < len(decoder_layers) - 2 else 'sigmoid')
          for i, layer in enumerate(decoder_layers[1:])]
    ], name="decoder")

    # Classification head
    classifier_layers = [tf.keras.layers.Input(shape=(latent_dim,))]
    for dim in classifier_hidden_dims:
        classifier_layers.extend([
            tf.keras.layers.Dense(dim, activation=activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3)
        ])
    classifier_layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    classifier_head = tf.keras.Sequential(classifier_layers, name="classifier_head")

    # Create joint model
    joint_model = JointVAEClassifier(
        encoder=encoder,
        decoder=decoder,
        classifier_head=classifier_head,
        alpha=alpha,
        beta=beta
    )

    # Custom training step
    class JointTrainer(tf.keras.Model):
        def __init__(self, joint_model, alpha, beta, gamma):
            super().__init__()
            self.joint_model = joint_model
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        def call(self, inputs):
            return self.joint_model(inputs)

        def train_step(self, data):
            x, y = data

            with tf.GradientTape() as tape:
                reconstructed, y_pred, z_mean, z_log_var = self.joint_model(x, training=True)

                # Reconstruction loss - clipped to prevent negative values
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(x, reconstructed)
                )
                reconstruction_loss = tf.maximum(reconstruction_loss, 0.0)

                # KL divergence - with annealing to prevent explosion
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                kl_loss = tf.maximum(kl_loss, 0.0)

                # Classification loss
                y_reshaped = tf.reshape(y, [-1, 1])
                classification_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(y_reshaped, y_pred)
                )

                # Total loss with safeguards against NaN
                total_loss = (self.alpha * reconstruction_loss +
                            self.beta * kl_loss +
                            self.gamma * classification_loss)

                # Clip loss to prevent explosion
                total_loss = tf.clip_by_value(total_loss, -1e6, 1e6)

            # Compute gradients
            trainable_vars = self.joint_model.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)

            # Clip gradients to prevent explosion
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
            self.optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

            # Compute accuracy
            y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
            y_reshaped_float = tf.cast(y_reshaped, tf.float32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(y_pred_binary, y_reshaped_float), tf.float32)
            )

            return {
                'loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'classification_loss': classification_loss,
                'accuracy': accuracy
            }

        def test_step(self, data):
            x, y = data
            reconstructed, y_pred, z_mean, z_log_var = self.joint_model(x, training=False)

            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x, reconstructed)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            y_reshaped = tf.reshape(y, [-1, 1])
            classification_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_reshaped, y_pred)
            )
            total_loss = (self.alpha * reconstruction_loss +
                        self.beta * kl_loss +
                        self.gamma * classification_loss)

            y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
            y_reshaped_float = tf.cast(y_reshaped, tf.float32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(y_pred_binary, y_reshaped_float), tf.float32)
            )

            return {
                'loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'classification_loss': classification_loss,
                'accuracy': accuracy
            }

    trainer = JointTrainer(joint_model, alpha, beta, gamma)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return trainer


# ============================================================================
# Utility Functions
# ============================================================================

def save_learning_curves(history, output_path, model_type):
    """Save learning curves for the model."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(history.history['loss'], label='Training')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title(f'{model_type} - Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Reconstruction loss
    axes[0, 1].plot(history.history['reconstruction_loss'], label='Training')
    if 'val_reconstruction_loss' in history.history:
        axes[0, 1].plot(history.history['val_reconstruction_loss'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Classification loss
    axes[1, 0].plot(history.history['classification_loss'], label='Training')
    if 'val_classification_loss' in history.history:
        axes[1, 0].plot(history.history['val_classification_loss'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Classification Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Accuracy
    axes[1, 1].plot(history.history['accuracy'], label='Training')
    if 'val_accuracy' in history.history:
        axes[1, 1].plot(history.history['val_accuracy'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Classification Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves: {output_path}")


def evaluate_model(trainer, X_test, y_test, model_type):
    """Evaluate the model and return metrics."""
    # Get predictions based on trainer type
    if hasattr(trainer, 'joint_model'):
        # JointTrainer
        _, y_pred_proba, _, _ = trainer.joint_model(X_test, training=False)
    else:
        # SSVAETrainer
        _, y_pred_proba, _, _ = trainer.ssvae(X_test, training=False)

    y_pred_proba = y_pred_proba.numpy().ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'auprc': average_precision_score(y_test, y_pred_proba)
    }

    return metrics


def print_metrics(metrics, model_type):
    """Print evaluation metrics."""
    print(f"\n{model_type} Evaluation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-PR: {metrics['auprc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Semi-supervised and Joint VAE Models')
    parser.add_argument('snp_data_loc', type=str, help='Path to SNP data file (.raw)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['ssvae', 'joint', 'both'],
                       help='Model type: ssvae (semi-supervised), joint, or both')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for KL divergence')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--output_dir', type=str, default='./model_outputs',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Semi-supervised and Joint VAE Models")
    print(f"{'=' * 60}")
    print(f"Model type: {args.model_type}")
    print(f"SNP data: {args.snp_data_loc}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"{'=' * 60}\n")

    # Load data
    print("Loading data...")
    data = pd.read_csv(args.snp_data_loc, sep='\\s+')

    # Extract phenotype and genotype data
    phenotype = data['PHENOTYPE'].values
    genotype_data = data.iloc[:, 6:].values

    # Convert labels: 1 (control) -> 0, 2 (case) -> 1
    labels = np.where(phenotype == 1, 0, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        genotype_data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    input_dim = X_train.shape[1]

    # Train Semi-supervised VAE
    if args.model_type in ['ssvae', 'both']:
        print(f"\n{'=' * 60}")
        print("Training Semi-supervised VAE (SSVAE)")
        print(f"{'=' * 60}\n")

        ssvae_model = create_ssvae_model(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            num_hidden_layers_encoder=2,
            num_hidden_layers_decoder=2,
            encoding_dimensions=256,
            decoding_dimensions=256,
            classifier_hidden_dim=128,
            activation='relu',
            learning_rate=args.learning_rate,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma
        )

        # Train with enhanced callbacks
        history_ssvae = ssvae_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=1e-4,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                ),
                NanStopper()
            ],
            verbose=1
        )

        # Save learning curves
        save_learning_curves(
            history_ssvae,
            os.path.join(args.output_dir, 'ssvae_learning_curves.png'),
            'SSVAE'
        )

        # Evaluate
        ssvae_metrics = evaluate_model(ssvae_model, X_test, y_test, 'SSVAE')
        print_metrics(ssvae_metrics, 'SSVAE')

        # Save model and metrics
        ssvae_model.save(os.path.join(args.output_dir, 'ssvae_model.keras'))
        with open(os.path.join(args.output_dir, 'ssvae_metrics.json'), 'w') as f:
            json.dump(ssvae_metrics, f, indent=2)

    # Train Joint VAE + Classifier
    if args.model_type in ['joint', 'both']:
        print(f"\n{'=' * 60}")
        print("Training Joint VAE + Classifier")
        print(f"{'=' * 60}\n")

        joint_model = create_joint_vae_classifier(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            num_hidden_layers_encoder=2,
            num_hidden_layers_decoder=2,
            encoding_dimensions=256,
            decoding_dimensions=256,
            classifier_hidden_dims=[128, 64],
            activation='relu',
            learning_rate=args.learning_rate,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma
        )

        # Train with enhanced callbacks
        history_joint = joint_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=1e-4,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                ),
                NanStopper()
            ],
            verbose=1
        )

        # Save learning curves
        save_learning_curves(
            history_joint,
            os.path.join(args.output_dir, 'joint_learning_curves.png'),
            'Joint VAE+Classifier'
        )

        # Evaluate
        joint_metrics = evaluate_model(joint_model, X_test, y_test, 'Joint')
        print_metrics(joint_metrics, 'Joint VAE+Classifier')

        # Save model and metrics
        joint_model.save(os.path.join(args.output_dir, 'joint_model.keras'))
        with open(os.path.join(args.output_dir, 'joint_metrics.json'), 'w') as f:
            json.dump(joint_metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete. Results saved to: {args.output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()