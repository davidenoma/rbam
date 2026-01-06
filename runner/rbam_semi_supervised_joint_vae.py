"""
Semi-supervised and Joint VAE Models for Genomics
Implements:
1. Semi-supervised VAE (SSVAE) - uses labels during training
2. Joint VAE + Classifier - single model for reconstruction and classification
3. Hyperparameter optimization with cross-validation
4. Independent test set evaluation
"""
import os
import argparse

# ============================================================================
# CRITICAL: Disable XLA JIT compilation BEFORE importing TensorFlow
# This prevents "Can not combine dim orders" errors on GPU
# ============================================================================
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0 --xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force eager execution for debugging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import json
import matplotlib.pyplot as plt
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK, space_eval
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                            average_precision_score, balanced_accuracy_score,
                            precision_score, recall_score, confusion_matrix)

# Disable XLA JIT compilation at TensorFlow level
try:
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({
        'disable_meta_optimizer': True,
        'disable_model_pruning': True
    })
except:
    pass

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

        # Reconstruction loss - MSE for StandardScaler normalized data
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_reconstructed))

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
    """Create a semi-supervised VAE model with improved numerical stability."""

    # Encoder with BatchNormalization for stability
    encoder_input = tf.keras.layers.Input(shape=(input_dim,))
    x = encoder_input
    for i in range(num_hidden_layers_encoder):
        x = tf.keras.layers.Dense(encoding_dimensions, activation=activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    # Split into mean and log_var with proper initialization
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean',
                                    kernel_initializer='glorot_uniform')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var',
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer=tf.keras.initializers.Constant(-2.0))(x)

    # Concatenate for single output
    encoder_output = tf.keras.layers.Concatenate()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

    # Decoder with BatchNormalization (linear output for StandardScaler data)
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
    x = decoder_input
    for i in range(num_hidden_layers_decoder):
        x = tf.keras.layers.Dense(decoding_dimensions, activation=activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    decoder_output = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")

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

    # Custom training step with KL annealing
    class SSVAETrainer(tf.keras.Model):
        def __init__(self, ssvae, alpha, beta, gamma):
            super().__init__()
            self.ssvae = ssvae
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self._train_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
            self._kl_warmup_steps = 1000  # Warmup steps for KL annealing

        def call(self, inputs):
            return self.ssvae(inputs)

        def _get_kl_weight(self):
            """Compute KL weight with linear warmup."""
            progress = tf.cast(self._train_counter, tf.float32) / tf.cast(self._kl_warmup_steps, tf.float32)
            return tf.minimum(progress, 1.0) * self.beta

        def train_step(self, data):
            x, y = data
            self._train_counter.assign_add(1)
            kl_weight = self._get_kl_weight()

            with tf.GradientTape() as tape:
                reconstructed, y_pred, z_mean, z_log_var = self.ssvae((x, y), training=True)

                # Clip z_log_var to prevent numerical issues
                z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
                z_mean = tf.clip_by_value(z_mean, -10.0, 10.0)

                # Reconstruction loss - MSE for StandardScaler normalized data
                reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))

                # KL divergence - with safeguard and capping
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                kl_loss = tf.clip_by_value(kl_loss, 0.0, 100.0)

                # Classification loss
                y_reshaped = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
                classification_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(y_reshaped, tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7))
                )

                # Total loss with KL annealing
                total_loss = (self.alpha * reconstruction_loss +
                            kl_weight * kl_loss +
                            self.gamma * classification_loss)

                # Clip loss to prevent explosion
                total_loss = tf.clip_by_value(total_loss, -1e6, 1e6)

            # Compute gradients
            trainable_vars = self.ssvae.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)

            # Filter out None gradients and clip
            gradients = [g if g is not None else tf.zeros_like(v)
                        for g, v in zip(gradients, trainable_vars)]
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
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

            # Clip for safety
            z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)

            # Reconstruction loss - MSE for StandardScaler normalized data
            reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))

            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            kl_loss = tf.clip_by_value(kl_loss, 0.0, 100.0)

            y_reshaped = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
            classification_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_reshaped, tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7))
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

        def get_config(self):
            """Return config for serialization."""
            config = super().get_config()
            config.update({
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Note: ssvae must be passed separately when reconstructing
            return cls(**config)

    trainer = SSVAETrainer(ssvae, alpha, beta, gamma)
    # Disable jit_compile to prevent XLA "Can not combine dim orders" errors on GPU
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        jit_compile=False
    )

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
    """Create a joint VAE + Classifier model with improved numerical stability."""

    # Shared encoder with BatchNormalization
    encoder_input = tf.keras.layers.Input(shape=(input_dim,))
    x = encoder_input
    for i in range(num_hidden_layers_encoder):
        x = tf.keras.layers.Dense(encoding_dimensions, activation=activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    # Split into mean and log_var with proper initialization
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean',
                                    kernel_initializer='glorot_uniform')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var',
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer=tf.keras.initializers.Constant(-2.0))(x)

    encoder_output = tf.keras.layers.Concatenate()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_input, encoder_output, name="shared_encoder")

    # Decoder with BatchNormalization (linear output for StandardScaler data)
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
    x = decoder_input
    for i in range(num_hidden_layers_decoder):
        x = tf.keras.layers.Dense(decoding_dimensions, activation=activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    decoder_output = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")

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

    # Custom training step with KL annealing
    class JointTrainer(tf.keras.Model):
        def __init__(self, joint_model, alpha, beta, gamma):
            super().__init__()
            self.joint_model = joint_model
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self._train_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
            self._kl_warmup_steps = 1000  # Warmup steps for KL annealing

        def call(self, inputs):
            return self.joint_model(inputs)

        def _get_kl_weight(self):
            """Compute KL weight with linear warmup."""
            progress = tf.cast(self._train_counter, tf.float32) / tf.cast(self._kl_warmup_steps, tf.float32)
            return tf.minimum(progress, 1.0) * self.beta

        def train_step(self, data):
            x, y = data
            self._train_counter.assign_add(1)
            kl_weight = self._get_kl_weight()

            with tf.GradientTape() as tape:
                reconstructed, y_pred, z_mean, z_log_var = self.joint_model(x, training=True)

                # Clip z_log_var and z_mean to prevent numerical issues
                z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
                z_mean = tf.clip_by_value(z_mean, -10.0, 10.0)

                # Reconstruction loss - MSE for StandardScaler normalized data
                reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))

                # KL divergence - with annealing and capping
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                kl_loss = tf.clip_by_value(kl_loss, 0.0, 100.0)

                # Classification loss
                y_reshaped = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
                classification_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(y_reshaped, tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7))
                )

                # Total loss with KL annealing
                total_loss = (self.alpha * reconstruction_loss +
                            kl_weight * kl_loss +
                            self.gamma * classification_loss)

                # Clip loss to prevent explosion
                total_loss = tf.clip_by_value(total_loss, -1e6, 1e6)

            # Compute gradients
            trainable_vars = self.joint_model.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)

            # Filter out None gradients and clip
            gradients = [g if g is not None else tf.zeros_like(v)
                        for g, v in zip(gradients, trainable_vars)]
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
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

            # Clip for safety
            z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)

            # Reconstruction loss - MSE for StandardScaler normalized data
            reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))

            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            kl_loss = tf.clip_by_value(kl_loss, 0.0, 100.0)

            y_reshaped = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
            classification_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_reshaped, tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7))
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

        def get_config(self):
            """Return config for serialization."""
            config = super().get_config()
            config.update({
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Note: joint_model must be passed separately when reconstructing
            return cls(**config)

    trainer = JointTrainer(joint_model, alpha, beta, gamma)
    # Disable jit_compile to prevent XLA "Can not combine dim orders" errors on GPU
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        jit_compile=False
    )

    return trainer


# ============================================================================
# Utility Functions
# ============================================================================

def save_learning_curves(history, output_path, model_type):
    """Save total loss learning curve for the model at 600 dpi."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(history.history['loss'], label='Training', linewidth=2)
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title(f'{model_type} - Training Loss Curve', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves: {output_path}")


def save_model_parameters(params, output_path, model_type):
    """Save model parameters to a text file."""
    with open(output_path, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"{model_type} - Model Parameters\n")
        f.write(f"{'=' * 60}\n\n")

        for k, v in params.items():
            f.write(f"  {k}: {v}\n")

    # Also save as JSON
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        # Convert any non-serializable types
        params_serializable = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                params_serializable[k] = list(v)
            elif isinstance(v, np.ndarray):
                params_serializable[k] = v.tolist()
            else:
                params_serializable[k] = v
        json.dump(params_serializable, f, indent=2)

    print(f"Saved model parameters: {output_path}")


def plot_model_architecture(model, output_path, model_type):
    """Plot and save model architecture if available."""
    try:
        # Try to use keras plot_model
        from tensorflow.keras.utils import plot_model

        # Get the underlying model
        if hasattr(model, 'joint_model'):
            underlying_model = model.joint_model
        elif hasattr(model, 'ssvae'):
            underlying_model = model.ssvae
        else:
            underlying_model = model

        plot_model(
            underlying_model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            dpi=600
        )
        print(f"Saved model architecture: {output_path}")
        return True
    except ImportError:
        print("Note: Install pydot and graphviz to plot model architecture")
        return False
    except Exception as e:
        print(f"Could not plot model architecture: {e}")
        return False


def evaluate_model_comprehensive(trainer, X_test, y_test, model_type):
    """
    Comprehensive model evaluation with all metrics.
    Returns detailed metrics dictionary.
    """
    # Get predictions based on trainer type
    if hasattr(trainer, 'joint_model'):
        _, y_pred_proba, z_mean, _ = trainer.joint_model(X_test, training=False)
    else:
        _, y_pred_proba, z_mean, _ = trainer.ssvae(X_test, training=False)

    y_pred_proba = y_pred_proba.numpy().ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Compute comprehensive metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
        'auc_prc': float(average_precision_score(y_test, y_pred_proba)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'n_samples': len(y_test),
        'n_positive': int(sum(y_test)),
        'n_negative': int(len(y_test) - sum(y_test))
    }

    return metrics, y_pred_proba, z_mean.numpy()


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


def print_comprehensive_metrics(metrics, model_type, output_file=None):
    """Print and optionally save comprehensive metrics."""
    output_lines = [
        f"\n{'=' * 60}",
        f"{model_type} - Comprehensive Evaluation Metrics",
        f"{'=' * 60}",
        f"  Accuracy:           {metrics['accuracy']:.4f}",
        f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}",
        f"  AUC-ROC:            {metrics['auc_roc']:.4f}",
        f"  AUC-PRC:            {metrics['auc_prc']:.4f}",
        f"  F1 Score:           {metrics['f1_score']:.4f}",
        f"  Precision:          {metrics['precision']:.4f}",
        f"  Recall:             {metrics['recall']:.4f}",
        f"  Sensitivity:        {metrics['sensitivity']:.4f}",
        f"  Specificity:        {metrics['specificity']:.4f}",
        f"  TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}, "
        f"FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}",
        f"  Total Samples: {metrics['n_samples']} (Pos: {metrics['n_positive']}, Neg: {metrics['n_negative']})",
        f"{'=' * 60}"
    ]

    for line in output_lines:
        print(line)

    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))


# ============================================================================
# Cross-Validation Functions
# ============================================================================

def run_cross_validation(X, y, model_type, input_dim, params, n_folds=5, epochs=50, batch_size=32):
    """
    Run k-fold cross-validation and return aggregated metrics.

    Args:
        X: Feature matrix
        y: Labels
        model_type: 'ssvae' or 'joint'
        input_dim: Input dimension
        params: Dictionary of model hyperparameters
        n_folds: Number of CV folds
        epochs: Training epochs per fold
        batch_size: Batch size

    Returns:
        Dictionary with mean and std of metrics across folds
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    fold_predictions = []
    fold_histories = []

    print(f"\n{'=' * 60}")
    print(f"Running {n_folds}-Fold Cross-Validation for {model_type.upper()}")
    print(f"{'=' * 60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Scale data using StandardScaler
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        # Create model
        if model_type == 'ssvae':
            model = create_ssvae_model(
                input_dim=input_dim,
                latent_dim=params.get('latent_dim', 128),
                num_hidden_layers_encoder=params.get('num_hidden_layers_encoder', 2),
                num_hidden_layers_decoder=params.get('num_hidden_layers_decoder', 2),
                encoding_dimensions=params.get('encoding_dimensions', 256),
                decoding_dimensions=params.get('decoding_dimensions', 256),
                classifier_hidden_dim=params.get('classifier_hidden_dim', 128),
                activation=params.get('activation', 'relu'),
                learning_rate=params.get('learning_rate', 0.001),
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 0.1),
                gamma=params.get('gamma', 1.0)
            )
        else:  # joint
            model = create_joint_vae_classifier(
                input_dim=input_dim,
                latent_dim=params.get('latent_dim', 128),
                num_hidden_layers_encoder=params.get('num_hidden_layers_encoder', 2),
                num_hidden_layers_decoder=params.get('num_hidden_layers_decoder', 2),
                encoding_dimensions=params.get('encoding_dimensions', 256),
                decoding_dimensions=params.get('decoding_dimensions', 256),
                classifier_hidden_dims=params.get('classifier_hidden_dims', [128, 64]),
                activation=params.get('activation', 'relu'),
                learning_rate=params.get('learning_rate', 0.001),
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 0.1),
                gamma=params.get('gamma', 1.0)
            )

        # Train
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10,
                    restore_best_weights=True, min_delta=1e-4, verbose=0
                ),
                NanStopper()
            ],
            verbose=0
        )

        # Save fold history (for aggregated CV results, not individual plots)
        fold_histories.append(history.history)


        # Evaluate
        metrics, y_pred_proba, _ = evaluate_model_comprehensive(model, X_val_fold, y_val_fold, model_type)
        fold_metrics.append(metrics)
        fold_predictions.append({
            'y_true': y_val_fold,
            'y_pred_proba': y_pred_proba,
            'fold': fold
        })

        print(f"  Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}, "
              f"AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1_score']:.4f}")

        # Clear memory
        tf.keras.backend.clear_session()

    # Aggregate metrics
    cv_results = {}
    metric_names = fold_metrics[0].keys()

    for metric in metric_names:
        if isinstance(fold_metrics[0][metric], (int, float)):
            values = [fm[metric] for fm in fold_metrics]
            cv_results[f'{metric}_mean'] = float(np.mean(values))
            cv_results[f'{metric}_std'] = float(np.std(values))

    cv_results['n_folds'] = n_folds
    cv_results['fold_metrics'] = fold_metrics
    cv_results['fold_histories'] = fold_histories

    print(f"\n--- Cross-Validation Summary ---")
    print(f"  Accuracy:     {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"  Balanced Acc: {cv_results['balanced_accuracy_mean']:.4f} ± {cv_results['balanced_accuracy_std']:.4f}")
    print(f"  AUC-ROC:      {cv_results['auc_roc_mean']:.4f} ± {cv_results['auc_roc_std']:.4f}")
    print(f"  F1 Score:     {cv_results['f1_score_mean']:.4f} ± {cv_results['f1_score_std']:.4f}")
    print(f"  AUC-PRC:      {cv_results['auc_prc_mean']:.4f} ± {cv_results['auc_prc_std']:.4f}")

    return cv_results, fold_predictions


# ============================================================================
# Hyperparameter Optimization
# ============================================================================

def run_hyperparameter_optimization(X_train, y_train, X_val, y_val, model_type, input_dim,
                                     max_evals=20, epochs=50, batch_size=32):
    """
    Run hyperparameter optimization using Hyperopt.

    Returns:
        best_params: Best hyperparameters found
        trials: Hyperopt trials object
    """
    print(f"\n{'=' * 60}")
    print(f"Running Hyperparameter Optimization for {model_type.upper()}")
    print(f"{'=' * 60}")

    # Define search space
    space = {
        'latent_dim': hp.choice('latent_dim', [32, 64, 128, 256]),
        'encoding_dimensions': hp.choice('encoding_dimensions', [128, 256, 512]),
        'decoding_dimensions': hp.choice('decoding_dimensions', [128, 256, 512]),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
        'alpha': hp.uniform('alpha', 0.5, 2.0),
        'beta': hp.uniform('beta', 0.01, 1.0),  # Reduced range to prevent KL explosion
        'gamma': hp.uniform('gamma', 0.5, 2.0),
        'num_hidden_layers_encoder': hp.choice('num_hidden_layers_encoder', [1, 2]),  # Reduced to prevent XLA issues
        'num_hidden_layers_decoder': hp.choice('num_hidden_layers_decoder', [1, 2]),  # Reduced to prevent XLA issues
    }

    def objective(params):
        """
        Objective function for hyperparameter optimization.
        Includes retry logic and proper GPU memory cleanup to handle XLA compilation failures.
        """
        max_retries = 2

        for attempt in range(max_retries):
            try:
                # Clear GPU memory before each trial
                tf.keras.backend.clear_session()
                import gc
                gc.collect()

                # Create model with jit_compile=False to avoid XLA issues
                if model_type == 'ssvae':
                    model = create_ssvae_model(
                        input_dim=input_dim,
                        latent_dim=params['latent_dim'],
                        num_hidden_layers_encoder=params['num_hidden_layers_encoder'],
                        num_hidden_layers_decoder=params['num_hidden_layers_decoder'],
                        encoding_dimensions=params['encoding_dimensions'],
                        decoding_dimensions=params['decoding_dimensions'],
                        classifier_hidden_dim=128,
                        activation='relu',
                        learning_rate=params['learning_rate'],
                        alpha=params['alpha'],
                        beta=params['beta'],
                        gamma=params['gamma']
                    )
                else:
                    model = create_joint_vae_classifier(
                        input_dim=input_dim,
                        latent_dim=params['latent_dim'],
                        num_hidden_layers_encoder=params['num_hidden_layers_encoder'],
                        num_hidden_layers_decoder=params['num_hidden_layers_decoder'],
                        encoding_dimensions=params['encoding_dimensions'],
                        decoding_dimensions=params['decoding_dimensions'],
                        classifier_hidden_dims=[128, 64],
                        activation='relu',
                        learning_rate=params['learning_rate'],
                        alpha=params['alpha'],
                        beta=params['beta'],
                        gamma=params['gamma']
                    )

                # Train with run_eagerly=True to avoid XLA graph compilation issues
                # This is slower but more stable
                model.compile(
                    optimizer=model.optimizer,
                    run_eagerly=True  # Disable graph compilation
                )

                # Train
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=5,
                            restore_best_weights=True, verbose=0
                        ),
                        NanStopper()
                    ],
                    verbose=0
                )

                # Evaluate
                metrics = evaluate_model(model, X_val, y_val, model_type)

                # Use negative AUC as loss (we want to maximize AUC)
                loss = -metrics['auc']

                # Cleanup
                del model
                tf.keras.backend.clear_session()
                gc.collect()

                return {'loss': loss, 'status': STATUS_OK, 'metrics': metrics}

            except Exception as e:
                error_msg = str(e)
                print(f"Trial attempt {attempt + 1}/{max_retries} failed: {error_msg[:200]}")

                # Clean up on failure
                tf.keras.backend.clear_session()
                import gc
                gc.collect()

                # If this is an XLA/dim order error, try again
                if "dim order" in error_msg.lower() or "xla" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print("  Retrying with different random seed...")
                        continue

                # For other errors or max retries reached, return failure
                return {'loss': 1.0, 'status': STATUS_OK}

        return {'loss': 1.0, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=True
    )

    # Convert best params from indices to actual values
    best_params = space_eval(space, best)

    print(f"\nBest Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, trials


def save_cv_results(cv_results, output_path, model_type, params=None):
    """Save cross-validation results to file."""
    with open(output_path, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"{model_type} - Cross-Validation Results\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Number of folds: {cv_results['n_folds']}\n\n")

        # Write model parameters if provided
        if params:
            f.write("Model Parameters:\n")
            f.write("-" * 40 + "\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("Aggregated Metrics (Mean ± Std):\n")
        f.write("-" * 40 + "\n")

        metric_names = ['accuracy', 'balanced_accuracy', 'auc_roc', 'auc_prc',
                       'f1_score', 'precision', 'recall', 'sensitivity', 'specificity']

        for metric in metric_names:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in cv_results:
                f.write(f"  {metric:20s}: {cv_results[mean_key]:.4f} ± {cv_results[std_key]:.4f}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("Per-Fold Metrics:\n")
        f.write("-" * 40 + "\n")

        for i, fold_metric in enumerate(cv_results.get('fold_metrics', [])):
            f.write(f"\nFold {i + 1}:\n")
            for k, v in fold_metric.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {k}: {v}\n")

    print(f"Saved CV results: {output_path}")


def save_independent_test_results(metrics, output_path, model_type, params=None):
    """Save independent test set results to file."""
    with open(output_path, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"{model_type} - Independent Test Set Results\n")
        f.write(f"{'=' * 60}\n\n")

        # Write model parameters if provided
        if params:
            f.write("Model Parameters:\n")
            f.write("-" * 40 + "\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("Classification Metrics:\n")
        f.write("-" * 40 + "\n")

        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"  {k:20s}: {v:.4f}\n")
            else:
                f.write(f"  {k:20s}: {v}\n")

    print(f"Saved independent test results: {output_path}")


def save_hyperparameters(params, output_path, model_type):
    """Save best hyperparameters to file."""
    with open(output_path, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"{model_type} - Best Hyperparameters\n")
        f.write(f"{'=' * 60}\n\n")

        for k, v in params.items():
            f.write(f"  {k}: {v}\n")

    # Also save as JSON for programmatic access
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Saved hyperparameters: {output_path}")


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
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Weight for KL divergence (lower default to prevent explosion)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--output_dir', type=str, default='./model_outputs',
                       help='Output directory for results')

    # Cross-validation arguments
    parser.add_argument('--run_cv', action='store_true',
                       help='Run cross-validation')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')

    # Hyperparameter optimization arguments
    parser.add_argument('--run_hyperopt', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--max_evals', type=int, default=5 ,
                       help='Maximum number of hyperopt evaluations')

    # Test set evaluation
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for independent test set')

    args = parser.parse_args()

    # Extract base name from input file for output naming
    input_basename = os.path.splitext(os.path.basename(args.snp_data_loc))[0]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    cv_dir = os.path.join(args.output_dir, 'cv')
    hyperopt_dir = os.path.join(args.output_dir, 'hyperopt')
    test_dir = os.path.join(args.output_dir, 'independent_test')
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(hyperopt_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Semi-supervised and Joint VAE Models")
    print(f"{'=' * 60}")
    print(f"Model type: {args.model_type}")
    print(f"SNP data: {args.snp_data_loc}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Run CV: {args.run_cv} (folds: {args.n_folds})")
    print(f"Run Hyperopt: {args.run_hyperopt} (max_evals: {args.max_evals})")
    print(f"{'=' * 60}\n")

    # Load data
    print("Loading data...")
    data = pd.read_csv(args.snp_data_loc, sep='\\s+')

    # Extract phenotype and genotype data
    phenotype = data['PHENOTYPE'].values
    genotype_data = data.iloc[:, 6:].values

    # Convert labels: 1 (control) -> 0, 2 (case) -> 1
    labels = np.where(phenotype == 1, 0, 1)

    # Split data: hold out independent test set first
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        genotype_data, labels, test_size=args.test_size, random_state=42, stratify=labels
    )

    # Normalize using StandardScaler
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train/Val samples: {X_trainval.shape[0]}")
    print(f"Independent Test samples: {X_test.shape[0]}")
    print(f"Features: {X_trainval.shape[1]}")
    print(f"Class distribution - Train/Val: {np.bincount(y_trainval)}, Test: {np.bincount(y_test)}")

    input_dim = X_trainval.shape[1]

    # Define base params
    base_params = {
        'latent_dim': args.latent_dim,
        'num_hidden_layers_encoder': 2,
        'num_hidden_layers_decoder': 2,
        'encoding_dimensions': 256,
        'decoding_dimensions': 256,
        'classifier_hidden_dim': 128,
        'classifier_hidden_dims': [128, 64],
        'activation': 'relu',
        'learning_rate': args.learning_rate,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma
    }

    # ============================================================================
    # SSVAE Training Pipeline
    # ============================================================================
    if args.model_type in ['ssvae', 'both']:
        print(f"\n{'=' * 60}")
        print("Training Semi-supervised VAE (SSVAE)")
        print(f"{'=' * 60}\n")

        best_params_ssvae = base_params.copy()

        # Hyperparameter optimization
        if args.run_hyperopt:
            X_train_hp, X_val_hp, y_train_hp, y_val_hp = train_test_split(
                X_trainval_scaled, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
            )
            best_params_ssvae, trials = run_hyperparameter_optimization(
                X_train_hp, y_train_hp, X_val_hp, y_val_hp,
                model_type='ssvae', input_dim=input_dim,
                max_evals=args.max_evals, epochs=args.epochs // 2, batch_size=args.batch_size
            )
            # Merge with base params
            base_params.update(best_params_ssvae)
            best_params_ssvae = base_params.copy()

            save_hyperparameters(
                best_params_ssvae,
                os.path.join(hyperopt_dir, f'{input_basename}_ssvae_best_hyperparameters.txt'),
                'SSVAE'
            )

        # Cross-validation
        if args.run_cv:
            cv_results_ssvae, fold_predictions = run_cross_validation(
                X_trainval_scaled, y_trainval,
                model_type='ssvae', input_dim=input_dim,
                params=best_params_ssvae,
                n_folds=args.n_folds, epochs=args.epochs, batch_size=args.batch_size
            )
            save_cv_results(
                cv_results_ssvae,
                os.path.join(cv_dir, f'{input_basename}_ssvae_cv_results.txt'),
                'SSVAE',
                params=best_params_ssvae
            )
            # Save CV results as JSON
            with open(os.path.join(cv_dir, f'{input_basename}_ssvae_cv_results.json'), 'w') as f:
                # Remove non-serializable items
                cv_save = {k: v for k, v in cv_results_ssvae.items() if k not in ['fold_metrics', 'fold_histories']}
                json.dump(cv_save, f, indent=2)

        # Train final model on all train/val data and evaluate on independent test set
        print(f"\n--- Training Final SSVAE Model ---")
        ssvae_model = create_ssvae_model(
            input_dim=input_dim,
            latent_dim=best_params_ssvae.get('latent_dim', args.latent_dim),
            num_hidden_layers_encoder=best_params_ssvae.get('num_hidden_layers_encoder', 2),
            num_hidden_layers_decoder=best_params_ssvae.get('num_hidden_layers_decoder', 2),
            encoding_dimensions=best_params_ssvae.get('encoding_dimensions', 256),
            decoding_dimensions=best_params_ssvae.get('decoding_dimensions', 256),
            classifier_hidden_dim=best_params_ssvae.get('classifier_hidden_dim', 128),
            activation=best_params_ssvae.get('activation', 'relu'),
            learning_rate=best_params_ssvae.get('learning_rate', args.learning_rate),
            alpha=best_params_ssvae.get('alpha', args.alpha),
            beta=best_params_ssvae.get('beta', args.beta),
            gamma=best_params_ssvae.get('gamma', args.gamma)
        )

        # Train with enhanced callbacks
        history_ssvae = ssvae_model.fit(
            X_trainval_scaled, y_trainval,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    min_delta=1e-4,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
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
            os.path.join(args.output_dir, f'{input_basename}_ssvae_learning_curves.png'),
            'SSVAE'
        )

        # Evaluate on independent test set
        ssvae_metrics, ssvae_pred_proba, ssvae_latent = evaluate_model_comprehensive(
            ssvae_model, X_test_scaled, y_test, 'SSVAE'
        )
        print_comprehensive_metrics(ssvae_metrics, 'SSVAE (Independent Test)')
        save_independent_test_results(
            ssvae_metrics,
            os.path.join(test_dir, f'{input_basename}_ssvae_test_results.txt'),
            'SSVAE',
            params=best_params_ssvae
        )

        # Save predictions
        np.savez(
            os.path.join(test_dir, f'{input_basename}_ssvae_test_predictions.npz'),
            y_true=y_test,
            y_pred_proba=ssvae_pred_proba,
            latent_space=ssvae_latent
        )

        # Save model and metrics
        ssvae_model.save(os.path.join(args.output_dir, f'{input_basename}_ssvae_model.keras'))
        with open(os.path.join(test_dir, f'{input_basename}_ssvae_test_metrics.json'), 'w') as f:
            json.dump(ssvae_metrics, f, indent=2)

        # Save model parameters
        save_model_parameters(
            best_params_ssvae,
            os.path.join(args.output_dir, f'{input_basename}_ssvae_model_parameters.txt'),
            'SSVAE'
        )

        # Plot model architecture if available
        plot_model_architecture(
            ssvae_model,
            os.path.join(args.output_dir, f'{input_basename}_ssvae_model_architecture.png'),
            'SSVAE'
        )

    # ============================================================================
    # Joint VAE + Classifier Training Pipeline
    # ============================================================================
    if args.model_type in ['joint', 'both']:
        print(f"\n{'=' * 60}")
        print("Training Joint VAE + Classifier")
        print(f"{'=' * 60}\n")

        best_params_joint = base_params.copy()

        # Hyperparameter optimization
        if args.run_hyperopt:
            X_train_hp, X_val_hp, y_train_hp, y_val_hp = train_test_split(
                X_trainval_scaled, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
            )
            best_params_joint, trials = run_hyperparameter_optimization(
                X_train_hp, y_train_hp, X_val_hp, y_val_hp,
                model_type='joint', input_dim=input_dim,
                max_evals=args.max_evals, epochs=args.epochs // 2, batch_size=args.batch_size
            )
            base_params.update(best_params_joint)
            best_params_joint = base_params.copy()

            save_hyperparameters(
                best_params_joint,
                os.path.join(hyperopt_dir, f'{input_basename}_joint_best_hyperparameters.txt'),
                'Joint VAE+Classifier'
            )

        # Cross-validation
        if args.run_cv:
            cv_results_joint, fold_predictions = run_cross_validation(
                X_trainval_scaled, y_trainval,
                model_type='joint', input_dim=input_dim,
                params=best_params_joint,
                n_folds=args.n_folds, epochs=args.epochs, batch_size=args.batch_size
            )
            save_cv_results(
                cv_results_joint,
                os.path.join(cv_dir, f'{input_basename}_joint_cv_results.txt'),
                'Joint VAE+Classifier',
                params=best_params_joint
            )
            with open(os.path.join(cv_dir, f'{input_basename}_joint_cv_results.json'), 'w') as f:
                cv_save = {k: v for k, v in cv_results_joint.items() if k not in ['fold_metrics', 'fold_histories']}
                json.dump(cv_save, f, indent=2)

        # Train final model
        print(f"\n--- Training Final Joint Model ---")
        joint_model = create_joint_vae_classifier(
            input_dim=input_dim,
            latent_dim=best_params_joint.get('latent_dim', args.latent_dim),
            num_hidden_layers_encoder=best_params_joint.get('num_hidden_layers_encoder', 2),
            num_hidden_layers_decoder=best_params_joint.get('num_hidden_layers_decoder', 2),
            encoding_dimensions=best_params_joint.get('encoding_dimensions', 256),
            decoding_dimensions=best_params_joint.get('decoding_dimensions', 256),
            classifier_hidden_dims=best_params_joint.get('classifier_hidden_dims', [128, 64]),
            activation=best_params_joint.get('activation', 'relu'),
            learning_rate=best_params_joint.get('learning_rate', args.learning_rate),
            alpha=best_params_joint.get('alpha', args.alpha),
            beta=best_params_joint.get('beta', args.beta),
            gamma=best_params_joint.get('gamma', args.gamma)
        )

        # Train with enhanced callbacks
        history_joint = joint_model.fit(
            X_trainval_scaled, y_trainval,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    min_delta=1e-4,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
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
            os.path.join(args.output_dir, f'{input_basename}_joint_learning_curves.png'),
            'Joint VAE+Classifier'
        )

        # Evaluate on independent test set
        joint_metrics, joint_pred_proba, joint_latent = evaluate_model_comprehensive(
            joint_model, X_test_scaled, y_test, 'Joint'
        )
        print_comprehensive_metrics(joint_metrics, 'Joint VAE+Classifier (Independent Test)')
        save_independent_test_results(
            joint_metrics,
            os.path.join(test_dir, f'{input_basename}_joint_test_results.txt'),
            'Joint VAE+Classifier',
            params=best_params_joint
        )

        # Save predictions
        np.savez(
            os.path.join(test_dir, f'{input_basename}_joint_test_predictions.npz'),
            y_true=y_test,
            y_pred_proba=joint_pred_proba,
            latent_space=joint_latent
        )

        # Save model and metrics
        joint_model.save(os.path.join(args.output_dir, f'{input_basename}_joint_model.keras'))
        with open(os.path.join(test_dir, f'{input_basename}_joint_test_metrics.json'), 'w') as f:
            json.dump(joint_metrics, f, indent=2)

        # Save model parameters
        save_model_parameters(
            best_params_joint,
            os.path.join(args.output_dir, f'{input_basename}_joint_model_parameters.txt'),
            'Joint VAE+Classifier'
        )

        # Plot model architecture if available
        plot_model_architecture(
            joint_model,
            os.path.join(args.output_dir, f'{input_basename}_joint_model_architecture.png'),
            'Joint VAE+Classifier'
        )

    print(f"\n{'=' * 60}")
    print(f"Training complete. Results saved to: {args.output_dir}")
    print(f"  - CV results: {cv_dir}")
    print(f"  - Hyperopt results: {hyperopt_dir}")
    print(f"  - Independent test results: {test_dir}")
    print(f"  - Model parameters: {args.output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

