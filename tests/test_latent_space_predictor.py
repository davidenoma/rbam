import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
from sklearn.utils import class_weight

# Mock sys.argv before importing the module to prevent main execution
with patch('sys.argv', ['script_name', 'test_data.raw']):
    # Add the runner directory to the path to import modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

    # Mock the data loading functions before import
    with patch('utils.load_real_genotype_data') as mock_load_data:
        mock_load_data.return_value = (
            np.random.random((100, 50)),  # X_train
            np.random.random((20, 50)),   # X_test
            np.random.random((120, 50)),  # snp_data
            np.random.randint(1, 3, 120), # phenotype
            np.random.randint(1, 3, 100), # y_train
            np.random.randint(1, 3, 20)   # y_test
        )

        import latent_space_predictor
        from latent_space_predictor import (
            VAE, vae_loss, CustomEarlyStopping, create_vae_model,
            create_logistic_regression_model, create_random_forest_model,
            create_xgboost_model, create_tf_classifier_model,
            save_model, load_model, objective_classifier
        )


class TestVAE(unittest.TestCase):
    """Test cases for the VAE class"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 100
        self.latent_dim = 10

        # Create simple encoder and decoder for testing
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2 * self.latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.input_dim, activation='sigmoid')
        ])

        self.vae = VAE(encoder=self.encoder, decoder=self.decoder)

    def test_vae_initialization(self):
        """Test VAE initialization"""
        self.assertIsInstance(self.vae, VAE)
        self.assertEqual(self.vae.encoder, self.encoder)
        self.assertEqual(self.vae.decoder, self.decoder)

    def test_reparameterize(self):
        """Test the reparameterization trick"""
        batch_size = 32
        mean = tf.random.normal((batch_size, self.latent_dim))
        log_var = tf.random.normal((batch_size, self.latent_dim))

        z = self.vae.reparameterize(mean, log_var)

        self.assertEqual(z.shape, (batch_size, self.latent_dim))
        self.assertEqual(z.dtype, tf.float32)

    def test_vae_call(self):
        """Test VAE forward pass"""
        batch_size = 32
        inputs = tf.random.normal((batch_size, self.input_dim))

        outputs = self.vae(inputs)

        self.assertEqual(outputs.shape, (batch_size, self.input_dim))

    def test_vae_config_serialization(self):
        """Test VAE configuration serialization and deserialization"""
        config = self.vae.get_config()

        self.assertIn('encoder', config)
        self.assertIn('decoder', config)

        # Test from_config
        vae_from_config = VAE.from_config(config)
        self.assertIsInstance(vae_from_config, VAE)


class TestVAELoss(unittest.TestCase):
    """Test cases for the VAE loss function"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 100
        self.latent_dim = 10

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2 * self.latent_dim)
        ])

        self.loss_fn = vae_loss(self.encoder)

    def test_vae_loss_function(self):
        """Test VAE loss calculation"""
        batch_size = 32
        x = tf.random.uniform((batch_size, self.input_dim), 0, 1)
        x_reconstructed = tf.random.uniform((batch_size, self.input_dim), 0, 1)

        loss = self.loss_fn(x, x_reconstructed)

        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreaterEqual(loss.numpy(), 0)  # Loss should be non-negative


class TestCustomEarlyStopping(unittest.TestCase):
    """Test cases for CustomEarlyStopping callback"""

    def test_early_stopping_negative_loss(self):
        """Test early stopping when loss goes negative"""
        callback = CustomEarlyStopping()

        # Mock model
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.model = mock_model

        # Test with negative training loss
        logs = {'loss': -0.1, 'val_loss': 0.5}

        with patch('builtins.print'):
            callback.on_epoch_end(0, logs)

        self.assertTrue(mock_model.stop_training)

        # Reset and test with negative validation loss
        mock_model.stop_training = False
        logs = {'loss': 0.5, 'val_loss': -0.1}

        with patch('builtins.print'):
            callback.on_epoch_end(0, logs)

        self.assertTrue(mock_model.stop_training)


class TestModelCreation(unittest.TestCase):
    """Test cases for model creation functions"""

    def test_create_vae_model(self):
        """Test VAE model creation"""
        params = {
            'input_dim': 100,
            'num_hidden_layers_encoder': 2,
            'num_hidden_layers_decoder': 2,
            'encoding_dimensions': 128,
            'decoding_dimensions': 128,
            'activation': 'relu',
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'latent_dim': 16
        }

        vae = create_vae_model(**params)

        self.assertIsInstance(vae, VAE)
        self.assertEqual(len(vae.encoder.layers), 4)  # Input + 2 hidden + output
        self.assertEqual(len(vae.decoder.layers), 4)  # Input + 2 hidden + output

    def test_create_logistic_regression_model(self):
        """Test logistic regression model creation"""
        # Mock class weights
        class_weights_dict = {0: 1.0, 1: 1.0}

        with patch('latent_space_predictor.class_weights_dict', class_weights_dict):
            model = create_logistic_regression_model(C=1.0, penalty='l2')

            self.assertEqual(model.C, 1.0)
            self.assertEqual(model.penalty, 'l2')

    def test_create_random_forest_model(self):
        """Test random forest model creation"""
        class_weights_dict = {0: 1.0, 1: 1.0}

        with patch('latent_space_predictor.class_weights_dict', class_weights_dict):
            model = create_random_forest_model(n_estimators=100, max_depth=10)

            self.assertEqual(model.n_estimators, 100)
            self.assertEqual(model.max_depth, 10)

    def test_create_xgboost_model(self):
        """Test XGBoost model creation"""
        class_weights_dict = {0: 1.0, 1: 1.0}

        with patch('latent_space_predictor.class_weights_dict', class_weights_dict):
            model = create_xgboost_model(learning_rate=0.1, n_estimators=100, max_depth=6)

            self.assertEqual(model.learning_rate, 0.1)
            self.assertEqual(model.n_estimators, 100)
            self.assertEqual(model.max_depth, 6)

    def test_create_tf_classifier_model(self):
        """Test TensorFlow classifier model creation"""
        model = create_tf_classifier_model(
            input_dim=50,
            classifier_hidden_dim=128,
            activation='relu',
            learning_rate=0.001,
            batch_size=32,
            epochs=10
        )

        self.assertIsInstance(model, tf.keras.Sequential)
        self.assertEqual(len(model.layers), 3)  # Input + 2 hidden + output


class TestModelSaveLoad(unittest.TestCase):
    """Test cases for model saving and loading"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.snp_data_loc = os.path.join(self.temp_dir, "test_data.raw")

        # Create a simple VAE for testing
        encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(20)
        ])

        decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(100, activation='sigmoid')
        ])

        self.vae = VAE(encoder=encoder, decoder=decoder)
        loss_function = vae_loss(self.vae.encoder)
        self.vae.compile(optimizer='adam', loss=loss_function)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('os.getcwd')
    def test_save_model(self, mock_getcwd):
        """Test model saving"""
        mock_getcwd.return_value = self.temp_dir

        save_model(self.vae, self.snp_data_loc, override=True)

        expected_path = os.path.join(self.temp_dir, "model_cc_com_qt", "test_data.keras")
        self.assertTrue(os.path.exists(expected_path))

    @patch('os.getcwd')
    def test_load_model(self, mock_getcwd):
        """Test model loading"""
        mock_getcwd.return_value = self.temp_dir

        # First save the model
        save_model(self.vae, self.snp_data_loc, override=True)

        # Then load it
        loaded_model = load_model(self.snp_data_loc)

        self.assertIsInstance(loaded_model, tf.keras.Model)

    @patch('os.getcwd')
    def test_load_model_not_exists(self, mock_getcwd):
        """Test loading non-existent model"""
        mock_getcwd.return_value = self.temp_dir

        loaded_model = load_model("nonexistent.raw")

        self.assertIsNone(loaded_model)


class TestObjectiveClassifier(unittest.TestCase):
    """Test cases for classifier objective function"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock training data
        self.z_mean_train = np.random.random((100, 20))
        self.y_train = np.random.randint(0, 2, 100)

        # Patch global variables
        self.patcher1 = patch('latent_space_predictor.z_mean_train', self.z_mean_train)
        self.patcher2 = patch('latent_space_predictor.y_train', self.y_train)
        self.patcher3 = patch('latent_space_predictor.class_weights_dict', {0: 1.0, 1: 1.0})

        self.patcher1.start()
        self.patcher2.start()
        self.patcher3.start()

    def tearDown(self):
        """Clean up patches"""
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()

    def test_objective_classifier_tf(self):
        """Test objective function for TensorFlow classifier"""
        params = {
            'classifier_hidden_dim': 128,
            'activation': 'relu',
            'learning_rate': 0.001,
            'epochs': 5,
            'batch_size': 32
        }

        result = objective_classifier(params, 'tf_classifier')

        self.assertIn('loss', result)
        self.assertIn('status', result)
        self.assertGreaterEqual(result['loss'], 0)

    def test_objective_classifier_logistic_regression(self):
        """Test objective function for logistic regression"""
        params = {
            'C': 1.0,
            'penalty': 'l2'
        }

        result = objective_classifier(params, 'logistic_regression')

        self.assertIn('loss', result)
        self.assertIn('status', result)
        self.assertGreaterEqual(result['loss'], 0)
        self.assertLessEqual(result['loss'], 1.0)

    def test_objective_classifier_random_forest(self):
        """Test objective function for random forest"""
        params = {
            'n_estimators': 50,
            'max_depth': 10
        }

        result = objective_classifier(params, 'random_forest')

        self.assertIn('loss', result)
        self.assertIn('status', result)
        self.assertGreaterEqual(result['loss'], 0)
        self.assertLessEqual(result['loss'], 1.0)

    def test_objective_classifier_xgboost(self):
        """Test objective function for XGBoost"""
        params = {
            'learning_rate': 0.1,
            'n_estimators': 50,
            'max_depth': 6
        }

        result = objective_classifier(params, 'xgboost')

        self.assertIn('loss', result)
        self.assertIn('status', result)
        self.assertGreaterEqual(result['loss'], 0)
        self.assertLessEqual(result['loss'], 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the latent space predictor module"""

    @patch('sys.argv', ['script_name', 'test_data.raw'])
    @patch('latent_space_predictor.load_real_genotype_data')
    @patch('latent_space_predictor.utils')
    def test_main_workflow_mock(self, mock_utils, mock_load_data):
        """Test the main workflow with mocked data"""
        # Mock data loading
        X_train = np.random.random((100, 50))
        X_test = np.random.random((20, 50))
        snp_data = np.random.random((120, 50))
        phenotype = np.random.randint(1, 3, 120)
        y_train = np.random.randint(1, 3, 100)
        y_test = np.random.randint(1, 3, 20)

        mock_load_data.return_value = (X_train, X_test, snp_data, phenotype, y_train, y_test)

        # Mock utility functions
        mock_utils.compute_rmse.return_value = 0.5
        mock_utils.evaluate_r2.return_value = [0.8] * 50
        mock_utils.cross_validate_vae.return_value = (0.25, 0.30, 0.8, 0.75, 0.9, 0.85)
        mock_utils.save_mse_values = MagicMock()
        mock_utils.save_r2_scores = MagicMock()
        mock_utils.save_mse_values_cv = MagicMock()
        mock_utils.save_r2_scores_cv = MagicMock()
        mock_utils.save_model = MagicMock()
        mock_utils.cross_validate_classifier.return_value = (0.85, 0.90)
        mock_utils.save_classifier_metrics = MagicMock()

        # This would test the main execution flow
        # In a real scenario, you might want to refactor the main script
        # to make it more testable by extracting the main logic into functions


if __name__ == '__main__':
    # Set up TensorFlow for testing
    tf.config.run_functions_eagerly(True)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestVAE))
    suite.addTest(unittest.makeSuite(TestVAELoss))
    suite.addTest(unittest.makeSuite(TestCustomEarlyStopping))
    suite.addTest(unittest.makeSuite(TestModelCreation))
    suite.addTest(unittest.makeSuite(TestModelSaveLoad))
    suite.addTest(unittest.makeSuite(TestObjectiveClassifier))
    suite.addTest(unittest.makeSuite(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
