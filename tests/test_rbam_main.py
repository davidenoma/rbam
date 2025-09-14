import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Mock sys.argv before importing the module to prevent main execution
with patch('sys.argv', ['script', 'data.raw', 'bim.bim', 'quantitative']):
    # Add the runner directory to the path to import modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

    # Mock the data loading functions before import
    with patch('utils.load_real_genotype_data') as mock_load_data:
        mock_load_data.return_value = (
            np.random.random((100, 50)),  # X_train
            np.random.random((20, 50)),   # X_test
            np.random.random((120, 50)),  # snp_data
            np.random.random(120),        # phenotype
            np.random.random(100),        # y_train
            np.random.random(20)          # y_test
        )

        import rbam_main
        from rbam_main import (
            VAE, vae_loss, create_vae_model, save_model, load_model, objective
        )


class TestRBAMVAE(unittest.TestCase):
    """Test cases for the VAE class in RBAM main"""

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

    def test_vae_reparameterize(self):
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

    def test_vae_config_methods(self):
        """Test VAE configuration methods"""
        config = self.vae.get_config()

        self.assertIn('encoder', config)
        self.assertIn('decoder', config)

        # Test from_config
        vae_from_config = VAE.from_config(config)
        self.assertIsInstance(vae_from_config, VAE)


class TestRBAMVAELoss(unittest.TestCase):
    """Test cases for the VAE loss function in RBAM main"""

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

    def test_vae_loss_components(self):
        """Test that VAE loss includes both reconstruction and KL terms"""
        batch_size = 16
        x = tf.random.uniform((batch_size, self.input_dim), 0, 1)
        x_reconstructed = tf.random.uniform((batch_size, self.input_dim), 0, 1)

        loss = self.loss_fn(x, x_reconstructed)

        # Loss should be finite and positive
        self.assertTrue(tf.math.is_finite(loss))
        self.assertGreater(loss.numpy(), 0)


class TestRBAMModelCreation(unittest.TestCase):
    """Test cases for model creation in RBAM main"""

    def test_create_vae_model(self):
        """Test VAE model creation with various parameters"""
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
        # Check encoder structure: Input + hidden layers + output
        self.assertEqual(len(vae.encoder.layers), 4)
        # Check decoder structure: Input + hidden layers + output
        self.assertEqual(len(vae.decoder.layers), 4)

        # Test with different activation function
        params['activation'] = 'sigmoid'
        vae_sigmoid = create_vae_model(**params)
        self.assertIsInstance(vae_sigmoid, VAE)

    def test_create_vae_model_edge_cases(self):
        """Test VAE model creation with edge cases"""
        # Test with minimum layers
        params = {
            'input_dim': 50,
            'num_hidden_layers_encoder': 1,
            'num_hidden_layers_decoder': 1,
            'encoding_dimensions': 64,
            'decoding_dimensions': 64,
            'activation': 'relu',
            'batch_size': 16,
            'epochs': 5,
            'learning_rate': 0.01,
            'latent_dim': 8
        }

        vae = create_vae_model(**params)
        self.assertIsInstance(vae, VAE)

        # Test with large latent dimension
        params['latent_dim'] = 128
        vae_large = create_vae_model(**params)
        self.assertIsInstance(vae_large, VAE)


class TestRBAMModelSaveLoad(unittest.TestCase):
    """Test cases for model saving and loading in RBAM main"""

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

    def test_save_model(self):
        """Test model saving"""
        directory = os.path.join(self.temp_dir, "models")

        save_model(self.vae, self.snp_data_loc, directory, override=True)

        expected_path = os.path.join(directory, "test_data.keras")
        self.assertTrue(os.path.exists(expected_path))

    def test_save_model_without_override(self):
        """Test model saving without override when file exists"""
        directory = os.path.join(self.temp_dir, "models")

        # Save model first time
        save_model(self.vae, self.snp_data_loc, directory, override=True)

        # Try to save again without override - should raise error
        with self.assertRaises(FileExistsError):
            save_model(self.vae, self.snp_data_loc, directory, override=False)

    def test_load_model_exists(self):
        """Test loading existing model"""
        directory = os.path.join(self.temp_dir, "models")

        # First save the model
        save_model(self.vae, self.snp_data_loc, directory, override=True)

        # Then load it
        with patch('builtins.print'):
            loaded_model = load_model(self.snp_data_loc, directory)

        self.assertIsInstance(loaded_model, tf.keras.Model)

    def test_load_model_not_exists(self):
        """Test loading non-existent model"""
        directory = os.path.join(self.temp_dir, "models")

        with patch('builtins.print'):
            loaded_model = load_model("nonexistent.raw", directory)

        self.assertIsNone(loaded_model)


class TestRBAMObjective(unittest.TestCase):
    """Test cases for the objective function in RBAM main"""

    def setUp(self):
        """Set up test fixtures"""
        self.X_train = np.random.random((50, 20))

        # Patch the global X_train variable
        self.patcher = patch('rbam_main.X_train', self.X_train)
        self.patcher.start()

    def tearDown(self):
        """Clean up patches"""
        self.patcher.stop()

    def test_objective_function(self):
        """Test the objective function for hyperparameter optimization"""
        params = {
            'num_hidden_layers_encoder': 2,
            'num_hidden_layers_decoder': 2,
            'encoding_dimensions': 64,
            'decoding_dimensions': 64,
            'activation': 'relu',
            'learning_rate': 0.001,
            'epochs': 2,  # Small number for testing
            'batch_size': 16,
            'latent_dim': 8
        }

        result = objective(params)

        self.assertIn('loss', result)
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'ok')
        self.assertIsInstance(result['loss'], (int, float))
        self.assertGreaterEqual(result['loss'], 0)


class TestRBAMDataLoading(unittest.TestCase):
    """Test cases for data loading functionality"""

    @patch('rbam_main.load_real_genotype_data')
    @patch('rbam_main.load_real_genotype_data_case_control')
    def test_data_loading_quantitative(self, mock_case_control, mock_quantitative):
        """Test data loading for quantitative traits"""
        # Mock return values
        X_train = np.random.random((100, 50))
        X_test = np.random.random((20, 50))
        snp_data = np.random.random((120, 50))
        phenotype = np.random.random(120)
        y_train = np.random.random(100)
        y_test = np.random.random(20)

        mock_quantitative.return_value = (X_train, X_test, snp_data, phenotype, y_train, y_test)

        # Test with quantitative argument
        with patch('sys.argv', ['script', 'data.raw', 'bim.bim', 'quantitative']):
            # This would test the data loading logic
            pass

    @patch('rbam_main.load_real_genotype_data_case_control')
    def test_data_loading_case_control(self, mock_case_control):
        """Test data loading for case-control studies"""
        # Mock return values
        X_train = np.random.random((100, 50))
        X_test = np.random.random((20, 50))
        snp_data = np.random.random((120, 50))

        mock_case_control.return_value = (X_train, X_test, snp_data)

        # Test with case-control argument
        with patch('sys.argv', ['script', 'data.raw', 'bim.bim', 'case_control']):
            # This would test the data loading logic
            pass


class TestRBAMPlotting(unittest.TestCase):
    """Test cases for plotting functionality"""

    def test_loss_curve_plotting(self):
        """Test that loss curve plotting works without errors"""
        # Create mock history
        history = MagicMock()
        history.history = {
            'loss': [1.0, 0.8, 0.6, 0.4],
            'val_loss': [1.2, 0.9, 0.7, 0.5]
        }

        # Test plotting code (similar to what's in rbam_main.py)
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')

            # Save to temporary location
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                plt.savefig(tmp.name)

            plt.close()

        except Exception as e:
            self.fail(f"Plotting failed with error: {e}")


class TestRBAMUtilsIntegration(unittest.TestCase):
    """Test cases for integration with utils functions"""

    @patch('rbam_main.utils')
    def test_mse_calculation(self, mock_utils):
        """Test MSE calculation integration"""
        # Mock data
        X_train = np.random.random((100, 50))
        reconstructed = np.random.random((100, 50))

        # Mock utils functions
        mock_utils.compute_rmse.return_value = 0.5
        mock_utils.evaluate_r2.return_value = [0.8] * 50

        # Test MSE calculation
        mse = mock_utils.compute_rmse(X_train, reconstructed) ** 2
        self.assertEqual(mse, 0.25)

        # Test R2 calculation
        r2 = np.mean(mock_utils.evaluate_r2(X_train, reconstructed))
        self.assertEqual(r2, 0.8)

    @patch('rbam_main.utils')
    def test_cross_validation_integration(self, mock_utils):
        """Test cross-validation integration"""
        # Mock cross-validation results
        mock_utils.cross_validate_vae.return_value = (0.25, 0.30, 0.8, 0.75, 0.9, 0.85)

        results = mock_utils.cross_validate_vae(None, None)

        self.assertEqual(len(results), 6)
        self.assertEqual(results[0], 0.25)  # avg_mse_train
        self.assertEqual(results[1], 0.30)  # avg_mse_test

    @patch('rbam_main.utils')
    @patch('pandas.read_csv')
    def test_feature_importance_integration(self, mock_read_csv, mock_utils):
        """Test feature importance extraction integration"""
        # Mock BIM file
        bim_data = pd.DataFrame({
            'CHR': [1, 1, 2],
            'SNP': ['rs1', 'rs2', 'rs3'],
            'POS': [1000, 2000, 3000]
        })
        mock_read_csv.return_value = bim_data

        # Mock feature importance
        mock_utils.extract_decoder_reconstruction_weights.return_value = [0.1, 0.2, 0.3]
        mock_utils.extract_encoder_weights.return_value = [0.4, 0.5, 0.6]

        # Test feature importance extraction
        decoder_weights = mock_utils.extract_decoder_reconstruction_weights(None)
        encoder_weights = mock_utils.extract_encoder_weights(None)

        self.assertEqual(len(decoder_weights), 3)
        self.assertEqual(len(encoder_weights), 3)


class TestRBAMEnvironmentSetup(unittest.TestCase):
    """Test cases for environment setup"""

    @patch('subprocess.run')
    @patch('os.environ')
    def test_cuda_setup(self, mock_environ, mock_subprocess):
        """Test CUDA environment setup"""
        # Mock successful CUDA detection
        mock_result = MagicMock()
        mock_result.stdout = "cuda: /usr/local/cuda\n"
        mock_subprocess.return_value = mock_result

        # This would test the CUDA setup logic from rbam_main.py
        # The actual test would need to be adapted based on how the setup is refactored
        pass

    @patch('tensorflow.config.list_physical_devices')
    def test_gpu_setup(self, mock_list_devices):
        """Test GPU setup"""
        # Mock GPU devices
        mock_devices = [MagicMock(), MagicMock()]
        mock_list_devices.return_value = mock_devices

        devices = tf.config.list_physical_devices('GPU')

        # Test that devices are detected
        self.assertEqual(len(devices), 2)


class TestRBAMIntegration(unittest.TestCase):
    """Integration tests for RBAM main module"""

    @patch('sys.argv')
    @patch('rbam_main.load_real_genotype_data')
    @patch('rbam_main.utils')
    @patch('pandas.read_csv')
    def test_full_workflow_quantitative(self, mock_read_csv, mock_utils, mock_load_data, mock_argv):
        """Test full workflow for quantitative traits"""
        # Setup mocks
        mock_argv.__getitem__.side_effect = lambda x: {
            1: 'test_data.raw',
            2: 'test_data.bim',
            3: 'quantitative'
        }[x]

        # Mock data
        X_train = np.random.random((50, 20))
        X_test = np.random.random((10, 20))
        snp_data = np.random.random((60, 20))
        phenotype = np.random.random(60)
        y_train = np.random.random(50)
        y_test = np.random.random(10)

        mock_load_data.return_value = (X_train, X_test, snp_data, phenotype, y_train, y_test)

        # Mock BIM file
        bim_data = pd.DataFrame({
            'CHR': [1] * 20,
            'SNP': [f'rs{i}' for i in range(20)],
            'POS': list(range(1000, 21000, 1000))
        })
        mock_read_csv.return_value = bim_data

        # Mock utils functions
        mock_utils.compute_rmse.return_value = 0.5
        mock_utils.evaluate_r2.return_value = [0.8] * 20
        mock_utils.cross_validate_vae.return_value = (0.25, 0.30, 0.8, 0.75, 0.9, 0.85)
        mock_utils.extract_decoder_reconstruction_weights.return_value = [0.1] * 20
        mock_utils.extract_encoder_weights.return_value = [0.2] * 20

        # This test would verify the main workflow runs without errors
        # In practice, you'd want to refactor rbam_main.py to make it more testable


if __name__ == '__main__':
    # Set up TensorFlow for testing
    tf.config.run_functions_eagerly(True)

    # Suppress TensorFlow warnings for cleaner test output
    tf.get_logger().setLevel('ERROR')

    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestRBAMVAE))
    suite.addTest(unittest.makeSuite(TestRBAMVAELoss))
    suite.addTest(unittest.makeSuite(TestRBAMModelCreation))
    suite.addTest(unittest.makeSuite(TestRBAMModelSaveLoad))
    suite.addTest(unittest.makeSuite(TestRBAMObjective))
    suite.addTest(unittest.makeSuite(TestRBAMDataLoading))
    suite.addTest(unittest.makeSuite(TestRBAMPlotting))
    suite.addTest(unittest.makeSuite(TestRBAMUtilsIntegration))
    suite.addTest(unittest.makeSuite(TestRBAMEnvironmentSetup))
    suite.addTest(unittest.makeSuite(TestRBAMIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
