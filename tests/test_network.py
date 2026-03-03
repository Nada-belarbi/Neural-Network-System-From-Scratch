"""
Unit tests for the NeuralNetwork class.
"""

import unittest
import numpy as np
import tempfile
import os
from src.models import NeuralNetwork, Sigmoid, ReLU
from src.data import DataGenerator, DataLoader


class TestNeuralNetwork(unittest.TestCase):
    """Test cases for NeuralNetwork class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = NeuralNetwork(input_size=3)
        self.network.add_layer(4, Sigmoid())
        self.network.add_layer(2, Sigmoid())
    
    def test_initialization(self):
        """Test network initialization."""
        network = NeuralNetwork(input_size=5)
        self.assertEqual(network.input_size, 5)
        self.assertEqual(len(network.layers), 0)
        self.assertEqual(network.output_size, 0)
    
    def test_add_layer(self):
        """Test adding layers to network."""
        network = NeuralNetwork(input_size=3)
        
        # Add first layer
        network.add_layer(5, Sigmoid())
        self.assertEqual(len(network.layers), 1)
        self.assertEqual(network.layers[0].num_neurons, 5)
        self.assertEqual(network.layers[0].num_inputs, 3)
        self.assertEqual(network.output_size, 5)
        
        # Add second layer
        network.add_layer(2, ReLU())
        self.assertEqual(len(network.layers), 2)
        self.assertEqual(network.layers[1].num_neurons, 2)
        self.assertEqual(network.layers[1].num_inputs, 5)
        self.assertEqual(network.output_size, 2)
    
    def test_forward_propagation(self):
        """Test forward propagation through network."""
        inputs = np.array([0.5, -0.3, 0.8])
        outputs = self.network.forward(inputs)
        
        # Check output shape
        self.assertEqual(len(outputs), 2)  # Final layer has 2 neurons
        
        # Check outputs are valid
        for output in outputs:
            self.assertIsInstance(output, (float, np.floating))
    
    def test_invalid_input_size(self):
        """Test error handling for invalid input size."""
        with self.assertRaises(ValueError):
            self.network.forward(np.array([0.5, 0.3]))  # Wrong size
    
    def test_training(self):
        """Test network training."""
        # Generate simple dataset
        features, targets = DataGenerator.generate_classification_data(
            n_samples=50,
            n_features=3,
            n_classes=2,
            noise=0.1
        )
        dataset = DataLoader.create_dataset(features, targets)
        
        # Train network
        history = self.network.train(
            dataset,
            epochs=10,
            learning_rate=0.1
        )
        
        # Check training history
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertIn('epochs', history)
        self.assertEqual(len(history['loss']), 10)
        self.assertEqual(history['epochs'], 10)
        
        # Loss should generally decrease
        self.assertLess(history['loss'][-1], history['loss'][0])
    
    def test_prediction(self):
        """Test network prediction."""
        inputs = np.array([0.5, -0.3, 0.8])
        outputs = self.network.predict(inputs)
        
        # Should be same as forward
        outputs2 = self.network.forward(inputs)
        np.testing.assert_array_equal(outputs, outputs2)
    
    def test_evaluation(self):
        """Test network evaluation."""
        # Generate test data
        features, targets = DataGenerator.generate_classification_data(
            n_samples=20,
            n_features=3,
            n_classes=2,
            noise=0.1
        )
        test_data = DataLoader.create_dataset(features, targets)
        
        # Evaluate
        loss, accuracy = self.network.evaluate(test_data)
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_save_load(self):
        """Test saving and loading network."""
        # Train network briefly
        features, targets = DataGenerator.generate_classification_data(
            n_samples=20,
            n_features=3,
            n_classes=2
        )
        dataset = DataLoader.create_dataset(features, targets)
        self.network.train(dataset, epochs=5, learning_rate=0.1)
        
        # Save network
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.network.save(temp_file)
            
            # Load into new network
            new_network = NeuralNetwork(1)  # Dummy size
            new_network.load(temp_file)
            
            # Compare architectures
            self.assertEqual(
                self.network.get_architecture(),
                new_network.get_architecture()
            )
            
            # Compare predictions
            test_input = np.array([0.5, -0.3, 0.8])
            output1 = self.network.predict(test_input)
            output2 = new_network.predict(test_input)
            np.testing.assert_array_almost_equal(output1, output2)
            
        finally:
            os.unlink(temp_file)
    
    def test_get_architecture(self):
        """Test getting network architecture."""
        architecture = self.network.get_architecture()
        self.assertEqual(architecture, [3, 4, 2])
    
    def test_training_observer(self):
        """Test training observer functionality."""
        observed_epochs = []
        
        def observer(epoch, loss, accuracy):
            observed_epochs.append(epoch)
        
        self.network.add_training_observer(observer)
        
        # Generate small dataset
        features, targets = DataGenerator.generate_classification_data(
            n_samples=10,
            n_features=3,
            n_classes=2
        )
        dataset = DataLoader.create_dataset(features, targets)
        
        # Train
        self.network.train(dataset, epochs=5, learning_rate=0.1)
        
        # Check observer was called
        self.assertEqual(len(observed_epochs), 5)
        self.assertEqual(observed_epochs, [1, 2, 3, 4, 5])
    
    def test_empty_network_error(self):
        """Test error when training empty network."""
        empty_network = NeuralNetwork(3)
        
        with self.assertRaises(RuntimeError):
            empty_network.train([], epochs=1)
    
    def test_batch_training(self):
        """Test mini-batch training."""
        # Generate dataset
        features, targets = DataGenerator.generate_classification_data(
            n_samples=100,
            n_features=3,
            n_classes=2
        )
        dataset = DataLoader.create_dataset(features, targets)
        
        # Train with batch size
        history = self.network.train(
            dataset,
            epochs=5,
            learning_rate=0.1,
            batch_size=10
        )
        
        # Should complete successfully
        self.assertEqual(len(history['loss']), 5)


if __name__ == '__main__':
    unittest.main()