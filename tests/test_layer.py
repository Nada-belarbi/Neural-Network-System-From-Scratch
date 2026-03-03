"""
Unit tests for the Layer class.
"""

import unittest
import numpy as np
from src.models import Layer, Sigmoid, ReLU


class TestLayer(unittest.TestCase):
    """Test cases for Layer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.layer = Layer(num_neurons=4, num_inputs=3, activation=Sigmoid())
    
    def test_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.num_neurons, 4)
        self.assertEqual(self.layer.num_inputs, 3)
        self.assertIsInstance(self.layer.activation, Sigmoid)
        self.assertEqual(len(self.layer.neurons), 4)
        
        # Check each neuron
        for neuron in self.layer.neurons:
            self.assertEqual(neuron.num_inputs, 3)
            self.assertIsInstance(neuron.activation, Sigmoid)
    
    def test_forward_propagation(self):
        """Test forward propagation through layer."""
        inputs = np.array([0.5, -0.3, 0.8])
        outputs = self.layer.forward(inputs)
        
        # Check output shape
        self.assertEqual(len(outputs), 4)  # 4 neurons
        
        # Check output values are valid (between 0 and 1 for sigmoid)
        for output in outputs:
            self.assertGreaterEqual(output, 0)
            self.assertLessEqual(output, 1)
    
    def test_backward_propagation(self):
        """Test backward propagation through layer."""
        # Forward pass first
        inputs = np.array([0.5, -0.3, 0.8])
        self.layer.forward(inputs)
        
        # Backward pass
        errors = np.array([0.1, -0.2, 0.15, -0.05])  # One error per neuron
        learning_rate = 0.1
        
        propagated_errors = self.layer.backward(errors, learning_rate)
        
        # Check propagated errors shape
        self.assertEqual(len(propagated_errors), 3)  # Same as input size
    
    def test_invalid_initialization(self):
        """Test error handling for invalid initialization."""
        with self.assertRaises(ValueError):
            Layer(num_neurons=0, num_inputs=3)
        
        with self.assertRaises(ValueError):
            Layer(num_neurons=3, num_inputs=0)
    
    def test_weights_matrix_operations(self):
        """Test getting and setting weight matrices."""
        # Get weights matrix
        weights = self.layer.get_weights_matrix()
        self.assertEqual(weights.shape, (4, 3))  # 4 neurons, 3 inputs each
        
        # Set new weights
        new_weights = np.random.randn(4, 3)
        self.layer.set_weights_matrix(new_weights)
        
        # Verify weights were set
        retrieved_weights = self.layer.get_weights_matrix()
        np.testing.assert_array_equal(retrieved_weights, new_weights)
    
    def test_bias_operations(self):
        """Test getting and setting biases."""
        # Get biases
        biases = self.layer.get_biases()
        self.assertEqual(len(biases), 4)  # 4 neurons
        
        # Set new biases
        new_biases = np.array([0.1, 0.2, 0.3, 0.4])
        self.layer.set_biases(new_biases)
        
        # Verify biases were set
        retrieved_biases = self.layer.get_biases()
        np.testing.assert_array_equal(retrieved_biases, new_biases)
    
    def test_invalid_weights_shape(self):
        """Test error handling for invalid weight matrix shape."""
        with self.assertRaises(ValueError):
            self.layer.set_weights_matrix(np.random.randn(3, 3))  # Wrong shape
    
    def test_invalid_bias_shape(self):
        """Test error handling for invalid bias vector shape."""
        with self.assertRaises(ValueError):
            self.layer.set_biases(np.array([0.1, 0.2]))  # Wrong length
    
    def test_get_output_size(self):
        """Test getting output size."""
        self.assertEqual(self.layer.get_output_size(), 4)
    
    def test_different_activation_functions(self):
        """Test layer with different activation functions."""
        relu_layer = Layer(num_neurons=2, num_inputs=3, activation=ReLU())
        
        inputs = np.array([-0.5, 0.5, -0.3])
        outputs = relu_layer.forward(inputs)
        
        # All outputs should be non-negative with ReLU
        for output in outputs:
            self.assertGreaterEqual(output, 0)
    
    def test_parameters_persistence(self):
        """Test getting and setting layer parameters."""
        # Get parameters
        params = self.layer.get_parameters()
        
        self.assertIn('num_neurons', params)
        self.assertIn('num_inputs', params)
        self.assertIn('activation', params)
        self.assertIn('neurons', params)
        
        # Create new layer and set parameters
        new_layer = Layer(4, 3)
        new_layer.set_parameters(params)
        
        # Verify parameters were set correctly
        for i, neuron in enumerate(new_layer.neurons):
            original_params = params['neurons'][i]
            np.testing.assert_array_equal(
                neuron.weights,
                np.array(original_params['weights'])
            )


if __name__ == '__main__':
    unittest.main()