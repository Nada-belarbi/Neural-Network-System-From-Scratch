"""
Unit tests for the Neuron class.
"""

import unittest
import numpy as np
from src.models import Neuron, Sigmoid, ReLU


class TestNeuron(unittest.TestCase):
    """Test cases for Neuron class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.neuron = Neuron(3, activation=Sigmoid())
    
    def test_initialization(self):
        """Test neuron initialization."""
        self.assertEqual(self.neuron.num_inputs, 3)
        self.assertIsInstance(self.neuron.activation, Sigmoid)
        self.assertEqual(len(self.neuron.weights), 3)
        self.assertIsNotNone(self.neuron.bias)
    
    def test_forward_propagation(self):
        """Test forward propagation."""
        inputs = np.array([0.5, -0.3, 0.8])
        output = self.neuron.forward(inputs)
        
        # Output should be a scalar
        self.assertIsInstance(output, (float, np.floating))
        
        # Output should be between 0 and 1 for sigmoid
        self.assertGreaterEqual(output, 0)
        self.assertLessEqual(output, 1)
    
    def test_backward_propagation(self):
        """Test backward propagation."""
        # First do forward pass
        inputs = np.array([0.5, -0.3, 0.8])
        self.neuron.forward(inputs)
        
        # Then backward pass
        error = 0.1
        learning_rate = 0.1
        
        old_weights = self.neuron.weights.copy()
        old_bias = self.neuron.bias
        
        error_to_propagate = self.neuron.backward(error, learning_rate)
        
        # Check that weights were updated
        self.assertFalse(np.array_equal(old_weights, self.neuron.weights))
        self.assertNotEqual(old_bias, self.neuron.bias)
        
        # Check error propagation shape
        self.assertEqual(len(error_to_propagate), 3)
    
    def test_invalid_input_size(self):
        """Test error handling for invalid input size."""
        with self.assertRaises(ValueError):
            self.neuron.forward(np.array([0.5, 0.3]))  # Wrong size
    
    def test_custom_weights_and_bias(self):
        """Test initialization with custom weights and bias."""
        weights = np.array([0.1, 0.2, 0.3])
        bias = 0.5
        neuron = Neuron(3, weights=weights, bias=bias)
        
        np.testing.assert_array_equal(neuron.weights, weights)
        self.assertEqual(neuron.bias, bias)
    
    def test_different_activation_functions(self):
        """Test neuron with different activation functions."""
        # Test with ReLU
        neuron_relu = Neuron(2, activation=ReLU())
        inputs = np.array([-0.5, 0.5])
        output = neuron_relu.forward(inputs)
        
        # ReLU output should be non-negative
        self.assertGreaterEqual(output, 0)
    
    def test_get_set_parameters(self):
        """Test getting and setting parameters."""
        # Get parameters
        params = self.neuron.get_parameters()
        
        self.assertIn('weights', params)
        self.assertIn('bias', params)
        self.assertIn('activation', params)
        self.assertIn('num_inputs', params)
        
        # Set parameters
        new_params = {
            'weights': [0.1, 0.2, 0.3],
            'bias': 0.5
        }
        self.neuron.set_parameters(new_params)
        
        np.testing.assert_array_equal(self.neuron.weights, np.array([0.1, 0.2, 0.3]))
        self.assertEqual(self.neuron.bias, 0.5)


if __name__ == '__main__':
    unittest.main()