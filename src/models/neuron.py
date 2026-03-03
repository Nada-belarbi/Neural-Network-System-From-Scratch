"""
Neuron class implementation.
A neuron is the basic unit of a neural network.
"""

import numpy as np
from typing import List, Optional, Union
from .activation_functions import ActivationFunction, Sigmoid


class Neuron:
    """
    Artificial neuron implementation.
    
    A neuron receives weighted inputs, adds a bias, applies an activation function,
    and produces an output.
    
    Attributes:
        weights: Weight coefficients for each input
        bias: Bias term added to the weighted sum
        activation: Activation function to apply
        output: Last computed output value
        input_cache: Cached input values for backpropagation
        weighted_sum_cache: Cached weighted sum for backpropagation
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 activation: Optional[ActivationFunction] = None,
                 weights: Optional[np.ndarray] = None,
                 bias: Optional[float] = None):
        """
        Initialize a neuron.
        
        Args:
            num_inputs: Number of input connections
            activation: Activation function (default: Sigmoid)
            weights: Initial weights (default: random initialization)
            bias: Initial bias (default: random initialization)
        """
        self.num_inputs = num_inputs
        self.activation = activation or Sigmoid()
        
        # Initialize weights and bias
        if weights is not None:
            if len(weights) != num_inputs:
                raise ValueError(f"Expected {num_inputs} weights, got {len(weights)}")
            self.weights = np.array(weights)
        else:
            # Xavier/He initialization
            self.weights = np.random.randn(num_inputs) * np.sqrt(2.0 / num_inputs)
        
        self.bias = bias if bias is not None else np.random.randn() * 0.01
        
        # Cache for backpropagation
        self.output: Optional[float] = None
        self.input_cache: Optional[np.ndarray] = None
        self.weighted_sum_cache: Optional[float] = None
        
        # Gradients
        self.weight_gradients: Optional[np.ndarray] = None
        self.bias_gradient: Optional[float] = None
    
    def forward(self, inputs: Union[List[float], np.ndarray]) -> float:
        """
        Perform forward propagation.
        
        Args:
            inputs: Input values
            
        Returns:
            Neuron output after activation
            
        Raises:
            ValueError: If input size doesn't match expected size
        """
        inputs = np.array(inputs)
        
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Cache inputs for backpropagation
        self.input_cache = inputs
        
        # Compute weighted sum: w1*x1 + w2*x2 + ... + wn*xn + bias
        self.weighted_sum_cache = np.dot(self.weights, inputs) + self.bias
        
        # Apply activation function
        self.output = self.activation.forward(self.weighted_sum_cache)
        
        return self.output
    
    def backward(self, error: float, learning_rate: float) -> np.ndarray:
        """
        Perform backpropagation and update weights.
        
        Args:
            error: Error signal from the next layer
            learning_rate: Learning rate for weight updates
            
        Returns:
            Error signals to propagate to previous layer
        """
        if self.input_cache is None or self.weighted_sum_cache is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Compute gradient of activation function
        activation_gradient = self.activation.derivative(self.weighted_sum_cache)
        
        # Compute delta (error * activation gradient)
        delta = error * activation_gradient
        
        # Compute weight gradients
        self.weight_gradients = delta * self.input_cache
        self.bias_gradient = delta
        
        # Update weights and bias
        self.weights -= learning_rate * self.weight_gradients
        self.bias -= learning_rate * self.bias_gradient
        
        # Compute error to propagate back
        error_to_propagate = delta * self.weights
        
        return error_to_propagate
    
    def get_parameters(self) -> dict:
        """
        Get neuron parameters.
        
        Returns:
            Dictionary containing weights, bias, and activation function
        """
        return {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'activation': str(self.activation),
            'num_inputs': self.num_inputs
        }
    
    def set_parameters(self, parameters: dict) -> None:
        """
        Set neuron parameters.
        
        Args:
            parameters: Dictionary containing weights and bias
        """
        if 'weights' in parameters:
            self.weights = np.array(parameters['weights'])
        if 'bias' in parameters:
            self.bias = float(parameters['bias'])
    
    def reset_gradients(self) -> None:
        """Reset gradient accumulators."""
        self.weight_gradients = None
        self.bias_gradient = None
    
    def __repr__(self) -> str:
        """String representation of the neuron."""
        return (f"Neuron(inputs={self.num_inputs}, "
                f"activation={self.activation}, "
                f"weights_shape={self.weights.shape})")