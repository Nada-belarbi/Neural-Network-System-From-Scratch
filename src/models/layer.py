"""
Layer class implementation.
A layer is a collection of neurons that process inputs in parallel.
"""

import numpy as np
from typing import List, Optional, Union
from .neuron import Neuron
from .activation_functions import ActivationFunction, Sigmoid


class Layer:
    """
    Neural network layer implementation.
    
    A layer contains multiple neurons that process inputs in parallel.
    All neurons in a layer share the same activation function.
    
    Attributes:
        neurons: List of neurons in the layer
        num_neurons: Number of neurons in the layer
        num_inputs: Number of inputs per neuron
        activation: Activation function used by all neurons
        outputs: Cached output values from forward pass
    """
    
    def __init__(self,
                 num_neurons: int,
                 num_inputs: int,
                 activation: Optional[ActivationFunction] = None):
        """
        Initialize a layer.
        
        Args:
            num_neurons: Number of neurons in the layer
            num_inputs: Number of inputs per neuron
            activation: Activation function (default: Sigmoid)
        """
        if num_neurons <= 0:
            raise ValueError("Number of neurons must be positive")
        if num_inputs <= 0:
            raise ValueError("Number of inputs must be positive")
        
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation = activation or Sigmoid()
        
        # Create neurons
        self.neurons = [
            Neuron(num_inputs, self.activation)
            for _ in range(num_neurons)
        ]
        
        # Cache for outputs
        self.outputs: Optional[np.ndarray] = None
    
    def forward(self, inputs: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Perform forward propagation through the layer.
        
        Args:
            inputs: Input values
            
        Returns:
            Array of outputs from all neurons
        """
        inputs = np.array(inputs)
        
        # Process inputs through each neuron
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        
        self.outputs = np.array(outputs)
        return self.outputs
    
    def backward(self, errors: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform backpropagation through the layer.
        
        Args:
            errors: Error signals from the next layer (one per neuron)
            learning_rate: Learning rate for weight updates
            
        Returns:
            Error signals to propagate to previous layer
        """
        if len(errors) != self.num_neurons:
            raise ValueError(f"Expected {self.num_neurons} errors, got {len(errors)}")
        
        # Collect error signals from each neuron
        propagated_errors = []
        for neuron, error in zip(self.neurons, errors):
            neuron_errors = neuron.backward(error, learning_rate)
            propagated_errors.append(neuron_errors)
        
        # Sum errors for each input
        propagated_errors = np.array(propagated_errors)
        summed_errors = np.sum(propagated_errors, axis=0)
        
        return summed_errors
    
    def get_weights_matrix(self) -> np.ndarray:
        """
        Get the weight matrix for the entire layer.
        
        Returns:
            2D array where each row contains weights for one neuron
        """
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.weights)
        return np.array(weights)
    
    def get_biases(self) -> np.ndarray:
        """
        Get the bias vector for the layer.
        
        Returns:
            1D array of biases
        """
        biases = []
        for neuron in self.neurons:
            biases.append(neuron.bias)
        return np.array(biases)
    
    def set_weights_matrix(self, weights: np.ndarray) -> None:
        """
        Set the weight matrix for the entire layer.
        
        Args:
            weights: 2D array where each row contains weights for one neuron
        """
        if weights.shape != (self.num_neurons, self.num_inputs):
            raise ValueError(f"Expected shape ({self.num_neurons}, {self.num_inputs}), "
                           f"got {weights.shape}")
        
        for i, neuron in enumerate(self.neurons):
            neuron.weights = weights[i].copy()
    
    def set_biases(self, biases: np.ndarray) -> None:
        """
        Set the bias vector for the layer.
        
        Args:
            biases: 1D array of biases
        """
        if len(biases) != self.num_neurons:
            raise ValueError(f"Expected {self.num_neurons} biases, got {len(biases)}")
        
        for neuron, bias in zip(self.neurons, biases):
            neuron.bias = float(bias)
    
    def get_parameters(self) -> dict:
        """
        Get layer parameters.
        
        Returns:
            Dictionary containing layer configuration and neuron parameters
        """
        return {
            'num_neurons': self.num_neurons,
            'num_inputs': self.num_inputs,
            'activation': str(self.activation),
            'neurons': [neuron.get_parameters() for neuron in self.neurons]
        }
    
    def set_parameters(self, parameters: dict) -> None:
        """
        Set layer parameters.
        
        Args:
            parameters: Dictionary containing neuron parameters
        """
        if 'neurons' in parameters:
            for i, neuron_params in enumerate(parameters['neurons']):
                if i < len(self.neurons):
                    self.neurons[i].set_parameters(neuron_params)
    
    def reset_gradients(self) -> None:
        """Reset gradient accumulators for all neurons."""
        for neuron in self.neurons:
            neuron.reset_gradients()
    
    def get_output_size(self) -> int:
        """Get the number of outputs from this layer."""
        return self.num_neurons
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return (f"Layer(neurons={self.num_neurons}, "
                f"inputs={self.num_inputs}, "
                f"activation={self.activation})")