"""
Neural Network class implementation.
A network is composed of interconnected layers of neurons.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Callable
from .layer import Layer
from .activation_functions import ActivationFunction, Sigmoid
import json


class NeuralNetwork:
    """
    Neural network implementation with configurable architecture.
    
    The network consists of sequential layers where the output of each layer
    becomes the input to the next layer.
    
    Attributes:
        layers: List of layers in the network
        input_size: Number of inputs to the network
        output_size: Number of outputs from the network
        training_history: History of training metrics
    """
    
    def __init__(self, input_size: int):
        """
        Initialize a neural network.
        
        Args:
            input_size: Number of inputs to the network
        """
        if input_size <= 0:
            raise ValueError("Input size must be positive")
        
        self.input_size = input_size
        self.layers: List[Layer] = []
        self.output_size = 0
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'epochs': 0
        }
        
        # Observers for training progress
        self._training_observers: List[Callable] = []
    
    def add_layer(self, 
                  num_neurons: int, 
                  activation: Optional[ActivationFunction] = None) -> 'NeuralNetwork':
        """
        Add a layer to the network.
        
        Args:
            num_neurons: Number of neurons in the layer
            activation: Activation function (default: Sigmoid)
            
        Returns:
            Self for method chaining
        """
        # Determine input size for the new layer
        if not self.layers:
            # First layer connects to network inputs
            layer_inputs = self.input_size
        else:
            # Subsequent layers connect to previous layer outputs
            layer_inputs = self.layers[-1].get_output_size()
        
        # Create and add the layer
        layer = Layer(num_neurons, layer_inputs, activation)
        self.layers.append(layer)
        self.output_size = num_neurons
        
        return self
    
    def forward(self, inputs: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Perform forward propagation through the network.
        
        Args:
            inputs: Input values
            
        Returns:
            Network output values
        """
        inputs = np.array(inputs)
        
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Pass through each layer
        current_output = inputs
        for layer in self.layers:
            current_output = layer.forward(current_output)
        
        return current_output
    
    def backward(self, targets: np.ndarray, outputs: np.ndarray, learning_rate: float) -> float:
        """
        Perform backpropagation through the network.
        
        Args:
            targets: Target values
            outputs: Actual output values
            learning_rate: Learning rate for weight updates
            
        Returns:
            Loss value
        """
        # Calculate output layer errors (MSE derivative)
        errors = outputs - targets
        loss = np.mean(errors ** 2)
        
        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            errors = layer.backward(errors, learning_rate)
        
        return loss
    
    def train(self,
              training_data: List[Tuple[np.ndarray, np.ndarray]],
              epochs: int,
              learning_rate: float = 0.1,
              validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
              batch_size: Optional[int] = None) -> dict:
        """
        Train the network on the provided data.
        
        Args:
            training_data: List of (input, target) pairs
            epochs: Number of training epochs
            learning_rate: Learning rate for weight updates
            validation_data: Optional validation data
            batch_size: Batch size for mini-batch training (None for full batch)
            
        Returns:
            Training history dictionary
        """
        if not self.layers:
            raise RuntimeError("Network has no layers")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(training_data))
            
            # Process batches
            for i in range(0, len(training_data), batch_size or len(training_data)):
                batch_indices = indices[i:i + (batch_size or len(training_data))]
                batch_loss = 0.0
                
                for idx in batch_indices:
                    inputs, targets = training_data[idx]
                    
                    # Forward pass
                    outputs = self.forward(inputs)
                    
                    # Backward pass
                    loss = self.backward(targets, outputs, learning_rate)
                    batch_loss += loss
                    
                    # Track accuracy for classification
                    if len(outputs) > 1:  # Multi-class
                        if np.argmax(outputs) == np.argmax(targets):
                            correct_predictions += 1
                    else:  # Binary
                        if (outputs[0] > 0.5) == (targets[0] > 0.5):
                            correct_predictions += 1
                
                epoch_loss += batch_loss
            
            # Calculate metrics
            avg_loss = epoch_loss / len(training_data)
            accuracy = correct_predictions / len(training_data)
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['epochs'] = epoch + 1
            
            # Notify observers
            self._notify_observers(epoch + 1, avg_loss, accuracy)
            
            # Validation
            if validation_data and epoch % 10 == 0:
                val_loss, val_accuracy = self.evaluate(validation_data)
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return self.training_history
    
    def predict(self, inputs: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Make a prediction for the given inputs.
        
        Args:
            inputs: Input values
            
        Returns:
            Predicted output values
        """
        return self.forward(inputs)
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
        """
        Evaluate the network on test data.
        
        Args:
            test_data: List of (input, target) pairs
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        total_loss = 0.0
        correct_predictions = 0
        
        for inputs, targets in test_data:
            outputs = self.forward(inputs)
            
            # Calculate loss
            errors = outputs - targets
            loss = np.mean(errors ** 2)
            total_loss += loss
            
            # Calculate accuracy
            if len(outputs) > 1:  # Multi-class
                if np.argmax(outputs) == np.argmax(targets):
                    correct_predictions += 1
            else:  # Binary
                if (outputs[0] > 0.5) == (targets[0] > 0.5):
                    correct_predictions += 1
        
        avg_loss = total_loss / len(test_data)
        accuracy = correct_predictions / len(test_data)
        
        return avg_loss, accuracy
    
    def save(self, filepath: str) -> None:
        """
        Save the network to a file.
        
        Args:
            filepath: Path to save the network
        """
        network_data = {
            'input_size': self.input_size,
            'layers': [layer.get_parameters() for layer in self.layers],
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(network_data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """
        Load the network from a file.
        
        Args:
            filepath: Path to load the network from
        """
        with open(filepath, 'r') as f:
            network_data = json.load(f)
        
        # Clear existing layers
        self.layers = []
        self.input_size = network_data['input_size']
        
        # Recreate layers
        for layer_params in network_data['layers']:
            layer = Layer(
                layer_params['num_neurons'],
                layer_params['num_inputs'],
                activation=None  # Will use default
            )
            layer.set_parameters(layer_params)
            self.layers.append(layer)
        
        self.output_size = self.layers[-1].get_output_size() if self.layers else 0
        self.training_history = network_data.get('training_history', {
            'loss': [], 'accuracy': [], 'epochs': 0
        })
    
    def add_training_observer(self, observer: Callable) -> None:
        """
        Add an observer for training progress.
        
        Args:
            observer: Callable that receives (epoch, loss, accuracy)
        """
        self._training_observers.append(observer)
    
    def _notify_observers(self, epoch: int, loss: float, accuracy: float) -> None:
        """Notify all training observers."""
        for observer in self._training_observers:
            observer(epoch, loss, accuracy)
    
    def get_architecture(self) -> List[int]:
        """
        Get the network architecture as a list of layer sizes.
        
        Returns:
            List of neurons per layer (including input size)
        """
        architecture = [self.input_size]
        for layer in self.layers:
            architecture.append(layer.num_neurons)
        return architecture
    
    def __repr__(self) -> str:
        """String representation of the network."""
        architecture = self.get_architecture()
        return f"NeuralNetwork(architecture={architecture})"