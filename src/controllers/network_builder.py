"""
Network Builder implementation.
Implements the Builder pattern for creating neural networks.
"""

from typing import Optional, List, Union
from ..models import NeuralNetwork, ActivationFunction, Sigmoid, create_activation


class NetworkBuilder:
    """
    Builder class for constructing neural networks.
    
    Provides a fluent interface for creating networks with
    configurable architecture and parameters.
    """
    
    def __init__(self, input_size: Optional[int] = None):
        """
        Initialize the network builder.
        
        Args:
            input_size: Number of inputs to the network
        """
        self.input_size = input_size
        self.layers_config: List[dict] = []
        self.network: Optional[NeuralNetwork] = None
    
    def set_input_size(self, size: int) -> 'NetworkBuilder':
        """
        Set the input size for the network.
        
        Args:
            size: Number of inputs
            
        Returns:
            Self for method chaining
        """
        if size <= 0:
            raise ValueError("Input size must be positive")
        self.input_size = size
        return self
    
    def add_layer(self, 
                  neurons: int, 
                  activation: Union[str, ActivationFunction] = 'sigmoid') -> 'NetworkBuilder':
        """
        Add a layer to the network configuration.
        
        Args:
            neurons: Number of neurons in the layer
            activation: Activation function (string name or instance)
            
        Returns:
            Self for method chaining
        """
        if neurons <= 0:
            raise ValueError("Number of neurons must be positive")
        
        # Convert string to activation function if needed
        if isinstance(activation, str):
            activation_fn = create_activation(activation)
        else:
            activation_fn = activation
        
        self.layers_config.append({
            'neurons': neurons,
            'activation': activation_fn
        })
        
        return self
    
    def add_hidden_layer(self, neurons: int, activation: str = 'sigmoid') -> 'NetworkBuilder':
        """
        Add a hidden layer (alias for add_layer).
        
        Args:
            neurons: Number of neurons
            activation: Activation function name
            
        Returns:
            Self for method chaining
        """
        return self.add_layer(neurons, activation)
    
    def add_output_layer(self, neurons: int, activation: str = 'sigmoid') -> 'NetworkBuilder':
        """
        Add an output layer.
        
        Args:
            neurons: Number of output neurons
            activation: Activation function name
            
        Returns:
            Self for method chaining
        """
        return self.add_layer(neurons, activation)
    
    def build(self) -> NeuralNetwork:
        """
        Build the neural network.
        
        Returns:
            Configured neural network
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.input_size is None:
            raise ValueError("Input size must be set before building")
        
        if not self.layers_config:
            raise ValueError("Network must have at least one layer")
        
        # Create the network
        network = NeuralNetwork(self.input_size)
        
        # Add layers
        for layer_config in self.layers_config:
            network.add_layer(
                layer_config['neurons'],
                layer_config['activation']
            )
        
        self.network = network
        return network
    
    def reset(self) -> 'NetworkBuilder':
        """
        Reset the builder to initial state.
        
        Returns:
            Self for method chaining
        """
        self.input_size = None
        self.layers_config = []
        self.network = None
        return self
    
    def from_architecture(self, architecture: List[int], 
                         activation: str = 'sigmoid',
                         output_activation: Optional[str] = None) -> 'NetworkBuilder':
        """
        Configure network from architecture list.
        
        Args:
            architecture: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation for hidden layers
            output_activation: Activation for output layer (default: same as hidden)
            
        Returns:
            Self for method chaining
        """
        if len(architecture) < 2:
            raise ValueError("Architecture must have at least input and output sizes")
        
        # Reset and set input size
        self.reset()
        self.set_input_size(architecture[0])
        
        # Add hidden layers
        for i in range(1, len(architecture) - 1):
            self.add_layer(architecture[i], activation)
        
        # Add output layer
        output_act = output_activation or activation
        self.add_layer(architecture[-1], output_act)
        
        return self
    
    def get_summary(self) -> str:
        """
        Get a summary of the current configuration.
        
        Returns:
            String summary of the network configuration
        """
        if self.input_size is None:
            return "NetworkBuilder: No configuration"
        
        summary = f"NetworkBuilder Configuration:\n"
        summary += f"  Input size: {self.input_size}\n"
        
        if self.layers_config:
            summary += "  Layers:\n"
            for i, layer in enumerate(self.layers_config):
                summary += f"    Layer {i+1}: {layer['neurons']} neurons, "
                summary += f"{layer['activation']} activation\n"
        else:
            summary += "  No layers configured\n"
        
        return summary
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            if self.input_size is None or self.input_size <= 0:
                return False
            if not self.layers_config:
                return False
            for layer in self.layers_config:
                if layer['neurons'] <= 0:
                    return False
            return True
        except:
            return False
    
    def __repr__(self) -> str:
        """String representation of the builder."""
        return self.get_summary()


class NetworkFactory:
    """
    Factory class for creating common network architectures.
    """
    
    @staticmethod
    def create_classifier(input_size: int, 
                         num_classes: int,
                         hidden_layers: Optional[List[int]] = None) -> NeuralNetwork:
        """
        Create a classification network.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes (default: [64, 32])
            
        Returns:
            Configured neural network
        """
        if hidden_layers is None:
            hidden_layers = [64, 32]
        
        builder = NetworkBuilder(input_size)
        
        # Add hidden layers
        for size in hidden_layers:
            builder.add_layer(size, 'relu')
        
        # Add output layer with sigmoid for multi-class
        builder.add_output_layer(num_classes, 'sigmoid')
        
        return builder.build()
    
    @staticmethod
    def create_regressor(input_size: int,
                        output_size: int = 1,
                        hidden_layers: Optional[List[int]] = None) -> NeuralNetwork:
        """
        Create a regression network.
        
        Args:
            input_size: Number of input features
            output_size: Number of outputs (default: 1)
            hidden_layers: List of hidden layer sizes (default: [64, 32])
            
        Returns:
            Configured neural network
        """
        if hidden_layers is None:
            hidden_layers = [64, 32]
        
        builder = NetworkBuilder(input_size)
        
        # Add hidden layers
        for size in hidden_layers:
            builder.add_layer(size, 'relu')
        
        # Add output layer with linear activation for regression
        builder.add_output_layer(output_size, 'linear')
        
        return builder.build()
    
    @staticmethod
    def create_autoencoder(input_size: int,
                          encoding_size: int,
                          symmetric: bool = True) -> NeuralNetwork:
        """
        Create an autoencoder network.
        
        Args:
            input_size: Size of input/output
            encoding_size: Size of the encoding layer
            symmetric: Whether to use symmetric architecture
            
        Returns:
            Configured neural network
        """
        builder = NetworkBuilder(input_size)
        
        if symmetric:
            # Encoder
            builder.add_layer(input_size // 2, 'relu')
            builder.add_layer(encoding_size, 'relu')
            # Decoder
            builder.add_layer(input_size // 2, 'relu')
            builder.add_layer(input_size, 'sigmoid')
        else:
            # Simple encoder-decoder
            builder.add_layer(encoding_size, 'relu')
            builder.add_layer(input_size, 'sigmoid')
        
        return builder.build()