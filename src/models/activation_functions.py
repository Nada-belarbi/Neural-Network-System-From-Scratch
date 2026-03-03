"""
Activation functions for neural networks.
Implements the Strategy pattern for different activation functions.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Any


class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply the activation function."""
        pass
    
    @abstractmethod
    def derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the derivative of the activation function."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the activation function."""
        pass


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + e^(-x))
    """
    
    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute sigmoid derivative."""
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def __str__(self) -> str:
        return "Sigmoid"


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    """
    
    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply ReLU function."""
        return np.maximum(0, x)
    
    def derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute ReLU derivative."""
        return np.where(x > 0, 1, 0)
    
    def __str__(self) -> str:
        return "ReLU"


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function.
    f(x) = tanh(x)
    """
    
    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply tanh function."""
        return np.tanh(x)
    
    def derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute tanh derivative."""
        tanh_x = self.forward(x)
        return 1 - tanh_x ** 2
    
    def __str__(self) -> str:
        return "Tanh"


class Linear(ActivationFunction):
    """
    Linear activation function (identity).
    f(x) = x
    """
    
    def forward(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply linear function."""
        return x
    
    def derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute linear derivative."""
        if isinstance(x, np.ndarray):
            return np.ones_like(x)
        return 1.0
    
    def __str__(self) -> str:
        return "Linear"


# Factory function for creating activation functions
def create_activation(name: str) -> ActivationFunction:
    """
    Factory function to create activation functions by name.
    
    Args:
        name: Name of the activation function ('sigmoid', 'relu', 'tanh', 'linear')
    
    Returns:
        ActivationFunction instance
    
    Raises:
        ValueError: If activation function name is not recognized
    """
    activations = {
        'sigmoid': Sigmoid,
        'relu': ReLU,
        'tanh': Tanh,
        'linear': Linear
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation function: {name}. "
                        f"Available: {list(activations.keys())}")
    
    return activations[name_lower]()