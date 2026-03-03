"""
Views package for neural network system.
Provides user interfaces (GUI and CLI).
"""

from .gui_interface import NeuralNetworkGUI
from .cli_interface import NeuralNetworkCLI

__all__ = [
    'NeuralNetworkGUI',
    'NeuralNetworkCLI'
]