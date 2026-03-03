"""
Visualization utilities for neural networks.
Provides functions for plotting training progress and network architecture.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from ..models import NeuralNetwork


class NetworkVisualizer:
    """
    Visualizer for neural network architecture and training.
    """
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], 
                            title: str = "Training History",
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            history: Dictionary with 'loss' and 'accuracy' lists
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot accuracy
        if 'accuracy' in history:
            ax2.plot(epochs, history['accuracy'], 'g-', label='Training Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Accuracy')
            ax2.grid(True)
            ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_network_architecture(network: NeuralNetwork,
                                title: str = "Network Architecture",
                                save_path: Optional[str] = None) -> None:
        """
        Visualize network architecture.
        
        Args:
            network: Neural network to visualize
            title: Plot title
            save_path: Optional path to save the plot
        """
        architecture = network.get_architecture()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate positions
        layer_spacing = 1.0 / (len(architecture) - 1) if len(architecture) > 1 else 0.5
        max_neurons = max(architecture)
        
        # Draw neurons and connections
        neuron_positions = []
        
        for layer_idx, layer_size in enumerate(architecture):
            x = layer_idx * layer_spacing
            layer_positions = []
            
            # Calculate vertical positions for neurons in this layer
            if layer_size == 1:
                y_positions = [0.5]
            else:
                y_spacing = 0.8 / (layer_size - 1)
                y_start = 0.1
                y_positions = [y_start + i * y_spacing for i in range(layer_size)]
            
            # Draw neurons
            for y in y_positions:
                circle = plt.Circle((x, y), 0.02, color='blue', zorder=2)
                ax.add_patch(circle)
                layer_positions.append((x, y))
            
            neuron_positions.append(layer_positions)
            
            # Draw connections to previous layer
            if layer_idx > 0:
                for prev_pos in neuron_positions[layer_idx - 1]:
                    for curr_pos in layer_positions:
                        ax.plot([prev_pos[0], curr_pos[0]], 
                               [prev_pos[1], curr_pos[1]], 
                               'gray', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Add layer labels
        for layer_idx, layer_size in enumerate(architecture):
            x = layer_idx * layer_spacing
            label = f"Input\n({layer_size})" if layer_idx == 0 else f"Layer {layer_idx}\n({layer_size})"
            if layer_idx == len(architecture) - 1:
                label = f"Output\n({layer_size})"
            ax.text(x, -0.1, label, ha='center', va='top')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(network: NeuralNetwork,
                             X: np.ndarray,
                             y: np.ndarray,
                             title: str = "Decision Boundary",
                             resolution: int = 100,
                             save_path: Optional[str] = None) -> None:
        """
        Plot decision boundary for 2D input data.
        
        Args:
            network: Trained neural network
            X: Input features (must be 2D)
            y: Target labels
            title: Plot title
            resolution: Grid resolution
            save_path: Optional path to save the plot
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plot requires 2D input features")
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Predict on mesh grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = []
        
        for point in grid_points:
            output = network.predict(point)
            predictions.append(output[0] if len(output) == 1 else np.argmax(output))
        
        Z = np.array(predictions).reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                            edgecolors='black', linewidth=1)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.colorbar(scatter)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_weight_distribution(network: NeuralNetwork,
                               title: str = "Weight Distribution",
                               save_path: Optional[str] = None) -> None:
        """
        Plot weight distribution across layers.
        
        Args:
            network: Neural network
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, len(network.layers), figsize=(4 * len(network.layers), 4))
        
        if len(network.layers) == 1:
            axes = [axes]
        
        for idx, (layer, ax) in enumerate(zip(network.layers, axes)):
            weights = layer.get_weights_matrix().flatten()
            
            ax.hist(weights, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Layer {idx + 1}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)
            ax.axvline(mean_weight, color='red', linestyle='--', 
                      label=f'Mean: {mean_weight:.3f}')
            ax.text(0.05, 0.95, f'Std: {std_weight:.3f}', 
                   transform=ax.transAxes, verticalalignment='top')
            ax.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Compute confusion matrix
        n_classes = len(np.unique(y_true))
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            cm[int(true), int(pred)] += 1
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=class_names,
               yticklabels=class_names,
               xlabel='Predicted Label',
               ylabel='True Label',
               title=title)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


class TrainingPlotter:
    """
    Real-time plotting during training.
    """
    
    def __init__(self):
        """Initialize the plotter."""
        self.fig = None
        self.axes = None
        self.loss_line = None
        self.acc_line = None
        self.epochs = []
        self.losses = []
        self.accuracies = []
        
        plt.ion()  # Interactive mode
    
    def setup(self) -> None:
        """Setup the plot."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        self.loss_line, = self.ax1.plot([], [], 'b-')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')
        self.ax1.grid(True)
        
        self.acc_line, = self.ax2.plot([], [], 'g-')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.set_title('Training Accuracy')
        self.ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def update(self, epoch: int, loss: float, accuracy: float) -> None:
        """
        Update the plot with new data.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            accuracy: Current accuracy
        """
        if self.fig is None:
            self.setup()
        
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
        # Update loss plot
        self.loss_line.set_data(self.epochs, self.losses)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update accuracy plot
        self.acc_line.set_data(self.epochs, self.accuracies)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Refresh
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self) -> None:
        """Close the plot."""
        plt.ioff()
        if self.fig:
            plt.close(self.fig)