"""
Command Line Interface for Neural Network System.
Provides a text-based interface for creating and training neural networks.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import json
import sys

from ..models import NeuralNetwork, create_activation
from ..controllers import NetworkBuilder, TrainingController
from ..data import DataLoader, DataGenerator
from ..utils import NetworkVisualizer


class NeuralNetworkCLI:
    """
    Command-line interface for the neural network system.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.network: Optional[NeuralNetwork] = None
        self.builder = NetworkBuilder()
        self.training_controller: Optional[TrainingController] = None
        self.training_data: Optional[List] = None
        self.test_data: Optional[List] = None
        
        self.commands = {
            'help': self.show_help,
            'new': self.new_network,
            'add': self.add_layer,
            'build': self.build_network,
            'show': self.show_network,
            'train': self.train_network,
            'test': self.test_network,
            'predict': self.predict,
            'save': self.save_network,
            'load': self.load_network,
            'data': self.manage_data,
            'visualize': self.visualize,
            'exit': self.exit_cli
        }
    
    def run(self):
        """Run the CLI."""
        print("\n" + "="*60)
        print("Neural Network Learning System - CLI")
        print("="*60)
        print("\nType 'help' for available commands\n")
        
        while True:
            try:
                command = input("nn> ").strip().lower()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in self.commands:
                    self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def show_help(self, args: List[str]):
        """Show help information."""
        help_text = """
Available Commands:
==================

Network Building:
  new <input_size>     - Create a new network with specified input size
  add <neurons> <activation> - Add a layer (activation: sigmoid, relu, tanh, linear)
  build               - Build the network with current configuration
  show                - Show current network architecture

Training:
  train               - Train the network with loaded data
  test                - Test the network on test data
  predict <values>    - Make a prediction (comma-separated values)

Data Management:
  data load <file>    - Load data from CSV or JSON file
  data generate       - Generate synthetic data
  data info           - Show information about loaded data

File Operations:
  save <filename>     - Save the network to a file
  load <filename>     - Load a network from a file

Visualization:
  visualize arch      - Plot network architecture
  visualize history   - Plot training history
  visualize weights   - Plot weight distribution

Other:
  help                - Show this help message
  exit                - Exit the program

Examples:
  nn> new 2
  nn> add 5 sigmoid
  nn> add 3 relu
  nn> add 1 sigmoid
  nn> build
  nn> data generate
  nn> train
  nn> predict 0.5,0.8
"""
        print(help_text)
    
    def new_network(self, args: List[str]):
        """Create a new network."""
        if not args:
            print("Usage: new <input_size>")
            return
        
        try:
            input_size = int(args[0])
            self.builder = NetworkBuilder(input_size)
            self.network = None
            self.training_controller = None
            print(f"Created new network builder with input size: {input_size}")
        except ValueError:
            print("Error: Input size must be an integer")
    
    def add_layer(self, args: List[str]):
        """Add a layer to the network."""
        if len(args) < 2:
            print("Usage: add <neurons> <activation>")
            print("Activations: sigmoid, relu, tanh, linear")
            return
        
        try:
            neurons = int(args[0])
            activation = args[1].lower()
            
            if self.builder.input_size is None:
                print("Error: Please create a new network first (use 'new' command)")
                return
            
            self.builder.add_layer(neurons, activation)
            print(f"Added layer: {neurons} neurons with {activation} activation")
            
        except ValueError as e:
            print(f"Error: {str(e)}")
    
    def build_network(self, args: List[str]):
        """Build the network."""
        try:
            if self.builder.input_size is None:
                print("Error: No network configuration. Use 'new' command first.")
                return
            
            if not self.builder.layers_config:
                print("Error: No layers added. Use 'add' command to add layers.")
                return
            
            self.network = self.builder.build()
            self.training_controller = TrainingController(self.network)
            
            print("Network built successfully!")
            self.show_network([])
            
        except Exception as e:
            print(f"Error building network: {str(e)}")
    
    def show_network(self, args: List[str]):
        """Show network architecture."""
        if self.network is None:
            print("No network built yet.")
            return
        
        architecture = self.network.get_architecture()
        print(f"\nNetwork Architecture: {architecture}")
        print(f"Total layers: {len(self.network.layers)}")
        
        for i, layer in enumerate(self.network.layers):
            print(f"  Layer {i+1}: {layer.num_neurons} neurons, {layer.activation} activation")
    
    def train_network(self, args: List[str]):
        """Train the network."""
        if self.network is None:
            print("Error: No network built. Use 'build' command first.")
            return
        
        if self.training_data is None:
            print("Error: No training data loaded. Use 'data' command first.")
            return
        
        # Get training parameters
        print("\nTraining Configuration:")
        try:
            epochs = int(input("  Epochs (default 100): ") or "100")
            learning_rate = float(input("  Learning rate (default 0.1): ") or "0.1")
            batch_size = input("  Batch size (default: full batch): ")
            batch_size = int(batch_size) if batch_size else None
            
            # Configure training
            self.training_controller.configure(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=True
            )
            
            print("\nTraining started...")
            results = self.training_controller.train(
                self.training_data,
                validation_data=self.test_data
            )
            
            print("\nTraining completed!")
            print(f"Final loss: {results['final_train_loss']:.4f}")
            print(f"Final accuracy: {results['final_train_accuracy']:.4f}")
            
            if 'final_val_loss' in results:
                print(f"Validation loss: {results['final_val_loss']:.4f}")
                print(f"Validation accuracy: {results['final_val_accuracy']:.4f}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        except Exception as e:
            print(f"Error during training: {str(e)}")
    
    def test_network(self, args: List[str]):
        """Test the network."""
        if self.network is None:
            print("Error: No network built.")
            return
        
        if self.test_data is None:
            print("Error: No test data loaded.")
            return
        
        try:
            loss, accuracy = self.network.evaluate(self.test_data)
            print(f"\nTest Results:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Total samples: {len(self.test_data)}")
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
    
    def predict(self, args: List[str]):
        """Make a prediction."""
        if self.network is None:
            print("Error: No network built.")
            return
        
        if not args:
            print("Usage: predict <comma-separated values>")
            print("Example: predict 0.5,0.8")
            return
        
        try:
            # Parse input
            input_str = ' '.join(args)
            input_values = [float(x.strip()) for x in input_str.split(',')]
            
            # Make prediction
            output = self.network.predict(np.array(input_values))
            
            print(f"\nInput: {input_values}")
            print(f"Output: {output}")
            
            if len(output) > 1:
                predicted_class = np.argmax(output)
                print(f"Predicted class: {predicted_class}")
                
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
    
    def save_network(self, args: List[str]):
        """Save the network."""
        if self.network is None:
            print("Error: No network to save.")
            return
        
        if not args:
            print("Usage: save <filename>")
            return
        
        try:
            filename = args[0]
            if not filename.endswith('.json'):
                filename += '.json'
            
            self.network.save(filename)
            print(f"Network saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving network: {str(e)}")
    
    def load_network(self, args: List[str]):
        """Load a network."""
        if not args:
            print("Usage: load <filename>")
            return
        
        try:
            filename = args[0]
            if not filename.endswith('.json'):
                filename += '.json'
            
            self.network = NeuralNetwork(1)  # Dummy input size
            self.network.load(filename)
            self.training_controller = TrainingController(self.network)
            
            print(f"Network loaded from: {filename}")
            self.show_network([])
            
        except Exception as e:
            print(f"Error loading network: {str(e)}")
    
    def manage_data(self, args: List[str]):
        """Manage data operations."""
        if not args:
            print("Usage: data <load|generate|info>")
            return
        
        subcmd = args[0]
        
        if subcmd == 'load':
            self._load_data(args[1:])
        elif subcmd == 'generate':
            self._generate_data()
        elif subcmd == 'info':
            self._show_data_info()
        else:
            print(f"Unknown data command: {subcmd}")
    
    def _load_data(self, args: List[str]):
        """Load data from file."""
        if not args:
            print("Usage: data load <filename>")
            return
        
        filename = args[0]
        
        try:
            if filename.endswith('.csv'):
                features, targets = DataLoader.load_from_csv(filename)
                dataset = DataLoader.create_dataset(features, targets)
            elif filename.endswith('.json'):
                dataset = DataLoader.load_from_json(filename)
            else:
                print("Error: Unsupported file format. Use .csv or .json")
                return
            
            # Split data
            self.training_data, self.test_data = DataLoader.split_dataset(
                dataset, train_ratio=0.8
            )
            
            print(f"Data loaded successfully!")
            print(f"  Training samples: {len(self.training_data)}")
            print(f"  Test samples: {len(self.test_data)}")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
    
    def _generate_data(self):
        """Generate synthetic data."""
        print("\nGenerate Synthetic Data")
        print("Data types: classification, regression, xor")
        
        try:
            data_type = input("Data type: ").lower()
            n_samples = int(input("Number of samples (default 200): ") or "200")
            
            if data_type == 'classification':
                n_features = int(input("Number of features (default 2): ") or "2")
                n_classes = int(input("Number of classes (default 2): ") or "2")
                noise = float(input("Noise level (default 0.1): ") or "0.1")
                
                features, targets = DataGenerator.generate_classification_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    noise=noise
                )
                
            elif data_type == 'regression':
                n_features = int(input("Number of features (default 1): ") or "1")
                noise = float(input("Noise level (default 0.1): ") or "0.1")
                
                features, targets = DataGenerator.generate_regression_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    noise=noise
                )
                
            elif data_type == 'xor':
                noise = float(input("Noise level (default 0.1): ") or "0.1")
                
                features, targets = DataGenerator.generate_xor_data(
                    n_samples=n_samples,
                    noise=noise
                )
            else:
                print(f"Unknown data type: {data_type}")
                return
            
            # Create dataset
            dataset = DataLoader.create_dataset(features, targets)
            self.training_data, self.test_data = DataLoader.split_dataset(
                dataset, train_ratio=0.8
            )
            
            print(f"\nData generated successfully!")
            print(f"  Training samples: {len(self.training_data)}")
            print(f"  Test samples: {len(self.test_data)}")
            print(f"  Feature shape: {features[0].shape}")
            print(f"  Target shape: {targets[0].shape}")
            
        except Exception as e:
            print(f"Error generating data: {str(e)}")
    
    def _show_data_info(self):
        """Show information about loaded data."""
        if self.training_data is None:
            print("No data loaded.")
            return
        
        print("\nData Information:")
        print(f"  Training samples: {len(self.training_data)}")
        print(f"  Test samples: {len(self.test_data) if self.test_data else 0}")
        
        if self.training_data:
            sample_input, sample_target = self.training_data[0]
            print(f"  Input shape: {sample_input.shape}")
            print(f"  Target shape: {sample_target.shape}")
            print(f"  Sample input: {sample_input}")
            print(f"  Sample target: {sample_target}")
    
    def visualize(self, args: List[str]):
        """Visualization commands."""
        if not args:
            print("Usage: visualize <arch|history|weights>")
            return
        
        if self.network is None:
            print("Error: No network built.")
            return
        
        viz_type = args[0]
        
        try:
            if viz_type == 'arch':
                NetworkVisualizer.plot_network_architecture(self.network)
            elif viz_type == 'history':
                if not self.network.training_history['epochs']:
                    print("Error: No training history available.")
                    return
                NetworkVisualizer.plot_training_history(self.network.training_history)
            elif viz_type == 'weights':
                NetworkVisualizer.plot_weight_distribution(self.network)
            else:
                print(f"Unknown visualization type: {viz_type}")
                
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def exit_cli(self, args: List[str]):
        """Exit the CLI."""
        print("\nGoodbye!")
        sys.exit(0)


def main():
    """Main entry point for CLI."""
    cli = NeuralNetworkCLI()
    cli.run()


if __name__ == "__main__":
    main()