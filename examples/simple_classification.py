"""
Simple classification example using the Neural Network Learning System.
This example demonstrates how to create, train, and test a neural network
for a binary classification problem.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.controllers import NetworkBuilder
from src.data import DataGenerator, DataLoader
from src.utils import NetworkVisualizer


def main():
    print("=== Simple Classification Example ===\n")
    
    # 1. Generate synthetic classification data
    print("1. Generating synthetic data...")
    features, targets = DataGenerator.generate_classification_data(
        n_samples=300,
        n_features=2,
        n_classes=2,
        noise=0.15,
        random_seed=42
    )
    
    # Create dataset
    dataset = DataLoader.create_dataset(features, targets)
    train_data, test_data = DataLoader.split_dataset(dataset, train_ratio=0.8)
    
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Test samples: {len(test_data)}")
    
    # 2. Build the neural network
    print("\n2. Building neural network...")
    builder = NetworkBuilder(input_size=2)
    network = builder.add_layer(5, 'sigmoid') \
                    .add_layer(3, 'sigmoid') \
                    .add_layer(2, 'sigmoid') \
                    .build()
    
    print(f"   - Architecture: {network.get_architecture()}")
    
    # 3. Train the network
    print("\n3. Training the network...")
    history = network.train(
        train_data,
        epochs=100,
        learning_rate=0.5,
        validation_data=test_data
    )
    
    print(f"   - Final training loss: {history['loss'][-1]:.4f}")
    print(f"   - Final training accuracy: {history['accuracy'][-1]:.4f}")
    
    # 4. Test the network
    print("\n4. Testing the network...")
    test_loss, test_accuracy = network.evaluate(test_data)
    print(f"   - Test loss: {test_loss:.4f}")
    print(f"   - Test accuracy: {test_accuracy:.4f}")
    
    # 5. Make some predictions
    print("\n5. Making predictions...")
    test_inputs = [
        np.array([0.5, 0.5]),
        np.array([-0.5, 0.5]),
        np.array([0.5, -0.5]),
        np.array([-0.5, -0.5])
    ]
    
    for input_data in test_inputs:
        output = network.predict(input_data)
        predicted_class = np.argmax(output)
        print(f"   Input: {input_data} -> Output: {output} -> Class: {predicted_class}")
    
    # 6. Visualize results
    print("\n6. Visualizing results...")
    
    # Plot training history
    NetworkVisualizer.plot_training_history(
        history,
        title="Training History - Classification Example"
    )
    
    # Plot network architecture
    NetworkVisualizer.plot_network_architecture(
        network,
        title="Network Architecture - Classification Example"
    )
    
    # Plot decision boundary (for 2D data)
    X = np.array([data[0] for data in train_data])
    y = np.array([np.argmax(data[1]) for data in train_data])
    NetworkVisualizer.plot_decision_boundary(
        network, X, y,
        title="Decision Boundary - Classification Example"
    )
    
    # 7. Save the network
    print("\n7. Saving the network...")
    network.save("examples/classification_model.json")
    print("   - Network saved to 'examples/classification_model.json'")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()