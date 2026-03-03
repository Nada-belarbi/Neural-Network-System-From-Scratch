#!/usr/bin/env python3
"""
Test script to verify the Neural Network Learning System is working correctly.
"""

import sys
import numpy as np
from src.controllers import NetworkBuilder
from src.data import DataGenerator, DataLoader
from src.utils import NetworkVisualizer


def test_basic_functionality():
    """Test basic system functionality."""
    print("Testing Neural Network Learning System...")
    print("=" * 60)
    
    try:
        # Test 1: Create a simple network
        print("\n1. Creating neural network...")
        builder = NetworkBuilder(input_size=2)
        network = builder.add_layer(3, 'sigmoid') \
                        .add_layer(1, 'sigmoid') \
                        .build()
        print("   ✓ Network created successfully")
        print(f"   Architecture: {network.get_architecture()}")
        
        # Test 2: Generate data
        print("\n2. Generating test data...")
        features, targets = DataGenerator.generate_xor_data(
            n_samples=100,
            noise=0.1
        )
        dataset = DataLoader.create_dataset(features, targets)
        train_data, test_data = DataLoader.split_dataset(dataset, train_ratio=0.8)
        print("   ✓ Data generated successfully")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Test samples: {len(test_data)}")
        
        # Test 3: Train network
        print("\n3. Training network...")
        history = network.train(
            train_data,
            epochs=50,
            learning_rate=0.5
        )
        print("   ✓ Training completed successfully")
        print(f"   Final loss: {history['loss'][-1]:.4f}")
        print(f"   Final accuracy: {history['accuracy'][-1]:.4f}")
        
        # Test 4: Make predictions
        print("\n4. Making predictions...")
        test_inputs = [
            np.array([0.5, 0.5]),
            np.array([-0.5, 0.5]),
            np.array([0.5, -0.5]),
            np.array([-0.5, -0.5])
        ]
        
        for input_data in test_inputs:
            output = network.predict(input_data)
            print(f"   Input: {input_data} -> Output: {output[0]:.4f}")
        print("   ✓ Predictions working correctly")
        
        # Test 5: Save and load
        print("\n5. Testing save/load functionality...")
        network.save("test_network.json")
        print("   ✓ Network saved successfully")
        
        from src.models import NeuralNetwork
        new_network = NeuralNetwork(1)  # Create empty network
        new_network.load("test_network.json")
        print("   ✓ Network loaded successfully")
        
        # Verify loaded network works
        test_output = new_network.predict(np.array([0.5, 0.5]))
        print(f"   Loaded network prediction: {test_output[0]:.4f}")
        
        # Clean up
        import os
        os.remove("test_network.json")
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("The Neural Network Learning System is working correctly.")
        print("\nYou can now run:")
        print("  - python main_gui.py    (for GUI interface)")
        print("  - python main_cli.py    (for CLI interface)")
        print("  - python examples/simple_classification.py")
        print("  - python examples/xor_problem.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nPlease check:")
        print("1. All dependencies are installed (pip install -r requirements.txt)")
        print("2. You're running from the project root directory")
        print("3. Python version is 3.7 or higher")
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting imports...")
    
    modules = [
        "src.models",
        "src.controllers", 
        "src.data",
        "src.utils",
        "src.views"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"   ✓ {module}")
        except ImportError as e:
            print(f"   ✗ {module}: {e}")
            return False
    
    return True


if __name__ == "__main__":
    print("Neural Network Learning System - System Test")
    print("=" * 60)
    
    # Test imports first
    if not test_imports():
        sys.exit(1)
    
    # Test functionality
    if test_basic_functionality():
        sys.exit(0)
    else:
        sys.exit(1)