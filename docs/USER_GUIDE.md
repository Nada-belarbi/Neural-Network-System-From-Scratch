# Neural Network Learning System - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Architecture Overview](#architecture-overview)
5. [Using the GUI](#using-the-gui)
6. [Using the CLI](#using-the-cli)
7. [Programming Interface](#programming-interface)
8. [Examples](#examples)
9. [Best Practices](#best-practices)

## Introduction

The Neural Network Learning System is an educational tool that demonstrates software engineering principles applied to artificial intelligence. It provides a complete implementation of neural networks as reusable software components, allowing users to:

- Create custom neural network architectures
- Train networks on various datasets
- Visualize network structure and training progress
- Save and load trained models

## Installation

1. Ensure you have Python 3.7+ installed
2. Clone or download the project
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### GUI Mode
```bash
python main_gui.py
```

### CLI Mode
```bash
python main_cli.py
```

### Programmatic Usage
```python
from src.controllers import NetworkBuilder
from src.data import DataGenerator, DataLoader

# Create a network
builder = NetworkBuilder(input_size=2)
network = builder.add_layer(5, 'sigmoid') \
                .add_layer(1, 'sigmoid') \
                .build()

# Generate data
features, targets = DataGenerator.generate_xor_data(n_samples=200)
dataset = DataLoader.create_dataset(features, targets)

# Train
network.train(dataset, epochs=100, learning_rate=0.5)
```

## Architecture Overview

The system follows object-oriented design principles with clear separation of concerns:

### Core Components

1. **Neuron** (`src/models/neuron.py`)
   - Basic unit of computation
   - Manages weights, bias, and activation
   - Implements forward and backward propagation

2. **Layer** (`src/models/layer.py`)
   - Collection of neurons processing in parallel
   - Manages connections between layers
   - Handles batch operations

3. **Network** (`src/models/network.py`)
   - Sequential composition of layers
   - Implements training algorithms
   - Provides save/load functionality

### Design Patterns Used

- **Factory Pattern**: For creating activation functions and network architectures
- **Builder Pattern**: For constructing networks with fluent interface
- **Strategy Pattern**: For interchangeable activation functions
- **Observer Pattern**: For monitoring training progress

## Using the GUI

### Main Features

1. **Architecture Tab**
   - Set input size
   - Add/remove layers
   - Configure neurons and activation functions
   - Build network

2. **Training Tab**
   - Set learning parameters (learning rate, epochs, batch size)
   - Monitor training progress
   - View real-time metrics

3. **Testing Tab**
   - Make individual predictions
   - Evaluate on test dataset
   - View performance metrics

4. **Visualization Tab**
   - Plot network architecture
   - View training history
   - Analyze weight distributions
   - Visualize decision boundaries (for 2D problems)

### Workflow

1. Create a new network or load existing one
2. Load or generate training data
3. Configure and build network architecture
4. Set training parameters and train
5. Test and visualize results
6. Save trained network

## Using the CLI

### Available Commands

```
Network Building:
  new <input_size>     - Create new network
  add <neurons> <activation> - Add layer
  build               - Build network
  show                - Show architecture

Training:
  train               - Train network
  test                - Test network
  predict <values>    - Make prediction

Data Management:
  data load <file>    - Load data
  data generate       - Generate synthetic data
  data info           - Show data info

File Operations:
  save <filename>     - Save network
  load <filename>     - Load network

Visualization:
  visualize arch      - Plot architecture
  visualize history   - Plot training history
  visualize weights   - Plot weights
```

### Example Session

```bash
nn> new 2
nn> add 4 sigmoid
nn> add 1 sigmoid
nn> build
nn> data generate
nn> train
nn> predict 0.5,0.8
nn> save my_network.json
```

## Programming Interface

### Creating Networks

```python
# Using NetworkBuilder
from src.controllers import NetworkBuilder

builder = NetworkBuilder(input_size=3)
network = builder.add_layer(5, 'relu') \
                .add_layer(3, 'sigmoid') \
                .build()

# Using NetworkFactory
from src.controllers import NetworkFactory

# For classification
network = NetworkFactory.create_classifier(
    input_size=10,
    num_classes=3,
    hidden_layers=[64, 32]
)

# For regression
network = NetworkFactory.create_regressor(
    input_size=5,
    output_size=1,
    hidden_layers=[32, 16]
)
```

### Training Networks

```python
from src.controllers import TrainingController

# Basic training
history = network.train(
    training_data,
    epochs=100,
    learning_rate=0.1
)

# Advanced training with controller
controller = TrainingController(network)
controller.configure(
    learning_rate=0.1,
    epochs=100,
    batch_size=32,
    early_stopping=True,
    patience=10
)

results = controller.train(training_data, validation_data)
```

### Data Handling

```python
from src.data import DataLoader, DataGenerator

# Load from CSV
features, targets = DataLoader.load_from_csv('data.csv')
dataset = DataLoader.create_dataset(features, targets)

# Generate synthetic data
features, targets = DataGenerator.generate_classification_data(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    noise=0.1
)

# Split data
train_data, test_data = DataLoader.split_dataset(
    dataset, 
    train_ratio=0.8
)

# Normalize features
normalized_features, params = DataLoader.normalize_features(
    features, 
    method='minmax'
)
```

### Visualization

```python
from src.utils import NetworkVisualizer

# Plot architecture
NetworkVisualizer.plot_network_architecture(network)

# Plot training history
NetworkVisualizer.plot_training_history(history)

# Plot decision boundary (2D only)
NetworkVisualizer.plot_decision_boundary(network, X, y)

# Plot weight distribution
NetworkVisualizer.plot_weight_distribution(network)
```

## Examples
```

### Classification
```python
# See examples/simple_classification.py
python examples/simple_classification.py
```

## Best Practices

### Network Design
1. Start with simple architectures
2. Add complexity gradually
3. Use appropriate activation functions:
   - Sigmoid/Tanh: For output layers in classification
   - ReLU: For hidden layers
   - Linear: For regression outputs

### Training
1. Normalize input data
2. Use appropriate learning rates (typically 0.01-0.5)
3. Monitor for overfitting
4. Use validation data
5. Consider early stopping

### Data Preparation
1. Ensure balanced classes for classification
2. Split data appropriately (e.g., 80/20 train/test)
3. Normalize features to similar scales
4. Handle missing values before training

### Performance Tips
1. Use batch training for large datasets
2. Start with smaller networks
3. Increase complexity only if needed
4. Save best models during training

## Troubleshooting

### Common Issues

1. **Network not learning**
   - Check learning rate (too high or too low)
   - Verify data is properly formatted
   - Ensure network has sufficient capacity

2. **Overfitting**
   - Reduce network size
   - Add more training data
   - Use early stopping

3. **Slow training**
   - Reduce batch size
   - Use fewer neurons/layers
   - Check for numerical instabilities

### Getting Help

- Check example scripts in `examples/`
- Review test cases in `tests/`
- Examine source code documentation