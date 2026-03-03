# Neural Network Learning System

A practical application demonstrating "What AI should learn from software engineering" - an interactive system for creating and configuring supervised learning algorithms based on neural networks.

## Project Overview

This project implements neural networks as reusable software components, allowing users to:
- Create and configure neural network architectures
- Set parameters (number of layers, neurons, inputs/outputs)
- Train networks on datasets
- Test and evaluate performance

## Architecture

The system follows object-oriented design principles and software engineering best practices:
- **Model-View-Controller (MVC)** pattern for separation of concerns
- **Factory pattern** for creating neural network components
- **Strategy pattern** for different activation functions
- **Observer pattern** for training progress updates

## Project Structure

```
neural-network-system/
├── src/
│   ├── models/          # Neural network core components
│   ├── controllers/     # Business logic and orchestration
│   ├── views/          # User interface components
│   ├── utils/          # Utility functions and helpers
│   └── data/           # Data loading and preprocessing
├── tests/              # Unit and integration tests
├── examples/           # Example usage and datasets
└── docs/              # Documentation
```

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.controllers.network_builder import NetworkBuilder
from src.models.activation_functions import Sigmoid

# Create a neural network
builder = NetworkBuilder()
network = builder.add_layer(3, activation=Sigmoid()) \
                .add_layer(5, activation=Sigmoid()) \
                .add_layer(2, activation=Sigmoid()) \
                .build()

# Train the network
network.train(training_data, epochs=1000, learning_rate=0.1)

# Test the network
predictions = network.predict(test_data)
```

## Key Components

### Neuron
- Weighted inputs with bias
- Configurable activation function
- Forward propagation capability

### Layer
- Collection of neurons
- Parallel processing of inputs
- Configurable size and activation

### Network
- Sequential layers architecture
- Training with backpropagation
- Prediction and evaluation methods

## Software Engineering Principles Applied

1. **SOLID Principles**
   - Single Responsibility: Each class has one clear purpose
   - Open/Closed: Extensible for new activation functions
   - Liskov Substitution: Interchangeable components
   - Interface Segregation: Clean, focused interfaces
   - Dependency Inversion: Abstractions over concrete implementations

2. **Design Patterns**
   - Factory Pattern for component creation
   - Strategy Pattern for activation functions
   - Observer Pattern for training monitoring
   - Builder Pattern for network construction

3. **Best Practices**
   - Comprehensive unit testing
   - Clear documentation
   - Type hints for better code clarity
   - Error handling and validation
   - Modular, reusable components

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Documentation

Detailed documentation is available in the `docs/` directory.

## License

This project is developed for educational purposes as part of a software engineering course.
