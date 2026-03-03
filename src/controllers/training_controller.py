"""
Training Controller implementation.
Manages the training process and provides monitoring capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from ..models import NeuralNetwork
import time
import json


class TrainingController:
    """
    Controller for managing neural network training.
    
    Implements the Observer pattern for monitoring training progress
    and provides various training strategies.
    """
    
    def __init__(self, network: NeuralNetwork):
        """
        Initialize the training controller.
        
        Args:
            network: Neural network to train
        """
        self.network = network
        self.training_log: List[Dict[str, Any]] = []
        self.callbacks: List[Callable] = []
        
        # Training configuration
        self.config = {
            'learning_rate': 0.1,
            'epochs': 100,
            'batch_size': None,
            'early_stopping': False,
            'patience': 10,
            'min_delta': 0.001,
            'verbose': True
        }
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_weights = None
        self.epochs_without_improvement = 0
    
    def configure(self, **kwargs) -> 'TrainingController':
        """
        Configure training parameters.
        
        Args:
            **kwargs: Training configuration parameters
            
        Returns:
            Self for method chaining
        """
        self.config.update(kwargs)
        return self
    
    def add_callback(self, callback: Callable) -> 'TrainingController':
        """
        Add a callback for training events.
        
        Args:
            callback: Function called with (epoch, metrics) during training
            
        Returns:
            Self for method chaining
        """
        self.callbacks.append(callback)
        return self
    
    def train(self,
              training_data: List[Tuple[np.ndarray, np.ndarray]],
              validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> Dict[str, Any]:
        """
        Train the network with advanced monitoring.
        
        Args:
            training_data: List of (input, target) pairs
            validation_data: Optional validation data
            
        Returns:
            Training results dictionary
        """
        # Reset training state
        self.training_log = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Add observer to network
        self.network.add_training_observer(self._on_epoch_end)
        
        # Start training
        start_time = time.time()
        
        try:
            # Perform training
            history = self.network.train(
                training_data,
                epochs=self.config['epochs'],
                learning_rate=self.config['learning_rate'],
                validation_data=validation_data,
                batch_size=self.config['batch_size']
            )
            
            # Calculate final metrics
            train_loss, train_acc = self.network.evaluate(training_data)
            
            results = {
                'status': 'completed',
                'epochs_trained': self.config['epochs'],
                'final_train_loss': train_loss,
                'final_train_accuracy': train_acc,
                'training_time': time.time() - start_time,
                'history': history,
                'training_log': self.training_log
            }
            
            if validation_data:
                val_loss, val_acc = self.network.evaluate(validation_data)
                results['final_val_loss'] = val_loss
                results['final_val_accuracy'] = val_acc
            
            return results
            
        except KeyboardInterrupt:
            return {
                'status': 'interrupted',
                'epochs_trained': len(self.training_log),
                'training_time': time.time() - start_time,
                'training_log': self.training_log
            }
    
    def _on_epoch_end(self, epoch: int, loss: float, accuracy: float) -> None:
        """
        Handle end of epoch event.
        
        Args:
            epoch: Current epoch number
            loss: Training loss
            accuracy: Training accuracy
        """
        # Log metrics
        metrics = {
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': time.time()
        }
        self.training_log.append(metrics)
        
        # Check for improvement
        if loss < self.best_loss - self.config['min_delta']:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            # Save best weights (simplified - in practice, would deep copy)
            self.best_weights = self._get_network_weights()
        else:
            self.epochs_without_improvement += 1
        
        # Early stopping check
        if (self.config['early_stopping'] and 
            self.epochs_without_improvement >= self.config['patience']):
            if self.config['verbose']:
                print(f"\nEarly stopping triggered at epoch {epoch}")
            # In practice, would need to signal training to stop
        
        # Verbose output
        if self.config['verbose'] and epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Notify callbacks
        for callback in self.callbacks:
            callback(epoch, metrics)
    
    def _get_network_weights(self) -> List[Dict]:
        """Get current network weights."""
        return [layer.get_parameters() for layer in self.network.layers]
    
    def cross_validate(self,
                      data: List[Tuple[np.ndarray, np.ndarray]],
                      k_folds: int = 5) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            data: Complete dataset
            k_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        n_samples = len(data)
        fold_size = n_samples // k_folds
        
        cv_results = {
            'fold_losses': [],
            'fold_accuracies': [],
            'mean_loss': 0,
            'mean_accuracy': 0,
            'std_loss': 0,
            'std_accuracy': 0
        }
        
        for fold in range(k_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else n_samples
            
            val_indices = list(range(start_idx, end_idx))
            train_indices = list(range(0, start_idx)) + list(range(end_idx, n_samples))
            
            train_data = [data[i] for i in train_indices]
            val_data = [data[i] for i in val_indices]
            
            # Train on fold
            if self.config['verbose']:
                print(f"\nTraining fold {fold + 1}/{k_folds}")
            
            self.train(train_data, val_data)
            
            # Evaluate on validation fold
            val_loss, val_acc = self.network.evaluate(val_data)
            cv_results['fold_losses'].append(val_loss)
            cv_results['fold_accuracies'].append(val_acc)
        
        # Calculate statistics
        cv_results['mean_loss'] = np.mean(cv_results['fold_losses'])
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
        cv_results['std_loss'] = np.std(cv_results['fold_losses'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
        
        return cv_results
    
    def grid_search(self,
                   training_data: List[Tuple[np.ndarray, np.ndarray]],
                   param_grid: Dict[str, List[Any]],
                   validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            training_data: Training dataset
            param_grid: Dictionary of parameters to search
            validation_data: Validation dataset
            
        Returns:
            Grid search results
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))
        
        results = {
            'best_params': {},
            'best_score': float('inf'),
            'all_results': []
        }
        
        for i, params in enumerate(param_combinations):
            # Create parameter dictionary
            current_params = dict(zip(param_names, params))
            
            if self.config['verbose']:
                print(f"\nTesting parameters {i+1}/{len(param_combinations)}: {current_params}")
            
            # Configure with current parameters
            self.configure(**current_params)
            
            # Train and evaluate
            train_results = self.train(training_data, validation_data)
            
            # Record results
            result_entry = {
                'params': current_params,
                'train_loss': train_results['final_train_loss'],
                'train_accuracy': train_results['final_train_accuracy']
            }
            
            if validation_data:
                result_entry['val_loss'] = train_results.get('final_val_loss', float('inf'))
                result_entry['val_accuracy'] = train_results.get('final_val_accuracy', 0)
                score = result_entry['val_loss']
            else:
                score = result_entry['train_loss']
            
            results['all_results'].append(result_entry)
            
            # Update best parameters
            if score < results['best_score']:
                results['best_score'] = score
                results['best_params'] = current_params
        
        return results
    
    def save_training_log(self, filepath: str) -> None:
        """
        Save training log to file.
        
        Args:
            filepath: Path to save the log
        """
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config,
                'training_log': self.training_log,
                'best_loss': self.best_loss
            }, f, indent=2)
    
    def load_training_log(self, filepath: str) -> None:
        """
        Load training log from file.
        
        Args:
            filepath: Path to load the log from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.config = data['config']
            self.training_log = data['training_log']
            self.best_loss = data['best_loss']


class TrainingMonitor:
    """
    Monitor for tracking training progress in real-time.
    """
    
    def __init__(self):
        """Initialize the training monitor."""
        self.metrics_history = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Update metrics history.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['losses'].append(metrics.get('loss', 0))
        self.metrics_history['accuracies'].append(metrics.get('accuracy', 0))
        
        if 'val_loss' in metrics:
            self.metrics_history['val_losses'].append(metrics['val_loss'])
        if 'val_accuracy' in metrics:
            self.metrics_history['val_accuracies'].append(metrics['val_accuracy'])
    
    def get_summary(self) -> str:
        """
        Get training summary.
        
        Returns:
            String summary of training progress
        """
        if not self.metrics_history['epochs']:
            return "No training data available"
        
        summary = "Training Summary:\n"
        summary += f"  Epochs trained: {len(self.metrics_history['epochs'])}\n"
        summary += f"  Final loss: {self.metrics_history['losses'][-1]:.4f}\n"
        summary += f"  Final accuracy: {self.metrics_history['accuracies'][-1]:.4f}\n"
        
        if self.metrics_history['val_losses']:
            summary += f"  Final val loss: {self.metrics_history['val_losses'][-1]:.4f}\n"
            summary += f"  Final val accuracy: {self.metrics_history['val_accuracies'][-1]:.4f}\n"
        
        return summary