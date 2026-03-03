"""
Data loading and preprocessing utilities.
Handles dataset loading, preprocessing, and splitting.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import json
import csv
from pathlib import Path


class DataLoader:
    """
    Utility class for loading and preprocessing data.
    """
    
    @staticmethod
    def load_from_csv(filepath: str, 
                     target_column: int = -1,
                     delimiter: str = ',',
                     skip_header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_column: Index of target column (default: last column)
            delimiter: CSV delimiter
            skip_header: Whether to skip the first row
            
        Returns:
            Tuple of (features, targets)
        """
        data = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            if skip_header:
                next(reader)
            
            for row in reader:
                data.append([float(val) for val in row])
        
        data = np.array(data)
        
        # Split features and targets
        if target_column == -1:
            features = data[:, :-1]
            targets = data[:, -1]
        else:
            features = np.delete(data, target_column, axis=1)
            targets = data[:, target_column]
        
        return features, targets
    
    @staticmethod
    def load_from_json(filepath: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load data from JSON file.
        
        Expected format:
        {
            "data": [
                {"input": [x1, x2, ...], "target": [y1, y2, ...]},
                ...
            ]
        }
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of (input, target) pairs
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = []
        for item in data['data']:
            inputs = np.array(item['input'])
            targets = np.array(item['target'])
            dataset.append((inputs, targets))
        
        return dataset
    
    @staticmethod
    def create_dataset(features: np.ndarray, 
                      targets: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create dataset from feature and target arrays.
        
        Args:
            features: 2D array of features
            targets: 1D or 2D array of targets
            
        Returns:
            List of (input, target) pairs
        """
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")
        
        dataset = []
        for i in range(len(features)):
            input_vec = features[i]
            target_vec = targets[i] if targets.ndim > 1 else np.array([targets[i]])
            dataset.append((input_vec, target_vec))
        
        return dataset
    
    @staticmethod
    def split_dataset(dataset: List[Tuple[np.ndarray, np.ndarray]], 
                     train_ratio: float = 0.8,
                     shuffle: bool = True,
                     random_seed: Optional[int] = None) -> Tuple[List, List]:
        """
        Split dataset into training and testing sets.
        
        Args:
            dataset: List of (input, target) pairs
            train_ratio: Ratio of training data
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        train_size = int(n_samples * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = [dataset[i] for i in train_indices]
        test_data = [dataset[i] for i in test_indices]
        
        return train_data, test_data
    
    @staticmethod
    def normalize_features(features: np.ndarray, 
                          method: str = 'minmax') -> Tuple[np.ndarray, dict]:
        """
        Normalize features.
        
        Args:
            features: Feature array
            method: Normalization method ('minmax' or 'standard')
            
        Returns:
            Tuple of (normalized_features, normalization_params)
        """
        if method == 'minmax':
            min_vals = features.min(axis=0)
            max_vals = features.max(axis=0)
            
            # Avoid division by zero
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            
            normalized = (features - min_vals) / range_vals
            
            params = {
                'method': 'minmax',
                'min': min_vals,
                'max': max_vals
            }
            
        elif method == 'standard':
            mean_vals = features.mean(axis=0)
            std_vals = features.std(axis=0)
            
            # Avoid division by zero
            std_vals[std_vals == 0] = 1
            
            normalized = (features - mean_vals) / std_vals
            
            params = {
                'method': 'standard',
                'mean': mean_vals,
                'std': std_vals
            }
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params
    
    @staticmethod
    def apply_normalization(features: np.ndarray, params: dict) -> np.ndarray:
        """
        Apply normalization using saved parameters.
        
        Args:
            features: Feature array
            params: Normalization parameters
            
        Returns:
            Normalized features
        """
        method = params['method']
        
        if method == 'minmax':
            range_vals = params['max'] - params['min']
            range_vals[range_vals == 0] = 1
            return (features - params['min']) / range_vals
            
        elif method == 'standard':
            std_vals = params['std'].copy()
            std_vals[std_vals == 0] = 1
            return (features - params['mean']) / std_vals
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class DataGenerator:
    """
    Generate synthetic datasets for testing.
    """
    
    @staticmethod
    def generate_classification_data(n_samples: int = 100,
                                   n_features: int = 2,
                                   n_classes: int = 2,
                                   noise: float = 0.1,
                                   random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic classification data.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            noise: Noise level
            random_seed: Random seed
            
        Returns:
            Tuple of (features, targets)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate cluster centers
        centers = np.random.randn(n_classes, n_features) * 3
        
        # Generate samples
        features = []
        targets = []
        
        samples_per_class = n_samples // n_classes
        
        for class_idx in range(n_classes):
            # Generate samples around center
            class_samples = centers[class_idx] + np.random.randn(samples_per_class, n_features) * noise
            features.extend(class_samples)
            
            # Create one-hot encoded targets
            target = np.zeros(n_classes)
            target[class_idx] = 1
            targets.extend([target] * samples_per_class)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Shuffle
        indices = np.random.permutation(len(features))
        features = features[indices]
        targets = targets[indices]
        
        return features, targets
    
    @staticmethod
    def generate_regression_data(n_samples: int = 100,
                               n_features: int = 1,
                               noise: float = 0.1,
                               function: Optional[callable] = None,
                               random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic regression data.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            noise: Noise level
            function: Function to generate targets (default: linear)
            random_seed: Random seed
            
        Returns:
            Tuple of (features, targets)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate features
        features = np.random.randn(n_samples, n_features)
        
        # Generate targets
        if function is None:
            # Default: linear function with random weights
            weights = np.random.randn(n_features)
            targets = np.dot(features, weights)
        else:
            targets = np.array([function(x) for x in features])
        
        # Add noise
        targets += np.random.randn(n_samples) * noise
        
        return features, targets.reshape(-1, 1)
    
    