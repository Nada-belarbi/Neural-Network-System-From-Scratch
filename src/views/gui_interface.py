"""
GUI Interface for Neural Network System.
Provides an interactive interface for creating and training neural networks.
"""

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from typing import Optional, List, Dict, Any
import json
import threading

from ..models import NeuralNetwork, create_activation
from ..controllers import NetworkBuilder, TrainingController
from ..data import DataLoader, DataGenerator
from ..utils import NetworkVisualizer


class NeuralNetworkGUI:
    """
    Main GUI application for the neural network system.
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Neural Network Learning System")
        self.root.geometry("800x600")
        
        # Model components
        self.network: Optional[NeuralNetwork] = None
        self.builder = NetworkBuilder()
        self.training_controller: Optional[TrainingController] = None
        self.training_data: Optional[List] = None
        self.test_data: Optional[List] = None
        
        # Setup GUI
        self._setup_menu()
        self._setup_tabs()
        self._setup_status_bar()
    
    def _setup_menu(self) -> None:
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Network", command=self._new_network)
        file_menu.add_command(label="Load Network", command=self._load_network)
        file_menu.add_command(label="Save Network", command=self._save_network)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="Load from CSV", command=self._load_csv_data)
        data_menu.add_command(label="Load from JSON", command=self._load_json_data)
        data_menu.add_command(label="Generate Synthetic", command=self._generate_data)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _setup_tabs(self) -> None:
        """Setup tab interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Architecture tab
        self.arch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.arch_frame, text="Architecture")
        self._setup_architecture_tab()
        
        # Training tab
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Training")
        self._setup_training_tab()
        
        # Testing tab
        self.test_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.test_frame, text="Testing")
        self._setup_testing_tab()
        
        # Visualization tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        self._setup_visualization_tab()
    
    def _setup_architecture_tab(self) -> None:
        """Setup architecture configuration tab."""
        # Input size
        input_frame = ttk.LabelFrame(self.arch_frame, text="Input Configuration")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Input Size:").grid(row=0, column=0, padx=5, pady=5)
        self.input_size_var = tk.IntVar(value=2)
        ttk.Spinbox(input_frame, from_=1, to=100, textvariable=self.input_size_var,
                   width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Layers configuration
        layers_frame = ttk.LabelFrame(self.arch_frame, text="Layers Configuration")
        layers_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Layer list
        self.layers_listbox = tk.Listbox(layers_frame, height=6)
        self.layers_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Layer controls
        controls_frame = ttk.Frame(layers_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Neurons:").grid(row=0, column=0, padx=5, pady=2)
        self.neurons_var = tk.IntVar(value=5)
        ttk.Spinbox(controls_frame, from_=1, to=100, textvariable=self.neurons_var,
                   width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(controls_frame, text="Activation:").grid(row=1, column=0, padx=5, pady=2)
        self.activation_var = tk.StringVar(value="sigmoid")
        activation_combo = ttk.Combobox(controls_frame, textvariable=self.activation_var,
                                      values=["sigmoid", "relu", "tanh", "linear"],
                                      width=10, state="readonly")
        activation_combo.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Button(controls_frame, text="Add Layer",
                  command=self._add_layer).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(controls_frame, text="Remove Layer",
                  command=self._remove_layer).grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Button(controls_frame, text="Clear All",
                  command=self._clear_layers).grid(row=4, column=0, columnspan=2, pady=5)
        
        # Build button
        ttk.Button(self.arch_frame, text="Build Network", command=self._build_network,
                  style="Accent.TButton").pack(pady=10)
    
    def _setup_training_tab(self) -> None:
        """Setup training configuration tab."""
        # Data loading section
        data_frame = ttk.LabelFrame(self.train_frame, text="Data Loading")
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Data loading buttons
        button_frame = ttk.Frame(data_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Load CSV", 
                  command=self._load_csv_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load JSON", 
                  command=self._load_json_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Data", 
                  command=self._generate_data).pack(side=tk.LEFT, padx=5)
        
        # Data info label
        self.data_info_label = ttk.Label(data_frame, text="No data loaded")
        self.data_info_label.pack(pady=5)
        
        # Training parameters
        params_frame = ttk.LabelFrame(self.train_frame, text="Training Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, pady=5)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=2, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(params_frame, from_=1, to=10000, textvariable=self.epochs_var,
                   width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, padx=5, pady=5)
        self.batch_var = tk.IntVar(value=32)
        ttk.Spinbox(params_frame, from_=1, to=1000, textvariable=self.batch_var,
                   width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Early stopping
        self.early_stop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Early Stopping",
                       variable=self.early_stop_var).grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Training progress
        progress_frame = ttk.LabelFrame(self.train_frame, text="Training Progress")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, length=300)
        self.progress_bar.pack(pady=10)
        
        # Training log
        self.training_log = tk.Text(progress_frame, height=10, width=60)
        self.training_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(progress_frame, command=self.training_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_log.config(yscrollcommand=scrollbar.set)
        
        # Control buttons
        button_frame = ttk.Frame(self.train_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Start Training",
                  command=self._start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Training",
                  command=self._stop_training).pack(side=tk.LEFT, padx=5)
    
    def _setup_testing_tab(self) -> None:
        """Setup testing tab."""
        # Test input
        input_frame = ttk.LabelFrame(self.test_frame, text="Test Input")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Input Values (comma-separated):").pack(padx=5, pady=5)
        self.test_input_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.test_input_var,
                 width=50).pack(padx=5, pady=5)
        
        ttk.Button(input_frame, text="Predict",
                  command=self._predict_single).pack(pady=5)
        
        # Prediction result
        result_frame = ttk.LabelFrame(self.test_frame, text="Prediction Result")
        result_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.prediction_label = ttk.Label(result_frame, text="No prediction yet",
                                        font=("Arial", 12))
        self.prediction_label.pack(padx=10, pady=10)
        
        # Batch testing
        batch_frame = ttk.LabelFrame(self.test_frame, text="Batch Testing")
        batch_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Button(batch_frame, text="Test on Dataset",
                  command=self._test_dataset).pack(pady=5)
        
        self.test_results = tk.Text(batch_frame, height=10, width=60)
        self.test_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _setup_visualization_tab(self) -> None:
        """Setup visualization tab."""
        # Visualization options
        viz_buttons = ttk.Frame(self.viz_frame)
        viz_buttons.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(viz_buttons, text="Plot Architecture",
                  command=self._plot_architecture).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons, text="Plot Training History",
                  command=self._plot_training_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons, text="Plot Weight Distribution",
                  command=self._plot_weights).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons, text="Plot Decision Boundary",
                  command=self._plot_decision_boundary).pack(side=tk.LEFT, padx=5)
        
        # Visualization info
        info_frame = ttk.LabelFrame(self.viz_frame, text="Visualization Info")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.viz_info = tk.Text(info_frame, height=15, width=60)
        self.viz_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.viz_info.insert(tk.END, "Visualization tools:\n\n")
        self.viz_info.insert(tk.END, "• Architecture: Shows network structure\n")
        self.viz_info.insert(tk.END, "• Training History: Plots loss and accuracy over epochs\n")
        self.viz_info.insert(tk.END, "• Weight Distribution: Shows weight values distribution\n")
        self.viz_info.insert(tk.END, "• Decision Boundary: For 2D input problems only\n")
    
    def _setup_status_bar(self) -> None:
        """Setup status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _update_status(self, message: str) -> None:
        """Update status bar message."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    # Architecture methods
    def _add_layer(self) -> None:
        """Add a layer to the configuration."""
        neurons = self.neurons_var.get()
        activation = self.activation_var.get()
        layer_desc = f"Layer: {neurons} neurons, {activation} activation"
        self.layers_listbox.insert(tk.END, layer_desc)
        self._update_status(f"Added layer: {neurons} neurons with {activation}")
    
    def _remove_layer(self) -> None:
        """Remove selected layer."""
        selection = self.layers_listbox.curselection()
        if selection:
            self.layers_listbox.delete(selection[0])
            self._update_status("Removed layer")
    
    def _clear_layers(self) -> None:
        """Clear all layers."""
        self.layers_listbox.delete(0, tk.END)
        self._update_status("Cleared all layers")
    
    def _build_network(self) -> None:
        """Build the neural network."""
        try:
            # Get input size
            input_size = self.input_size_var.get()
            self.builder = NetworkBuilder(input_size)
            
            # Add layers
            layer_count = self.layers_listbox.size()
            if layer_count == 0:
                messagebox.showerror("Error", "Please add at least one layer")
                return
            
            for i in range(layer_count):
                layer_desc = self.layers_listbox.get(i)
                # Parse layer description
                parts = layer_desc.split()
                neurons = int(parts[1])
                activation = parts[3]
                self.builder.add_layer(neurons, activation)
            
            # Build network
            self.network = self.builder.build()
            self.training_controller = TrainingController(self.network)
            
            self._update_status(f"Network built: {self.network.get_architecture()}")
            messagebox.showinfo("Success", "Neural network built successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build network: {str(e)}")
    
    # Training methods
    def _start_training(self) -> None:
        """Start training the network."""
        if self.network is None:
            messagebox.showerror("Error", "Please build a network first")
            return
        
        if self.training_data is None:
            messagebox.showerror("Error", "Please load training data first")
            return
        
        # Configure training
        self.training_controller.configure(
            learning_rate=self.lr_var.get(),
            epochs=self.epochs_var.get(),
            batch_size=self.batch_var.get(),
            early_stopping=self.early_stop_var.get(),
            verbose=False
        )
        
        # Add callback for progress updates
        self.training_controller.add_callback(self._training_callback)
        
        # Clear log
        self.training_log.delete(1.0, tk.END)
        self.training_log.insert(tk.END, "Starting training...\n")
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._train_network,
            daemon=True
        )
        self.training_thread.start()
    
    def _train_network(self) -> None:
        """Train the network (runs in separate thread)."""
        try:
            results = self.training_controller.train(
                self.training_data,
                validation_data=self.test_data
            )
            
            # Update UI in main thread
            self.root.after(0, self._training_complete, results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
    
    def _training_callback(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Callback for training progress updates."""
        # Update progress bar
        progress = (epoch / self.epochs_var.get()) * 100
        self.root.after(0, lambda: self.progress_var.set(progress))
        
        # Update log
        log_message = f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}\n"
        self.root.after(0, lambda: self._append_to_log(log_message))
    
    def _append_to_log(self, message: str) -> None:
        """Append message to training log."""
        self.training_log.insert(tk.END, message)
        self.training_log.see(tk.END)
    
    def _training_complete(self, results: Dict[str, Any]) -> None:
        """Handle training completion."""
        self.progress_var.set(100)
        self._append_to_log("\nTraining completed!\n")
        self._append_to_log(f"Final loss: {results['final_train_loss']:.4f}\n")
        self._append_to_log(f"Final accuracy: {results['final_train_accuracy']:.4f}\n")
        self._update_status("Training completed")
        messagebox.showinfo("Success", "Training completed successfully!")
    
    def _stop_training(self) -> None:
        """Stop training (placeholder)."""
        messagebox.showinfo("Info", "Training stop not implemented in this version")
    
    # Testing methods
    def _predict_single(self) -> None:
        """Make a single prediction."""
        if self.network is None:
            messagebox.showerror("Error", "Please build and train a network first")
            return
        
        try:
            # Parse input
            input_str = self.test_input_var.get()
            input_values = [float(x.strip()) for x in input_str.split(',')]
            
            # Make prediction
            output = self.network.predict(np.array(input_values))
            
            # Display result
            result_str = f"Output: {output}"
            if len(output) > 1:
                predicted_class = np.argmax(output)
                result_str += f"\nPredicted class: {predicted_class}"
            
            self.prediction_label.config(text=result_str)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def _test_dataset(self) -> None:
        """Test on entire dataset."""
        if self.network is None:
            messagebox.showerror("Error", "Please build and train a network first")
            return
        
        if self.test_data is None:
            messagebox.showerror("Error", "Please load test data first")
            return
        
        try:
            # Evaluate
            loss, accuracy = self.network.evaluate(self.test_data)
            
            # Display results
            self.test_results.delete(1.0, tk.END)
            self.test_results.insert(tk.END, "Test Results:\n\n")
            self.test_results.insert(tk.END, f"Test Loss: {loss:.4f}\n")
            self.test_results.insert(tk.END, f"Test Accuracy: {accuracy:.4f}\n")
            self.test_results.insert(tk.END, f"Total samples: {len(self.test_data)}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {str(e)}")
    
    # Visualization methods
    def _plot_architecture(self) -> None:
        """Plot network architecture."""
        if self.network is None:
            messagebox.showerror("Error", "Please build a network first")
            return
        
        try:
            NetworkVisualizer.plot_network_architecture(self.network)
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
    
    def _plot_training_history(self) -> None:
        """Plot training history."""
        if self.network is None or not self.network.training_history['epochs']:
            messagebox.showerror("Error", "No training history available")
            return
        
        try:
            NetworkVisualizer.plot_training_history(self.network.training_history)
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
    
    def _plot_weights(self) -> None:
        """Plot weight distribution."""
        if self.network is None:
            messagebox.showerror("Error", "Please build a network first")
            return
        
        try:
            NetworkVisualizer.plot_weight_distribution(self.network)
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
    
    def _plot_decision_boundary(self) -> None:
        """Plot decision boundary."""
        if self.network is None:
            messagebox.showerror("Error", "Please build and train a network first")
            return
        
        if self.training_data is None:
            messagebox.showerror("Error", "Please load training data first")
            return
        
        try:
            # Extract features and labels
            X = np.array([data[0] for data in self.training_data])
            y = np.array([np.argmax(data[1]) if len(data[1]) > 1 else data[1][0] 
                         for data in self.training_data])
            
            NetworkVisualizer.plot_decision_boundary(self.network, X, y)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
    
    # File operations
    def _new_network(self) -> None:
        """Create a new network."""
        self.network = None
        self.builder = NetworkBuilder()
        self.training_controller = None
        self._clear_layers()
        self._update_status("Ready for new network")
    
    def _load_network(self) -> None:
        """Load network from file."""
        filename = filedialog.askopenfilename(
            title="Load Network",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.network = NeuralNetwork(1)  # Dummy input size
                self.network.load(filename)
                self.training_controller = TrainingController(self.network)
                self._update_status(f"Loaded network: {self.network.get_architecture()}")
                messagebox.showinfo("Success", "Network loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load network: {str(e)}")
    
    def _save_network(self) -> None:
        """Save network to file."""
        if self.network is None:
            messagebox.showerror("Error", "No network to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Network",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.network.save(filename)
                self._update_status(f"Network saved to {filename}")
                messagebox.showinfo("Success", "Network saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save network: {str(e)}")
    
    # Data operations
    def _load_csv_data(self) -> None:
        """Load data from CSV file."""
        filename = filedialog.askopenfilename(
            title="Load CSV Data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                features, targets = DataLoader.load_from_csv(filename)
                dataset = DataLoader.create_dataset(features, targets)
                
                # Split into train/test
                self.training_data, self.test_data = DataLoader.split_dataset(
                    dataset, train_ratio=0.8
                )
                
                self._update_status(f"Loaded {len(dataset)} samples from CSV")
                self.data_info_label.config(text=f"Loaded: {len(self.training_data)} train, {len(self.test_data)} test samples")
                messagebox.showinfo("Success", 
                                  f"Loaded {len(self.training_data)} training samples "
                                  f"and {len(self.test_data)} test samples")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def _load_json_data(self) -> None:
        """Load data from JSON file."""
        filename = filedialog.askopenfilename(
            title="Load JSON Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                dataset = DataLoader.load_from_json(filename)
                
                # Split into train/test
                self.training_data, self.test_data = DataLoader.split_dataset(
                    dataset, train_ratio=0.8
                )
                
                self._update_status(f"Loaded {len(dataset)} samples from JSON")
                self.data_info_label.config(text=f"Loaded: {len(self.training_data)} train, {len(self.test_data)} test samples")
                messagebox.showinfo("Success", 
                                  f"Loaded {len(self.training_data)} training samples "
                                  f"and {len(self.test_data)} test samples")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def _generate_data(self) -> None:
        """Generate synthetic data."""
        dialog = DataGeneratorDialog(self.root)
        self.root.wait_window(dialog.top)
        
        if dialog.result:
            self.training_data = dialog.result['train']
            self.test_data = dialog.result['test']
            self._update_status(f"Generated {len(self.training_data)} training samples")
            self.data_info_label.config(text=f"Generated: {len(self.training_data)} train, {len(self.test_data)} test samples")
            messagebox.showinfo("Success", 
                              f"Generated {len(self.training_data)} training samples "
                              f"and {len(self.test_data)} test samples")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """Neural Network Learning System
        
A practical application demonstrating
"What AI should learn from software engineering"

This system allows you to:
• Create neural network architectures
• Configure training parameters
• Train on various datasets
• Visualize results

Developed as a software engineering project."""
        
        messagebox.showinfo("About", about_text)


class DataGeneratorDialog:
    """Dialog for generating synthetic data."""
    
    def __init__(self, parent):
        """Initialize the dialog."""
        self.top = tk.Toplevel(parent)
        self.top.title("Generate Synthetic Data")
        self.top.geometry("400x300")
        
        self.result = None
        
        # Data type
        ttk.Label(self.top, text="Data Type:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.data_type = tk.StringVar(value="classification")
        ttk.Radiobutton(self.top, text="Classification", variable=self.data_type,
                       value="classification").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(self.top, text="Regression", variable=self.data_type,
                       value="regression").grid(row=0, column=2, padx=5, pady=5)
        ttk.Radiobutton(self.top, text="XOR", variable=self.data_type,
                       value="xor").grid(row=0, column=3, padx=5, pady=5)
        
        # Parameters
        ttk.Label(self.top, text="Number of Samples:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.n_samples = tk.IntVar(value=200)
        ttk.Spinbox(self.top, from_=10, to=10000, textvariable=self.n_samples,
                   width=15).grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        
        ttk.Label(self.top, text="Number of Features:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.n_features = tk.IntVar(value=2)
        ttk.Spinbox(self.top, from_=1, to=100, textvariable=self.n_features,
                   width=15).grid(row=2, column=1, columnspan=2, padx=5, pady=5)
        
        ttk.Label(self.top, text="Number of Classes:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.n_classes = tk.IntVar(value=2)
        ttk.Spinbox(self.top, from_=2, to=10, textvariable=self.n_classes,
                   width=15).grid(row=3, column=1, columnspan=2, padx=5, pady=5)
        
        ttk.Label(self.top, text="Noise Level:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.noise = tk.DoubleVar(value=0.1)
        ttk.Entry(self.top, textvariable=self.noise, width=15).grid(row=4, column=1, columnspan=2, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.top)
        button_frame.grid(row=5, column=0, columnspan=4, pady=20)
        
        ttk.Button(button_frame, text="Generate", command=self._generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.top.destroy).pack(side=tk.LEFT, padx=5)
    
    def _generate(self):
        """Generate the data."""
        try:
            data_type = self.data_type.get()
            n_samples = self.n_samples.get()
            n_features = self.n_features.get()
            n_classes = self.n_classes.get()
            noise = self.noise.get()
            
            if data_type == "classification":
                features, targets = DataGenerator.generate_classification_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    noise=noise
                )
            elif data_type == "regression":
                features, targets = DataGenerator.generate_regression_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    noise=noise
                )
            elif data_type == "xor":
                features, targets = DataGenerator.generate_xor_data(
                    n_samples=n_samples,
                    noise=noise
                )
            
            # Create dataset
            dataset = DataLoader.create_dataset(features, targets)
            train_data, test_data = DataLoader.split_dataset(dataset, train_ratio=0.8)
            
            self.result = {
                'train': train_data,
                'test': test_data
            }
            
            self.top.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")