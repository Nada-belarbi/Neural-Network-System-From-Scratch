#!/usr/bin/env python3
"""
Main entry point for the GUI version of the Neural Network Learning System.
"""

import tkinter as tk
from src.views import NeuralNetworkGUI


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()