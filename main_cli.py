#!/usr/bin/env python3
"""
Main entry point for the CLI version of the Neural Network Learning System.
"""

from src.views import NeuralNetworkCLI


def main():
    """Launch the CLI application."""
    cli = NeuralNetworkCLI()
    cli.run()


if __name__ == "__main__":
    main()