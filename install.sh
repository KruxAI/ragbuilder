#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Download the Brewfile
echo "Downloading Brewfile..."
curl -fsSL https://raw.githubusercontent.com/KruxAI/ragbuilder-readme/main/Brewfile -o Brewfile

# Install Homebrew packages using Brewfile
echo "Installing Homebrew packages..."
brew bundle install --file=Brewfile

# Proceed with your Python package setup
VENV_PATH="./venv"

echo "Creating Python virtual environment in the current folder at $VENV_PATH..."
python3 -m venv $VENV_PATH

echo "Activating virtual environment..."
source $VENV_PATH/bin/activate

echo "Installing ragbuilder..."
python3 -m pip install ragbuilder

echo "Setup completed successfully."
