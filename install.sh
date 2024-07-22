#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Download the Brewfile
echo "Downloading Brewfile..."
curl -fsSL https://raw.githubusercontent.com/KruxAI/ragbuilder/main/Brewfile -o Brewfile

# Install Homebrew packages using Brewfile
echo "Installing Homebrew packages..."
brew bundle install --file=Brewfile
 
echo "Installing ragbuilder..."
python3 -m pip install ragbuilder

echo "Setup completed successfully."
