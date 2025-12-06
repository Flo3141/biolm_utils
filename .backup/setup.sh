#!/bin/bash
# BioLM Setup Script - Minimal Friction Installation

set -e  # Exit on any error

echo "================================================"
echo "   BioLM Framework + Plugins Setup"
echo "================================================"
echo ""

# Check prerequisites
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is required but not installed."
    exit 1
fi

if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry is required but not installed."
    echo "   Install from: https://python-poetry.org/docs/#installation"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$(printf '%s\n' "$PYTHON_VERSION" "3.10" | sort -V | head -n1)" != "3.10" ]]; then
    echo "❌ Error: Python 3.10+ is required. Current: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Prerequisites: Git, Poetry, Python $PYTHON_VERSION"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Setting up BioLM Framework..."
echo "  Repository: $SCRIPT_DIR"
echo ""

# Set up Poetry environment
echo "Installing dependencies with Poetry..."
cd "$SCRIPT_DIR"
poetry install

echo ""
echo "================================================"
echo "   ✓ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Navigate to the framework:"
echo "   cd $SCRIPT_DIR"
echo ""
echo "2. Run your first experiment:"
echo "   poetry run biolm fine-tune --config-path ./biolm/plugins/saluki/exampleconfigs/tokenize_fine-tune.yaml"
echo ""
echo "3. View available commands:"
echo "   poetry run biolm --help"
echo ""
echo "Available Plugins (built-in):"
echo "  • Saluki (RNA analysis)"
echo "  • XLNet (Protein analysis)"
echo ""
