#!/bin/bash
# install.sh - BioLM 2.0 Framework Installation Script
# Installs the framework only. Plugins install themselves separately.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Poetry is installed
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed"
        echo ""
        echo "Install Poetry with:"
        echo "  curl -sSL https://install.python-poetry.org | python3 -"
        echo ""
        echo "Or visit: https://python-poetry.org/docs/#installation"
        exit 1
    fi
    print_success "Poetry found: $(poetry --version)"
}

# Install framework
install_framework() {
    print_header "Installing BioLM Framework"
    
    cd "$SCRIPT_DIR"
    
    print_info "Running: poetry install --no-interaction"
    poetry install --no-interaction
    
    print_success "Framework installed"
}

# Verify plugin registration
verify_plugins() {
    print_header "Checking for Registered Plugins"
    
    cd "$SCRIPT_DIR"
    
    poetry run python -c "
import importlib.metadata

eps = list(importlib.metadata.entry_points(group='biolm.plugins'))
if eps:
    print(f'Found {len(eps)} registered plugin(s):')
    for ep in eps:
        print(f'  ✓ {ep.name}')
else:
    print('No plugins installed yet.')
    print('')
    print('To install plugins, run their installation scripts:')
    print('  cd /path/to/plugin')
    print('  ./install.sh')
" || true
}

# Main installation flow
main() {
    print_header "BioLM 2.0 Framework Installation"
    echo ""
    
    # Check prerequisites
    check_poetry
    echo ""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                echo "Usage: $0"
                echo ""
                echo "Installs the BioLM 2.0 framework."
                echo ""
                echo "To install plugins after installing the framework:"
                echo "  cd /path/to/plugin"
                echo "  ./install.sh"
                echo ""
                echo "Example:"
                echo "  # Install framework"
                echo "  cd biolm_utils && ./install.sh"
                echo ""
                echo "  # Install Saluki plugin"
                echo "  cd rna_saluki_cnn && ./install.sh"
                echo ""
                echo "  # Install XLNet plugin"
                echo "  cd rna_protein_xlnet && ./install.sh"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Install framework
    install_framework
    echo ""
    
    # Verify installation
    verify_plugins
    echo ""
    
    # Final message
    print_header "Installation Complete!"
    echo ""
    print_success "BioLM 2.0 framework is ready to use"
    echo ""
    print_info "Quick start:"
    echo "  source .venv/bin/activate"
    echo "  biolm --help"
    echo ""
    print_info "To install plugins:"
    echo "  biolm install-plugin <git-url>"
    echo ""
}

# Run main function
main "$@"
