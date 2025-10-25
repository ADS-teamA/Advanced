#!/usr/bin/env bash
#
# RAG Agent CLI Launcher
#
# Launches interactive RAG agent with dependency checks.
#
# Usage:
#   ./run_cli.sh                           # Use default vector store (vector_db/)
#   ./run_cli.sh output/my_doc/vector_store # Use specific vector store
#   ./run_cli.sh --debug                   # Enable debug mode
#
# Requirements:
#   - Python 3.10+
#   - uv package manager
#   - ANTHROPIC_API_KEY in .env
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Print header
echo ""
echo "========================================"
echo "  RAG Agent CLI Launcher"
echo "========================================"
echo ""

# Check Python version
print_info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $PYTHON_VERSION found, but $REQUIRED_VERSION or higher required."
    exit 1
fi
print_success "Python $PYTHON_VERSION"

# Check uv package manager
print_info "Checking uv package manager..."
if ! command -v uv &> /dev/null; then
    print_error "uv not found. Please install uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "uv $(uv --version | awk '{print $2}')"

# Check .env file
print_info "Checking environment configuration..."
if [ ! -f .env ]; then
    print_warning ".env file not found"
    print_info "Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        print_warning "Please edit .env and add your ANTHROPIC_API_KEY"
        exit 1
    else
        print_error ".env.example not found"
        exit 1
    fi
fi

# Check ANTHROPIC_API_KEY
if ! grep -q "ANTHROPIC_API_KEY=sk-ant-" .env 2>/dev/null; then
    print_warning "ANTHROPIC_API_KEY not configured in .env"
    print_info "Please add your API key to .env file"
    exit 1
fi
print_success "Environment configured"

# Parse arguments
VECTOR_STORE="vector_db"
DEBUG_FLAG=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --debug)
            DEBUG_FLAG="--debug"
            print_info "Debug mode enabled"
            ;;
        --help|-h)
            echo "Usage: ./run_cli.sh [vector_store_path] [--debug]"
            echo ""
            echo "Arguments:"
            echo "  vector_store_path    Path to vector store (default: vector_db/)"
            echo "  --debug              Enable debug logging"
            echo ""
            echo "Examples:"
            echo "  ./run_cli.sh                              # Use default vector_db/"
            echo "  ./run_cli.sh output/my_doc/vector_store   # Use specific store"
            echo "  ./run_cli.sh --debug                      # Debug mode"
            exit 0
            ;;
        --*)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
        *)
            if [ -d "$arg" ] || [ "$arg" != "--debug" ]; then
                VECTOR_STORE="$arg"
            fi
            ;;
    esac
done

# Check if vector store exists
if [ ! -d "$VECTOR_STORE" ]; then
    print_warning "Vector store not found: $VECTOR_STORE"
    print_info "Create one by running:"
    echo "  ./add_to_vector_db.sh data/your_document.pdf"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch agent
echo ""
print_info "Launching RAG Agent..."
print_info "Vector store: $VECTOR_STORE"
echo ""

# Run with uv
exec uv run python -m src.agent.cli --vector-store "$VECTOR_STORE" $DEBUG_FLAG $EXTRA_ARGS
