#!/usr/bin/env bash
#
# Add Documents to Central Vector Database
#
# Indexes document(s) and adds them to the central vector_db/ store.
#
# Usage:
#   ./add_to_vector_db.sh <document_or_directory>
#   ./add_to_vector_db.sh data/document.pdf
#   ./add_to_vector_db.sh data/documents/
#
# Environment:
#   SPEED_MODE - Set to "eco" for 50% cost savings (overnight batch)
#   See .env for all configuration options
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
echo "  Add to Central Vector Database"
echo "========================================"
echo ""

# Check arguments
if [ $# -eq 0 ]; then
    print_error "No input file or directory specified"
    echo ""
    echo "Usage: ./add_to_vector_db.sh <document_or_directory>"
    echo ""
    echo "Examples:"
    echo "  ./add_to_vector_db.sh data/document.pdf"
    echo "  ./add_to_vector_db.sh data/documents/"
    echo ""
    exit 1
fi

INPUT_PATH="$1"

# Check if input exists
if [ ! -e "$INPUT_PATH" ]; then
    print_error "Input not found: $INPUT_PATH"
    exit 1
fi

# Check if input is file or directory
if [ -f "$INPUT_PATH" ]; then
    print_info "Indexing single document: $INPUT_PATH"
elif [ -d "$INPUT_PATH" ]; then
    print_info "Indexing directory: $INPUT_PATH"
    # Count supported files
    FILE_COUNT=$(find "$INPUT_PATH" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.pptx" -o -name "*.xlsx" \) | wc -l | tr -d ' ')
    print_info "Found $FILE_COUNT document(s)"
else
    print_error "Invalid input: $INPUT_PATH"
    exit 1
fi

# Check dependencies
print_info "Checking dependencies..."
if ! command -v uv &> /dev/null; then
    print_error "uv not found. Please install uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check .env
if [ ! -f .env ]; then
    print_error ".env file not found. Copy .env.example and configure API keys."
    exit 1
fi

# Get speed mode from .env
SPEED_MODE=$(grep "^SPEED_MODE=" .env 2>/dev/null | cut -d'=' -f2 | tr -d '"' || echo "fast")
if [ "$SPEED_MODE" = "eco" ]; then
    print_warning "Running in ECO mode (50% cheaper, 15-30 min)"
else
    print_info "Running in FAST mode (2-3 min, full price)"
fi

echo ""
print_info "Starting indexing pipeline..."
echo ""

# Run pipeline
uv run python run_pipeline.py "$INPUT_PATH"

# Check if pipeline succeeded
if [ $? -ne 0 ]; then
    print_error "Pipeline failed"
    exit 1
fi

echo ""
print_success "Pipeline completed successfully"

# Find the output directory (most recent in output/)
OUTPUT_DIR=$(find output -maxdepth 1 -type d -name "*" | grep -v "^output$" | sort -r | head -n 1)

if [ -z "$OUTPUT_DIR" ]; then
    print_error "Could not find output directory"
    exit 1
fi

print_info "Output directory: $OUTPUT_DIR"

# Check if vector store was created
VECTOR_STORE="$OUTPUT_DIR/phase4_vector_store"
if [ ! -d "$VECTOR_STORE" ]; then
    # Try hybrid_store as fallback
    VECTOR_STORE="$OUTPUT_DIR/hybrid_store"
    if [ ! -d "$VECTOR_STORE" ]; then
        print_error "Vector store not found in output directory"
        exit 1
    fi
fi

print_info "Vector store: $VECTOR_STORE"

# Create central vector_db if it doesn't exist
if [ ! -d "vector_db" ]; then
    print_info "Creating central vector_db/"
    mkdir -p vector_db
fi

# Check if vector_db is empty
if [ -z "$(ls -A vector_db)" ]; then
    print_info "Initializing central vector database..."
    cp -r "$VECTOR_STORE"/* vector_db/
    print_success "Vector store copied to vector_db/"
else
    print_warning "Central vector_db already exists"
    print_info "To merge stores, use run_pipeline.py with existing vector_db/"
    print_info "Or manually copy files from: $VECTOR_STORE"
    echo ""
    print_info "Current store preserved at: $VECTOR_STORE"
fi

echo ""
print_success "Document(s) indexed successfully!"
print_info "Vector store: vector_db/"
echo ""
print_info "To use with agent:"
echo "  ./run_cli.sh"
echo ""
