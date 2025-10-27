#!/bin/bash
#
# Start SUJBOT2 Backend Only
#
# Run this in one terminal, then start frontend separately
#

set -e

echo "🚀 Starting SUJBOT2 Backend..."

# Check if we're in the backend directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the backend/ directory"
    echo "  cd backend && ./start_backend.sh"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -r requirements.txt -q

# Start server
echo "✅ Starting FastAPI on http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
uv run python main.py
