#!/bin/bash

set -e

echo "🔧 Starting ExoplanetAI Backend..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    print_status "Please run this script from the project root directory"
    exit 1
fi

cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs temp_data ml_models

# Start the server
print_success "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8001 --reload --log-level info

print_success "Backend server started!"
echo "  API: http://localhost:8001"
echo "  Docs: http://localhost:8001/docs"
echo "  Health: http://localhost:8001/health"
