#!/bin/bash

echo "🚀 Starting ExoplanetAI Backend Only (Local Development)"
echo "======================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Setup backend
print_status "Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

print_status "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Check if main.py exists
if [ ! -f "main.py" ]; then
    print_status "Error: main.py not found in backend directory"
    exit 1
fi

print_success "Backend setup complete"

# Start backend server
print_status "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

print_success "Backend server started!"
echo "  API: http://localhost:8001"
echo "  Docs: http://localhost:8001/docs"
echo "  Health: http://localhost:8001/health"
