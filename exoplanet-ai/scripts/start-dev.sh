#!/bin/bash

set -e

echo "🚀 Starting ExoplanetAI Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "exoplanet-ai" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

cd exoplanet-ai

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs temp_data ml_models

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d'.' -f1-2)
print_status "Python version: $PYTHON_VERSION"

# Check Node.js version
print_status "Checking Node.js version..."
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed."
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$(echo "$NODE_VERSION < 18" | bc -l)" = 1 ]; then
    print_warning "Node.js version $NODE_VERSION is below recommended 18+"
fi

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

# Check if main.py exists and is valid
if [ ! -f "main.py" ]; then
    print_error "main.py not found in backend directory"
    exit 1
fi

print_success "Backend setup complete"

# Setup frontend
print_status "Setting up frontend..."
cd ../frontend

if [ ! -d "node_modules" ]; then
    print_status "Installing frontend dependencies..."
    npm install
fi

print_success "Frontend setup complete"

cd ..

# Start services
print_status "Starting development servers..."

# Start backend
print_status "Starting backend server..."
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!
cd ..

print_success "Backend started on http://localhost:8001"

# Start frontend
print_status "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

print_success "Frontend started on http://localhost:5173"

echo
print_success "🎉 Development environment started successfully!"
echo
print_status "URLs:"
echo "  Frontend: http://localhost:5173"
echo "  Backend API: http://localhost:8001/api/v1"
echo "  Health Check: http://localhost:8001/health"
echo "  API Docs: http://localhost:8001/docs"
echo
print_status "Press Ctrl+C to stop all services"
echo

# Wait for user interrupt
trap 'echo -e "\n${YELLOW}Shutting down servers...${NC}"; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

# Keep script running
wait
