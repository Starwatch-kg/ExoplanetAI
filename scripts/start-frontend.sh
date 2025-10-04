#!/bin/bash

set -e

echo "🎨 Starting ExoplanetAI Frontend..."

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
if [ ! -d "frontend" ]; then
    print_status "Navigating to project root directory"
    cd "$(dirname "$0")/.."
    if [ ! -d "frontend" ]; then
        echo "Error: frontend directory not found"
        exit 1
    fi
fi

cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
fi

# Start the development server
print_success "Starting Vite development server..."
npm run dev

print_success "Frontend server started!"
echo "  App: http://localhost:5173"
