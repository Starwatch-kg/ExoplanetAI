#!/bin/bash

echo "🎨 Starting ExoplanetAI Frontend Only (Local Development)"
echo "======================================================="

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

# Setup frontend
print_status "Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    print_status "Installing frontend dependencies..."
    npm install
fi

print_success "Frontend setup complete"

# Start frontend server
print_status "Starting Vite development server..."
npm run dev

print_success "Frontend server started!"
echo "  App: http://localhost:5173"
