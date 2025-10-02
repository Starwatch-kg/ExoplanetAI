#!/bin/bash

set -e

echo "🚀 Starting ExoplanetAI deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker --version | grep -q "compose"; then
    print_error "Docker Compose is not available. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p logs temp_data ml_models

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file..."
    cp config/.env.example .env 2>/dev/null || echo "No .env.example found"
fi

# Build and start services
print_status "Building Docker images..."
docker-compose build --parallel

print_status "Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_status "Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    print_success "Services started successfully!"

    # Show service status
    echo
    print_status "Service Status:"
    docker-compose ps

    echo
    print_status "Application URLs:"
    echo "  Frontend: http://localhost"
    echo "  Backend API: http://localhost/api/v1"
    echo "  Health Check: http://localhost/health"
    echo "  API Documentation: http://localhost/docs"

    echo
    print_status "Useful commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart services: docker-compose restart"
    echo "  View service status: docker-compose ps"

else
    print_error "Some services failed to start. Check logs:"
    docker-compose logs
    exit 1
fi

# Run health check
print_status "Running health check..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    print_success "Health check passed!"
else
    print_warning "Health check failed. Services might still be starting."
fi

print_success "Deployment completed successfully!"
echo
print_status "You can now access the application at http://localhost"
