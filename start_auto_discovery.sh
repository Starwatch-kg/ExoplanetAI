#!/bin/bash

# ExoplanetAI Auto Discovery Pipeline Startup Script
# Запускает полный пайплайн автоматического обнаружения экзопланет

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
PYTHON_ENV=".venv"
NODE_ENV="node_modules"
BACKEND_PORT=8001
FRONTEND_PORT=5173

echo -e "${PURPLE} ExoplanetAI Auto Discovery Pipeline v2.0${NC}"
echo -e "${CYAN}=================================================${NC}"
echo -e "${BLUE} Automated Exoplanet Discovery System${NC}"
echo -e "${BLUE} Real-time NASA/MAST data ingestion${NC}"
echo -e "${BLUE} ML-powered candidate detection${NC}"
echo -e "${BLUE} Model versioning and deployment${NC}"
echo -e "${CYAN}=================================================${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Backend URL: http://localhost:$BACKEND_PORT"
echo "  Confidence Threshold: $CONFIDENCE_THRESHOLD"
echo "  Check Interval: $CHECK_INTERVAL hours"
echo ""

# Function to check if backend is running
check_backend() {
    echo -e "${BLUE}Checking backend status...${NC}"
    if curl -s -f "$BACKEND_URL/api/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Backend is not responding${NC}"
        echo "  Please start the backend first: cd backend && python main.py"
        return 1
    fi
}

# Function to start discovery service
start_discovery() {
    echo ""
    echo -e "${BLUE}Starting automated discovery service...${NC}"
    
    RESPONSE=$(curl -s -X POST "$BACKEND_URL/api/v1/auto-discovery/start" \
        -H "Content-Type: application/json" \
        -d "{
            \"confidence_threshold\": $CONFIDENCE_THRESHOLD,
            \"check_interval_hours\": $CHECK_INTERVAL,
            \"max_concurrent_tasks\": $MAX_CONCURRENT
        }")
    
    STATUS=$(echo "$RESPONSE" | jq -r '.status' 2>/dev/null || echo "error")
    
    if [ "$STATUS" = "started" ] || [ "$STATUS" = "already_running" ]; then
        echo -e "${GREEN}✓ Discovery service is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Failed to start discovery service${NC}"
        echo "  Response: $RESPONSE"
        return 1
    fi
}

# Function to start scheduler
start_scheduler() {
    echo ""
    echo -e "${BLUE}Starting task scheduler...${NC}"
    
    RESPONSE=$(curl -s -X POST "$BACKEND_URL/api/v1/scheduler/start")
    STATUS=$(echo "$RESPONSE" | jq -r '.status' 2>/dev/null || echo "error")
    
    if [ "$STATUS" = "started" ] || [ "$STATUS" = "already_running" ]; then
        echo -e "${GREEN}✓ Scheduler is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Failed to start scheduler${NC}"
        echo "  Response: $RESPONSE"
        return 1
    fi
}

# Function to setup default schedule
setup_schedule() {
    echo ""
    echo -e "${BLUE}Setting up automated schedule...${NC}"
    
    # Main discovery cycle - every 6 hours
    echo "  Creating main discovery task (every 6 hours)..."
    curl -s -X POST "$BACKEND_URL/api/v1/scheduler/tasks/cron" \
        -H "Content-Type: application/json" \
        -d '{
            "task_id": "main_discovery",
            "name": "Main Discovery Cycle",
            "cron_expression": "0 */6 * * *",
            "max_retries": 3
        }' > /dev/null
    
    # Health check - every 30 minutes
    echo "  Creating health check task (every 30 minutes)..."
    curl -s -X POST "$BACKEND_URL/api/v1/scheduler/tasks/interval" \
        -H "Content-Type: application/json" \
        -d '{
            "task_id": "health_check",
            "name": "System Health Check",
            "hours": 0,
            "minutes": 30,
            "seconds": 0,
            "max_retries": 1
        }' > /dev/null
    
    echo -e "${GREEN}✓ Schedule configured${NC}"
}

# Function to show status
show_status() {
    echo ""
    echo -e "${BLUE}Current Status:${NC}"
    
    STATUS=$(curl -s "$BACKEND_URL/api/v1/auto-discovery/status")
    
    IS_RUNNING=$(echo "$STATUS" | jq -r '.is_running' 2>/dev/null || echo "false")
    TOTAL_PROCESSED=$(echo "$STATUS" | jq -r '.total_processed' 2>/dev/null || echo "0")
    TOTAL_CANDIDATES=$(echo "$STATUS" | jq -r '.total_candidates' 2>/dev/null || echo "0")
    HIGH_CONFIDENCE=$(echo "$STATUS" | jq -r '.high_confidence_candidates' 2>/dev/null || echo "0")
    
    echo "  Running: $IS_RUNNING"
    echo "  Total Processed: $TOTAL_PROCESSED"
    echo "  Total Candidates: $TOTAL_CANDIDATES"
    echo "  High Confidence: $HIGH_CONFIDENCE"
}

# Function to show dashboard URL
show_dashboard() {
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✓ Automated Discovery System is Running!"
    echo -e "==========================================${NC}"
    echo ""
    echo "📊 Dashboard: http://localhost:5173/auto-discovery"
    echo "📚 API Docs: $BACKEND_URL/docs"
    echo "📈 Monitoring: $BACKEND_URL/api/v1/monitoring/dashboard"
    echo ""
    echo "Useful commands:"
    echo "  View status:     curl $BACKEND_URL/api/v1/auto-discovery/status"
    echo "  View candidates: curl $BACKEND_URL/api/v1/auto-discovery/candidates/top?limit=10"
    echo "  Stop service:    curl -X POST $BACKEND_URL/api/v1/auto-discovery/stop"
    echo ""
}

# Main execution
main() {
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}⚠ jq is not installed. Install it for better output formatting.${NC}"
        echo "  Ubuntu/Debian: sudo apt-get install jq"
        echo "  macOS: brew install jq"
        echo ""
    fi
    
    # Check backend
    if ! check_backend; then
        exit 1
    fi
    
    # Start discovery service
    if ! start_discovery; then
        exit 1
    fi
    
    # Start scheduler
    if ! start_scheduler; then
        echo -e "${YELLOW}⚠ Scheduler failed to start, but discovery service is running${NC}"
    fi
    
    # Setup schedule (optional, may fail if tasks already exist)
    setup_schedule || true
    
    # Show status
    show_status
    
    # Show dashboard info
    show_dashboard
}

# Run main function
main
