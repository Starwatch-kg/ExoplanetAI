#!/bin/bash

# ExoplanetAI - Render Deployment Readiness Check (Simplified)

echo "🚀 ExoplanetAI - Render Deployment Readiness Check"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

echo "📁 Checking required files..."
echo "----------------------------"

# Required files
FILES=(
    "render.yaml"
    "backend/requirements.txt"
    "backend/main.py"
    "backend/Dockerfile.render"
    "backend/gunicorn_config.py"
    ".env.render.example"
    "RENDER_DEPLOYMENT.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $file missing"
        ((FAILED++))
    fi
done

echo ""
echo "🐍 Checking Python dependencies..."
echo "-----------------------------------"

# Required dependencies
DEPS=("fastapi" "uvicorn" "gunicorn" "redis" "astroquery" "lightkurve")

for dep in "${DEPS[@]}"; do
    if grep -q "$dep" backend/requirements.txt 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $dep dependency found"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $dep dependency missing"
        ((FAILED++))
    fi
done

echo ""
echo "⚙️  Checking configuration..."
echo "-----------------------------"

# Check main.py
if grep -q "FastAPI" backend/main.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC} FastAPI configured"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} FastAPI not configured"
    ((FAILED++))
fi

if grep -q "CORSMiddleware" backend/main.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC} CORS middleware configured"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} CORS middleware not found"
    ((WARNINGS++))
fi

echo ""
echo "📦 Checking Git..."
echo "------------------"

if [ -d ".git" ]; then
    echo -e "${GREEN}✓${NC} Git repository initialized"
    ((PASSED++))
    
    if git remote -v | grep -q "origin" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Git remote configured"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} Git remote not configured"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} Git not initialized"
    ((WARNINGS++))
fi

if [ -f ".gitignore" ]; then
    echo -e "${GREEN}✓${NC} .gitignore exists"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} .gitignore missing"
    ((WARNINGS++))
fi

echo ""
echo "🔒 Security check..."
echo "--------------------"

if [ -f ".env" ] || [ -f "backend/.env" ]; then
    echo -e "${YELLOW}⚠${NC} .env file found - should not be committed"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} No .env files in repository"
    ((PASSED++))
fi

echo ""
echo "=================================================="
echo "📊 Summary"
echo "=================================================="
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC} $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Your project is ready for Render deployment!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Push code to GitHub: git push origin main"
    echo "2. Go to https://dashboard.render.com"
    echo "3. Create new Blueprint from render.yaml"
    echo "4. Configure environment variables"
    echo "5. Deploy!"
    exit 0
else
    echo -e "${RED}❌ Fix the failed checks above before deployment.${NC}"
    exit 1
fi
