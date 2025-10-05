#!/bin/bash

# ExoplanetAI - Render Deployment Readiness Check
# Проверяет готовность проекта к развертыванию на Render

set -e

echo "🚀 ExoplanetAI - Render Deployment Readiness Check"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Check function
check_passed() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_failed() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# 1. Check required files
echo "📁 Checking required files..."
echo "----------------------------"

[ -f "render.yaml" ] && check_passed "render.yaml exists" || check_failed "render.yaml exists"
[ -f "backend/requirements.txt" ] && check_passed "backend/requirements.txt exists" || check_failed "backend/requirements.txt exists"
[ -f "backend/main.py" ] && check_passed "backend/main.py exists" || check_failed "backend/main.py exists"
[ -f "backend/Dockerfile.render" ] && check_passed "backend/Dockerfile.render exists" || check_failed "backend/Dockerfile.render exists"
[ -f "backend/gunicorn_config.py" ] && check_passed "backend/gunicorn_config.py exists" || check_failed "backend/gunicorn_config.py exists"
[ -f ".env.render.example" ] && check_passed ".env.render.example exists" || check_failed ".env.render.example exists"
[ -f "RENDER_DEPLOYMENT.md" ] && check_passed "RENDER_DEPLOYMENT.md exists" || check_failed "RENDER_DEPLOYMENT.md exists"

echo ""

# 2. Check Python dependencies
echo "🐍 Checking Python dependencies..."
echo "-----------------------------------"

if [ -f "backend/requirements.txt" ]; then
    grep -q "fastapi" backend/requirements.txt && check_passed "FastAPI dependency" || check_failed "FastAPI dependency"
    grep -q "uvicorn" backend/requirements.txt && check_passed "Uvicorn dependency" || check_failed "Uvicorn dependency"
    grep -q "gunicorn" backend/requirements.txt && check_passed "Gunicorn dependency" || check_failed "Gunicorn dependency"
    grep -q "redis" backend/requirements.txt && check_passed "Redis dependency" || check_failed "Redis dependency"
    grep -q "astroquery" backend/requirements.txt && check_passed "Astroquery dependency" || check_failed "Astroquery dependency"
    grep -q "lightkurve" backend/requirements.txt && check_passed "Lightkurve dependency" || check_failed "Lightkurve dependency"
fi

echo ""

# 3. Check main.py configuration
echo "⚙️  Checking main.py configuration..."
echo "-------------------------------------"

if [ -f "backend/main.py" ]; then
    grep -q "FastAPI" backend/main.py && check_passed "FastAPI import" || check_failed "FastAPI import"
    grep -q "CORSMiddleware" backend/main.py && check_passed "CORS middleware" || check_failed "CORS middleware"
    grep -q "get_cache" backend/main.py && check_passed "Cache initialization" || check_failed "Cache initialization"
    grep -q "create_api_router" backend/main.py && check_passed "API router" || check_failed "API router"
fi

echo ""

# 4. Check environment variables
echo "🔐 Checking environment configuration..."
echo "----------------------------------------"

if [ -f ".env.render.example" ]; then
    grep -q "PORT" .env.render.example && check_passed "PORT variable defined" || check_failed "PORT variable defined"
    grep -q "ENVIRONMENT" .env.render.example && check_passed "ENVIRONMENT variable defined" || check_failed "ENVIRONMENT variable defined"
    grep -q "CACHE_REDIS_URL" .env.render.example && check_passed "CACHE_REDIS_URL variable defined" || check_failed "CACHE_REDIS_URL variable defined"
    grep -q "JWT_SECRET_KEY" .env.render.example && check_passed "JWT_SECRET_KEY variable defined" || check_failed "JWT_SECRET_KEY variable defined"
    grep -q "ALLOWED_ORIGINS" .env.render.example && check_passed "ALLOWED_ORIGINS variable defined" || check_failed "ALLOWED_ORIGINS variable defined"
fi

echo ""

# 5. Check Git configuration
echo "📦 Checking Git configuration..."
echo "--------------------------------"

if [ -d ".git" ]; then
    check "Git repository initialized"
    
    if git remote -v | grep -q "origin"; then
        check "Git remote 'origin' configured"
    else
        warn "Git remote 'origin' not configured - add with: git remote add origin <url>"
    fi
    
    if [ -f ".gitignore" ]; then
        check ".gitignore exists"
        grep -q ".env" .gitignore && check_passed ".env in .gitignore" || warn ".env not in .gitignore"
        grep -q "__pycache__" .gitignore && check_passed "__pycache__ in .gitignore" || warn "__pycache__ not in .gitignore"
    else
        warn ".gitignore not found"
    fi
else
    warn "Git repository not initialized - run: git init"
fi

echo ""

# 6. Check render.yaml configuration
echo "☁️  Checking render.yaml configuration..."
echo "-----------------------------------------"

if [ -f "render.yaml" ]; then
    grep -q "type: web" render.yaml && check_passed "Backend web service defined" || check_failed "Backend web service defined"
    grep -q "type: redis" render.yaml && check_passed "Redis service defined" || check_failed "Redis service defined"
    grep -q "env: python" render.yaml && check_passed "Python environment specified" || check_failed "Python environment specified"
    grep -q "buildCommand" render.yaml && check_passed "Build command defined" || check_failed "Build command defined"
    grep -q "startCommand" render.yaml && check_passed "Start command defined" || check_failed "Start command defined"
    grep -q "healthCheckPath" render.yaml && check_passed "Health check path defined" || check_failed "Health check path defined"
fi

echo ""

# 7. Check API health endpoint
echo "🏥 Checking API health endpoint..."
echo "----------------------------------"

if grep -r "\/health" backend/api/routes/ > /dev/null 2>&1; then
    check "Health endpoint exists in routes"
else
    warn "Health endpoint not found - should be at /api/v1/health"
fi

echo ""

# 8. Check for common issues
echo "🔍 Checking for common issues..."
echo "--------------------------------"

# Check for hardcoded ports
if grep -r "localhost:8001" backend/ --include="*.py" > /dev/null 2>&1; then
    warn "Hardcoded localhost:8001 found - use environment variable PORT"
else
    check "No hardcoded ports in backend"
fi

# Check for .env files in repo
if [ -f ".env" ] || [ -f "backend/.env" ]; then
    warn ".env file found - should not be committed to Git"
else
    check "No .env files in repository"
fi

# Check for large files
LARGE_FILES=$(find . -type f -size +10M 2>/dev/null | grep -v ".git" | wc -l)
if [ "$LARGE_FILES" -gt 0 ]; then
    warn "Found $LARGE_FILES files larger than 10MB - consider using Git LFS"
else
    check "No large files detected"
fi

echo ""

# 9. Frontend check (if exists)
if [ -d "frontend" ]; then
    echo "🎨 Checking frontend configuration..."
    echo "------------------------------------"
    
    [ -f "frontend/package.json" ] && check_passed "package.json exists" || warn "package.json not found"
    [ -f "frontend/vite.config.ts" ] && check_passed "vite.config.ts exists" || warn "vite.config.ts not found"
    
    if [ -f "frontend/package.json" ]; then
        grep -q "\"build\":" frontend/package.json && check_passed "Build script defined" || warn "Build script not defined"
    fi
    
    echo ""
fi

# 10. Security check
echo "🔒 Security check..."
echo "--------------------"

# Check for exposed secrets
if grep -r "sk-" . --include="*.py" --include="*.ts" --include="*.tsx" --include="*.js" > /dev/null 2>&1; then
    warn "Potential API key found in code - use environment variables"
else
    check "No exposed API keys detected"
fi

# Check for SQL injection vulnerabilities
if grep -r "execute.*%.*%" backend/ --include="*.py" > /dev/null 2>&1; then
    warn "Potential SQL injection vulnerability - use parameterized queries"
else
    check "No obvious SQL injection vulnerabilities"
fi

echo ""

# Summary
echo "=================================================="
echo "📊 Summary"
echo "=================================================="
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC} $FAILED"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ Your project is ready for Render deployment!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Push code to GitHub: git push origin main"
    echo "2. Go to https://dashboard.render.com"
    echo "3. Create new Blueprint from render.yaml"
    echo "4. Configure environment variables"
    echo "5. Deploy!"
    exit 0
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Your project is mostly ready, but has some warnings.${NC}"
    echo "Review warnings above and fix if necessary."
    exit 0
else
    echo -e "${RED}❌ Your project has issues that need to be fixed before deployment.${NC}"
    echo "Fix the failed checks above and run this script again."
    exit 1
fi
