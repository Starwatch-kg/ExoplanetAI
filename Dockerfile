# Multi-stage Docker build for ExoplanetAI

# Stage 1: Frontend build
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY exoplanet-ai/frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY exoplanet-ai/frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Backend build
FROM python:3.11-slim AS backend-builder

WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY exoplanet-ai/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Production image
FROM python:3.11-slim

LABEL maintainer="ExoplanetAI Team"
LABEL version="2.0.0"
LABEL description="Advanced AI-powered exoplanet detection system"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Create directories
RUN mkdir -p /app/{frontend,backend,ml,logs,static} && \
    chown -R app:app /app

# Copy frontend build
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Copy backend code
COPY --chown=app:app exoplanet-ai/backend /app/backend

# Copy ML models and configurations
COPY --chown=app:app exoplanet-ai/ml /app/ml
COPY --chown=app:app exoplanet-ai/config /app/config

# Copy startup scripts
COPY --chown=app:app scripts/ /app/scripts/

# Install Python dependencies
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# Create nginx configuration
RUN rm /etc/nginx/sites-enabled/default
COPY --chown=app:app config/nginx.conf /etc/nginx/sites-available/exoplanet-ai
RUN ln -s /etc/nginx/sites-available/exoplanet-ai /etc/nginx/sites-enabled/

# Create supervisor configuration
COPY --chown=app:app config/supervisord.conf /etc/supervisor/conf.d/exoplanet-ai.conf

# Set proper permissions
RUN chmod +x /app/scripts/*

# Switch to app user
USER app

# Expose ports
EXPOSE 80 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/exoplanet-ai.conf"]
