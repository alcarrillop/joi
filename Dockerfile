# Multi-stage build for smaller image size
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Install only necessary build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy dependency files
COPY uv.lock pyproject.toml README.md ./

# Install dependencies in a virtual environment
RUN uv sync --frozen --no-cache --no-dev

# Install the package in editable mode
COPY src/ ./src/
RUN uv pip install -e . --no-deps

# Production stage - smaller base image
FROM python:3.12-slim-bookworm AS production

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user for security (before copying files)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory and ownership
WORKDIR /app
RUN chown appuser:appuser /app

# Switch to app user
USER appuser

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Set virtual environment path
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check for Railway (using PORT env var at runtime)
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/debug/health || exit 1

# Run the application (Railway compatible)
CMD ["sh", "-c", "uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
