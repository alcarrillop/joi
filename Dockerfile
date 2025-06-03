# Use Python 3.12 with uv package manager
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Install system dependencies including PostgreSQL client
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy dependency files
COPY uv.lock pyproject.toml README.md ./

# Install dependencies
RUN uv sync --frozen --no-cache --no-dev

# Copy application source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set virtual environment path
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install the package in editable mode
RUN uv pip install -e .

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8000}

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/debug/health || exit 1

# Run the application (Railway compatible)
CMD ["sh", "-c", "fastapi run src/agent/interfaces/whatsapp/webhook_endpoint.py --port ${PORT:-8000} --host 0.0.0.0 --workers 1"]
