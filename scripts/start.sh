#!/bin/bash

# Get port from environment variable, default to 8000 if not set
PORT=${PORT:-8000}

echo "Starting application on port $PORT"

# Start the application
exec uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 