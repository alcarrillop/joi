#!/bin/bash

# Quick setup for Joi - AI English Learning Assistant
# Optimized for Railway, GitHub Codespaces, and similar environments

set -e

echo "🚀 Setting up Joi..."

# Install uv if needed
if ! command -v uv &> /dev/null; then
    pip install uv
fi

# Install dependencies and project
uv sync
uv pip install -e '.[dev]'

# Verify setup
python3 -c "from agent.graph.graph import graph; print('✅ Setup complete - Graph created with', len(graph.get_graph().nodes), 'nodes')"

echo "✅ Ready to run: langgraph dev" 