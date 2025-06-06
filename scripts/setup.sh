#!/bin/bash

# Joi - AI English Learning Assistant Setup Script
# This script sets up the development environment from scratch

set -e  # Exit on any error

echo "🚀 Setting up Joi - AI English Learning Assistant..."
echo "=================================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.12"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    echo "✅ Python $python_version (meets requirement: >= $required_version)"
else
    echo "❌ Python $python_version is too old. Please install Python 3.12 or higher."
    exit 1
fi

# Install uv if not available
echo ""
echo "📦 Checking uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Installing uv..."
    pip install uv
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi

# Sync dependencies
echo ""
echo "📚 Installing project dependencies..."
uv sync
echo "✅ Dependencies installed successfully"

# Install project in development mode
echo ""
echo "🔧 Installing project in development mode..."
uv pip install -e '.[dev]'
echo "✅ Project installed in editable mode"

# Verify imports work
echo ""
echo "🧪 Verifying package imports..."
if python3 -c "from agent.graph.graph import graph; from agent.core.database import get_db_connection" 2>/dev/null; then
    echo "✅ Core imports working correctly"
else
    echo "❌ Import verification failed"
    exit 1
fi

# Check if .env file exists
echo ""
echo "🔐 Checking environment configuration..."
if [ -f ".env" ]; then
    echo "✅ .env file found"
    
    # Check for required environment variables
    required_vars=("DATABASE_URL" "GROQ_API_KEY")
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" .env; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -eq 0 ]; then
        echo "✅ Required environment variables are present"
    else
        echo "⚠️  Missing required environment variables: ${missing_vars[*]}"
        echo "   Please check .env.example for reference"
    fi
else
    echo "⚠️  .env file not found"
    if [ -f ".env.example" ]; then
        echo "   Copy .env.example to .env and fill in your values:"
        echo "   cp .env.example .env"
    else
        echo "   Create a .env file with your configuration"
    fi
fi

# Database setup check
echo ""
echo "🗄️  Database setup..."
if [ -f "scripts/init_db.sql" ]; then
    echo "✅ Database schema script available"
    echo "   Run this when you have database access:"
    echo "   psql \$DATABASE_URL -f scripts/init_db.sql"
else
    echo "❌ Database schema script not found"
fi

# Final verification
echo ""
echo "🎯 Running final verification..."

# Test LangGraph import
if python3 -c "import langgraph; print('LangGraph version:', langgraph.__version__)" 2>/dev/null; then
    echo "✅ LangGraph is properly installed"
else
    echo "❌ LangGraph import failed"
    exit 1
fi

# Test if langgraph dev can be imported (not run, just imported)
if python3 -c "from agent.graph.graph import create_workflow_graph; graph = create_workflow_graph(); print('Graph created with', len(graph.get_graph().nodes), 'nodes')" 2>/dev/null; then
    echo "✅ Workflow graph can be created successfully"
else
    echo "❌ Workflow graph creation failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. 📝 Configure your .env file with API keys and database URL"
echo "2. 🗄️  Set up your database by running: psql \$DATABASE_URL -f scripts/init_db.sql"
echo "3. 🚀 Start development server: langgraph dev"
echo "4. 🧪 Run tests: pytest"
echo ""
echo "For deployment to Railway:"
echo "- Make sure all environment variables are set in Railway dashboard"
echo "- Database tables will be created automatically on first deployment"
echo ""
echo "Happy coding! 🎯" 