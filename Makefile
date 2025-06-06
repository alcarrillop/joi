.PHONY: help install dev-install test test-cov lint format clean run run-dev docker-build docker-run docker-stop setup check pre-commit

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install production dependencies
	uv pip install -e .

dev-install: ## Install development dependencies
	uv pip install -e ".[dev]"
	pre-commit install

setup: ## Setup the project for development
	@echo "Setting up the project..."
	chmod +x scripts/setup.sh
	./scripts/setup.sh

# Development commands
run: ## Run the application locally
	uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --host 0.0.0.0 --port 8000 --workers 1

run-dev: ## Run the application in development mode with reload
	uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --host 0.0.0.0 --port 8000 --reload

run-script: ## Run using the start script
	chmod +x scripts/start.sh
	./scripts/start.sh

# Testing
test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	pytest -f

# Code quality
lint: ## Run linting
	ruff check .

lint-fix: ## Run linting with auto-fix
	ruff check . --fix

format: ## Format code
	ruff format .

format-check: ## Check if code is formatted
	ruff format . --check

check: ## Run all checks (lint + format + tests)
	@echo "Running all checks..."
	ruff check .
	ruff format . --check
	pytest

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# Docker commands
docker-build: ## Build Docker image
	docker-compose build

docker-run: ## Run with Docker Compose
	docker-compose up

docker-run-bg: ## Run with Docker Compose in background
	docker-compose up -d

docker-stop: ## Stop Docker containers
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

# Database commands
db-init: ## Initialize database
	python -m scripts.update_database_schema

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

clean-all: clean ## Clean everything including virtual environment
	rm -rf .venv/

# Development shortcuts
dev: dev-install pre-commit ## Setup development environment
	@echo "Development environment ready!"

quick-check: lint test ## Quick check before commit

ci: ## Run CI-like checks
	ruff check .
	ruff format . --check
	pytest --cov=src

# Local development server
serve: run-dev ## Alias for run-dev

# Environment info
info: ## Show environment info
	@echo "Python version: $(shell python --version)"
	@echo "UV version: $(shell uv --version)"
	@echo "Project: $(shell pwd)" 