# FinScope AI — Makefile
# Convenience commands for development and deployment

.PHONY: help install train run test lint docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e ".[dev]"

train: ## Train all models
	python -m scripts.train_models

run: ## Start API server (local)
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run test suite
	pytest tests/ -v --tb=short

lint: ## Run linting
	ruff check .
	black --check .

format: ## Format code
	black .
	ruff check --fix .

docker-up: ## Start Docker services
	docker-compose up --build -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View API logs
	docker-compose logs -f api

migrate: ## Run database migrations
	alembic upgrade head

clean: ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
