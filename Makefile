.PHONY: help install install-dev clean format lint typecheck check pre-release test test-cov test-watch serve build all

# Fivccliche - Production-Ready AI Agent Backend Framework
# Makefile for development and testing commands

# Virtual environment path
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python
UV := uv

# Default target - display help
help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  Fivccliche - AI Agent Backend Framework                      ║"
	@echo "║  Development Commands                                         ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 INSTALLATION COMMANDS:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "🎨 CODE QUALITY COMMANDS:"
	@echo "  make format           Format code using Black"
	@echo "  make lint             Run Ruff linter"
	@echo "  make typecheck        Run MyPy type checker"
	@echo "  make check            Run all code quality checks"
	@echo "  make pre-release      Run check + test before a version bump commit"
	@echo ""
	@echo "🧪 TESTING COMMANDS:"
	@echo "  make test             Run pytest with verbose output"
	@echo "  make test-cov         Run pytest with coverage report"
	@echo "  make test-watch       Run pytest in watch mode"
	@echo ""
	@echo "🚀 DEVELOPMENT COMMANDS:"
	@echo "  make serve            Start development server with auto-reload"
	@echo "  make clean            Remove Python cache files"
	@echo ""
	@echo "🔨 BUILD COMMANDS:"
	@echo "  make build            Build package distribution"
	@echo ""
	@echo "🎯 UTILITY COMMANDS:"
	@echo "  make all              Run full development workflow"
	@echo "  make help             Display this help message"
	@echo ""

# Installation Commands
install:
	@echo "📦 Installing production dependencies..."
	$(UV) pip install -e .
	@echo "✅ Production dependencies installed"

install-dev:
	@echo "📦 Installing development dependencies..."
	$(UV) pip install -e ".[dev]"
	@echo "✅ Development dependencies installed"

# Code Quality Commands
format: install-dev
	@echo "🎨 Formatting code with Black..."
	$(VENV_BIN)/black src/ tests/
	@echo "✅ Code formatted"

lint: install-dev
	@echo "🔍 Running Ruff linter..."
	$(PYTHON) scripts/ruff_wrapper.py src/ tests/ --fix --output-format=pylint
	@echo "✅ Linting complete"

typecheck:
	@echo "📝 Running MyPy type checker..."
	$(VENV_BIN)/mypy src/
	@echo "✅ Type checking complete"

check: format lint typecheck
	@echo "✅ All code quality checks passed"

# Required before bumping __version__ and committing a release
pre-release: check test
	@echo "✅ pre-release checks passed; safe to bump version and commit"

# Testing Commands
test:
	@echo "🧪 Running tests with verbose output..."
	$(VENV_BIN)/pytest -v
	@echo "✅ Tests complete"

test-cov:
	@echo "🧪 Running tests with coverage report..."
	$(VENV_BIN)/pytest -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated"

test-watch:
	@echo "👀 Running tests in watch mode..."
	$(VENV_BIN)/pytest-watch
	@echo "✅ Watch mode stopped"

# Development Commands
serve:
	@echo "🚀 Starting development server..."
	$(VENV_BIN)/uvicorn fivccliche.cli:app --reload --host 0.0.0.0 --port 8000
	@echo "✅ Server stopped"

clean:
	@echo "🧹 Cleaning Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	@echo "✅ Cache files removed"

# Build Commands
build:
	@echo "🔨 Building package distribution..."
	$(PYTHON) -m build
	@echo "✅ Package built successfully"

# Utility Commands
all: install-dev format lint typecheck test
	@echo "✅ Full development workflow complete"

