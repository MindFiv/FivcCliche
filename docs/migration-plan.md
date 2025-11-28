# Migration Plan: Converting to `uv` with `pyproject.toml` - FastAPI Stack

## Project Analysis

### Current State
- **Project Name**: fivccliche
- **Package Location**: `src/fivccliche/` (src-layout)
- **Previous Stack**: Django 3.2.3 + DRF (DEPRECATED)
- **New Stack**: FastAPI + SQLModel (MODERN ASYNC)
- **Build System**: Now using `pyproject.toml` with setuptools
- **License**: MIT (2023 Charlie Zhang)

## New Stack Architecture

### Core Dependencies (FastAPI + SQLModel)
- **fastapi** (>=0.104.0) - Modern async web framework
- **sqlmodel** (>=0.0.14) - SQL ORM combining SQLAlchemy + Pydantic
- **uvicorn[standard]** (>=0.24.0) - ASGI server
- **pydantic** (>=2.0.0) - Data validation
- **pydantic-settings** (>=2.0.0) - Configuration management
- **python-multipart** (>=0.0.6) - Form data parsing
- **python-dotenv** (>=1.0.0) - Environment variable loading
- **typer** (>=0.9.0) - CLI framework
- **rich** (>=13.0.0) - Beautiful console output
- **fivcglue** (>=0.1.3) - Dependency injection & component system
- **fivcplayground** (>=0.1.1) - Component utilities

### Development Dependencies
- **pytest** (>=7.4.0) - Testing framework
- **pytest-asyncio** (>=0.21.0) - Async test support
- **httpx** (>=0.25.0) - Async HTTP client for testing
- **black** (>=23.0.0) - Code formatter
- **ruff** (>=0.1.0) - Fast Python linter
- **mypy** (>=1.5.0) - Static type checker
- **pytest-cov** (>=4.1.0) - Coverage reporting

## Configuration Details

### Python Version
- **Minimum**: Python 3.10
- **Recommended**: Python 3.11 or 3.12
- **Rationale**: FastAPI and SQLModel leverage modern async/await features

### Build System
- **Backend**: setuptools (PEP 517/518 compliant)
- **Layout**: src-layout (`src/fivccliche/`)
- **Package Discovery**: Automatic via setuptools

### Tool Configurations
- **Black**: 100 char line length, targets Python 3.10+
- **Ruff**: Fast linting with comprehensive rule set
- **MyPy**: Type checking with flexible strictness
- **Pytest**: Async mode auto-enabled, strict markers

## Files Created/Modified

✅ **Created**: `pyproject.toml` - Complete project configuration
✅ **Created**: `src/fivccliche/__init__.py` - Package initialization
✅ **Created**: `src/fivccliche/cli.py` - Professional CLI with Typer and Rich
✅ **Preserved**: `requirements.txt` and `requirements-dev.txt` - For reference only

## CLI Implementation

✅ **Professional CLI Framework**: Typer-based CLI with Rich console output
✅ **Commands**: run, info, clean, setup
✅ **Features**: Dry-run mode, verbose output, error handling, interactive prompts

## Next Steps

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Install project: `uv pip install -e .`
3. Install dev dependencies: `uv pip install -e ".[dev]"`
4. Start refactoring Django code to FastAPI
5. Create tests in `tests/` directory

