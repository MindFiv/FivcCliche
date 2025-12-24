# Fivccliche

A **production-ready, multi-user backend framework** designed specifically for **AI agents**. Built with **FastAPI** and **SQLModel** for high-performance, type-safe async operations that handle concurrent AI agent requests at scale.

## ✨ Features

- **AI Agent Backend** - Purpose-built for multi-user AI agent interactions and orchestration
- **FastAPI** - Modern, fast web framework for building high-performance APIs with Python 3.10+
- **SQLModel** - SQL ORM combining SQLAlchemy and Pydantic for type-safe database operations
- **Async/Await** - Full async support for handling concurrent AI agent requests at scale
- **Type Safety** - Built-in type hints with Pydantic 2.0 validation for reliable data handling
- **Multi-User Support** - Designed for managing multiple AI agents with proper isolation and access control
- **Testing** - Pytest with async support for comprehensive test coverage
- **Code Quality** - Black, Ruff, and MyPy configured for professional code standards
- **Package Management** - `uv` for fast, reliable dependency management

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- `uv` package manager ([install](https://docs.astral.sh/uv/))

### Installation

```bash
# Clone the repository
git clone https://github.com/MindFiv/FivcCliche.git
cd FivcCliche

# Install production dependencies
uv pip install -e .

# Or install with development tools
uv pip install -e ".[dev]"
```

### Using the CLI

The easiest way to run FivcCliche is using the built-in CLI:

```bash
# Start the server
python -m fivccliche.cli run

# Show project information
python -m fivccliche.cli info

# Clean temporary files and cache
python -m fivccliche.cli clean

# Initialize configuration
python -m fivccliche.cli setup
```

Visit http://localhost:8000/docs for interactive API documentation.

### CLI Options

```bash
# Custom host and port
python -m fivccliche.cli run --host 127.0.0.1 --port 9000

# Production mode (no auto-reload)
python -m fivccliche.cli run --no-reload

# Test configuration without running
python -m fivccliche.cli run --dry-run

# Verbose output
python -m fivccliche.cli run --verbose
```

## 📚 Documentation

For detailed information, see the documentation in the `docs/` folder:

- **[Getting Started](docs/getting-started.md)** - Comprehensive tutorial with examples
- **[Setup Summary](docs/setup-summary.md)** - Installation and project structure
- **[Migration Plan](docs/migration-plan.md)** - Technical migration details
- **[Completion Summary](docs/completion-summary.md)** - What was accomplished

## 🛠️ Development

### CLI Commands
```bash
make format  # Format code with Black
make lint    # Lint with Ruff
make check   # Run all checks (format, lint, type check)
```

### Run Tests
```bash
pytest
pytest -v --cov=src  # With coverage
```

### Code Quality
```bash
black src/ tests/      # Format code
ruff check src/ tests/ # Lint code
mypy src/              # Type check
```

### Project Structure
```
fivccliche/
├── pyproject.toml              # Project configuration
├── src/
│   └── fivccliche/
│       ├── __init__.py
│       ├── cli.py              # CLI implementation
│       ├── services/
│       ├── utils/
│       ├── settings/
│       └── modules/
├── tests/                      # Add your tests here
└── docs/                       # Documentation
```

## 📦 Dependencies

**Production Core**: FastAPI, SQLModel, Uvicorn, Pydantic, SQLAlchemy

**CLI & Output**: Typer, Rich, python-dotenv

**Component System**: fivcglue, fivcplayground

**Development**: Pytest, Black, Ruff, MyPy, Coverage

See `pyproject.toml` for complete dependency list and versions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Charlie Zhang (sunnypig2002@gmail.com)

## 🔗 Links

- **Repository**: https://github.com/MindFiv/FivcCliche
- **FastAPI**: https://fastapi.tiangolo.com/
- **SQLModel**: https://sqlmodel.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/

