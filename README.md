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

### Configuration APIs

Authenticated users can manage user-scoped AI agent resources under `/configs/`.
Superusers create global configs (`user_uuid = null`) that regular users can read but not update
or delete.

- `/configs/embeddings/` - embedding provider/model configs
- `/configs/models/` - LLM provider/model configs with `id`, `description`, `provider`,
  `model`, `api_key`, `base_url`, `temperature`, `max_tokens`, optional nullable
  `enable_thinking`, `user_uuid`, `updated_at`, and `updated_user_uuid` fields.
  `enable_thinking = null` leaves provider behavior unchanged, `true` enables
  provider-supported thinking output, and `false` disables it.
- `/configs/agents/` - agent configs that compose models, tools, and skills
- `/configs/tools/` - tool configs, including MCP/function transports
- `/configs/skills/` - reusable skill configs and resources
- `/configs/questions/` - reusable user question configs with `id`, `question`, optional
  `answer`, `is_active`, `user_uuid`, `updated_at`, and `updated_user_uuid` fields; list with
  `?is_active=true` or `?is_active=false` to filter by active state

`python -m fivccliche.cli migrate` creates missing tables but does not alter existing
tables. Existing databases created before `user_llm.enable_thinking` was added need a
manual nullable boolean column on `user_llm` or a table rebuild.

### Chat Message Cards

Chat messages can have persisted UI cards for agent-built experiences such as questions,
images, charts, or external links. `GET /chats/{chat_uuid}/messages/` returns message
history without loading cards. Use `GET /chats/{chat_uuid}/messages/{message_uuid}/cards/`
to list the cards for one message. Each card contains `uuid`, `message_uuid`, and a
`context` object.

Card `context` is arbitrary user/application-defined JSON. FivcCliche stores and returns it
without enforcing a backend card shape. There are no public card write endpoints; card rows
are created through internal application logic or direct service/model usage.

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
