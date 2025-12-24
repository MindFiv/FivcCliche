# Getting Started with Fivccliche

## Environment Setup

### 1. Ensure uv is Installed
```bash
# Check if uv is installed
which uv

# If not installed, install it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Project with Dependencies
```bash
cd /Users/charlie/Works/FivcCliche

# Install production dependencies
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

## Using the FivcCliche CLI

The easiest way to get started is using the built-in CLI:

```bash
# Start the server
python -m fivccliche.cli run

# Show project information
python -m fivccliche.cli info

# Clean temporary files
python -m fivccliche.cli clean

# Initialize configuration
python -m fivccliche.cli setup
```

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

## Creating Your First FastAPI Application

### 1. Create Main Application File
Create `src/fivccliche/main.py`:

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application starting up...")
    yield
    # Shutdown
    print("Application shutting down...")

app = FastAPI(
    title="Fivccliche API",
    description="A modern async Python framework",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/")
async def root():
    return {"message": "Welcome to Fivccliche"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Run the Application
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the app
python src/fivccliche/main.py

# Or use uvicorn directly
uvicorn src.fivccliche.main:app --reload
```

### 3. Access the API
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## Creating Database Models with SQLModel

### 1. Create Models File
Create `src/fivccliche/models.py`:

```python
from typing import Optional
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(index=True)
    is_active: bool = True

class Item(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: Optional[str] = None
    owner_id: int = Field(foreign_key="user.id")
```

### 2. Create Database Configuration
Create `src/fivccliche/database.py`:

```python
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy.pool import StaticPool

DATABASE_URL = "sqlite:///database.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
```

## Writing Tests

### 1. Create Tests Directory
```bash
mkdir -p tests
touch tests/__init__.py
```

### 2. Create Test File
Create `tests/test_main.py`:

```python
import pytest
from fastapi.testclient import TestClient
from src.fivccliche.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Fivccliche"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### 3. Run Tests
```bash
pytest
pytest -v  # Verbose
pytest --cov=src  # With coverage
```

## Code Quality Tools

### Format Code
```bash
black src/ tests/
```

### Lint Code
```bash
ruff check src/ tests/
ruff check --fix src/ tests/  # Auto-fix issues
```

### Type Check
```bash
mypy src/
```

## Useful Commands

```bash
# List installed packages
uv pip list

# Show package info
uv pip show fastapi

# Freeze requirements
uv pip freeze > requirements-lock.txt

# Update packages
uv pip install --upgrade fastapi

# Remove package
uv pip uninstall fastapi
```

## Next Steps

1. ✅ Environment is set up
2. 📝 Create your first API endpoint
3. 🗄️ Design your database models
4. 🧪 Write tests for your code
5. 🚀 Deploy your application

Happy coding! 🎉

