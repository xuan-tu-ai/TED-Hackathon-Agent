# Tests Directory

## Structure

```
tests/
├── test_*.py        # Test files
└── conftest.py      # Pytest fixtures
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_orchestrator.py::test_name -v
```

## Test Types
- Unit tests for individual components
- Integration tests for API endpoints
- Simulation tests for agent behavior
