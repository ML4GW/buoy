# Contributing to buoy

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ML4GW/buoy
   cd buoy
   ```

2. Install with development dependencies using [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync --group dev
   ```

3. Install pre-commit hooks:
   ```bash
   uv run prek install
   ```

## Running Tests

```bash
uv run pytest
```

## Linting and Formatting

Run all pre-commit hooks manually:
```bash
uvx prek run --all-files
```

Or run ruff directly:
```bash
uv run ruff check --fix .
uv run ruff format .
```

## Type Checking (optional)

```bash
uv run ty check src/
```

## Building Documentation

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

## Pull Request Process

1. Open a PR against `main` with a clear description of the change
2. Ensure CI passes before requesting review
