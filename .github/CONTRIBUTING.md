# Contributing to SynthLine AI

Thank you for your interest in contributing to SynthLine AI! We welcome bug fixes, documentation improvements, new defect generators, and workflow enhancements.

## Code of Conduct

Please be respectful, collaborative, and constructive in all discussions and code reviews.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/thrivecodes/synthline-ai.git
   cd synthline-ai
   ```

2. Set up a Python 3.11+ virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Workflow & Coding Standards

- **Code Style & Linting**: We use [Ruff](https://astral.sh/ruff). Run:
  ```bash
  ruff check src tests
  ruff format --check src tests
  ```
- **Type Checking**: We use strict [Mypy](https://mypy-lang.org/). Run:
  ```bash
  mypy src
  ```
- **Testing**: We write unit and integration tests using [pytest](https://pytest.org/). Run:
  ```bash
  pytest -v --cov=synthline_ai
  ```
- **Reproducibility**: If you implement or modify a procedural defect generator, ensure it accepts a `random_seed` and produces deterministic results across invocations.

## Pull Request Guidelines

1. Create a descriptive feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-generator
   ```
2. Keep pull requests focused on a single change or feature.
3. Ensure CI checks pass before requesting review.
4. Fill out the PR template completely with test evidence.
