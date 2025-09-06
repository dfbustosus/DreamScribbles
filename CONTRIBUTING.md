# Contributing to DreamScribbles

Thanks for your interest in contributing! Please follow these guidelines:

## Getting Started

1) Fork and clone the repo
2) Create a virtual environment and install dev deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
```

## Development

- Run lints and tests locally before opening a PR:

```bash
make fmt
make lint
make type
make test
```

- For large changes, open an issue first to discuss.
- Keep PRs focused and well-documented.

## Commit Style

- Use clear, descriptive commits.
- Reference issues when relevant (e.g., `Fixes #12`).

## Security

See `SECURITY.md` for reporting vulnerabilities.
