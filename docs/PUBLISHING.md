# Publishing Guide

Guide for publishing BioLM framework and plugins to PyPI (when ready).

## Prerequisites

- PyPI account (https://pypi.org/account/register/)
- TestPyPI account for testing (https://test.pypi.org/)
- Poetry installed
- Repository maintainer access

## Publishing Workflow

### 1. Version Update

Update version in `pyproject.toml`:
```toml
[project]
version = "0.1.0"  # Semantic versioning
```

### 2. Update Changelog

Document changes in `CHANGELOG.md`

### 3. Build Package

```bash
poetry build
```

Creates:
- `dist/biolm-0.1.0.tar.gz`
- `dist/biolm-0.1.0-py3-none-any.whl`

### 4. Test on TestPyPI

```bash
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry publish -r testpypi
```

Install and test:
```bash
pip install --index-url https://test.pypi.org/simple/ biolm
```

### 5. Publish to PyPI

```bash
poetry publish
```

### 6. Tag Release

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Semantic Versioning

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes

## Plugin Dependencies

### Development
```toml
biolm = {path = "../biolm_utils", develop = true}
```

### Production
```toml
biolm = "^0.1.0"  # Compatible with 0.1.x
```

## Automation with GitHub Actions

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      - name: Build
        run: poetry build
      - name: Publish
        env:
          POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
        run: poetry publish
```

## Version Compatibility Matrix

| Framework | Saluki | XLNet | Notes |
|-----------|--------|-------|-------|
| 0.0.3     | 0.1.0  | 0.1.0 | Current development |
| 0.1.0     | TBD    | TBD   | First stable release |

## Checklist

- [ ] Version updated in pyproject.toml
- [ ] CHANGELOG.md updated
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Built and tested on TestPyPI
- [ ] Published to PyPI
- [ ] Git tag created
- [ ] GitHub release created
- [ ] Announcement posted

## Notes

- Framework and plugins can be versioned independently
- Plugins should specify compatible framework versions
- Use `^version` for semantic versioning compatibility
