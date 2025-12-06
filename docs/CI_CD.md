# CI/CD Pipeline Guide

Comprehensive guide to BioLM's continuous integration and continuous deployment workflows. This document explains how automated testing, validation, and quality checks work.

## 📋 Table of Contents

- [Overview](#overview)
- [Workflow Files](#workflow-files)
- [When Jobs Run](#when-jobs-run)
- [Job Details](#job-details)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

BioLM uses **GitHub Actions** for automated CI/CD with 6 specialized workflows:

| Workflow | Purpose | Trigger | Duration |
|----------|---------|---------|----------|
| **ci.yml** | Main CI pipeline | Push/PR | ~5-8 min |
| **poetry-tests.yml** | Multi-Python testing | Push/PR | ~4-6 min |
| **unit-tests-fast.yml** | Quick validation | PR/Code change | ~1-2 min |
| **unit-tests-full.yml** | Comprehensive testing | Push/PR | ~6-10 min |
| **plugin-compat.yml** | Plugin integration | Push/PR | ~3-5 min |
| **lockfile-sync.yml** | Dependency validation | Push/PR | ~30 sec |

**Philosophy:** Multiple specialized jobs catch different types of issues early, keeping the codebase stable.

---

## 📁 Workflow Files

All workflows are in `.github/workflows/`:

```
.github/workflows/
├── ci.yml                 # Main CI pipeline
├── poetry-tests.yml       # Multi-version Python testing
├── unit-tests-fast.yml    # Quick unit test subset
├── unit-tests-full.yml    # Full test suite + nightly
├── plugin-compat.yml      # Plugin system validation
└── lockfile-sync.yml      # Dependency lockfile check
```

---

## ⏰ When Jobs Run

### Automatic Triggers

```yaml
# ci.yml, poetry-tests.yml, unit-tests-full.yml, plugin-compat.yml, lockfile-sync.yml
on:
  push:
    branches: [ main, biolm-2.0 ]
  pull_request:
    branches: [ main, biolm-2.0 ]
```

**What this means:**
- **Push to `main` or `biolm-2.0`** → All workflows run
- **Pull request to these branches** → All workflows run
- Ensures code quality before merging

### Smart Triggers

```yaml
# unit-tests-fast.yml
on:
  pull_request: ...
  push:
    paths:
      - 'biolm/**'  # Only run when core code changes
```

**What this means:**
- Saves CI time by only running when relevant code changes
- Doc-only changes don't trigger expensive tests

### Scheduled Runs

```yaml
# unit-tests-full.yml (nightly-heavy-tests job)
schedule:
  - cron: '0 3 * * 0'  # Weekly, Sunday at 3 AM UTC
```

**What this means:**
- Weekly comprehensive testing with expensive operations
- Catches issues that only appear in full integration scenarios

---

## 🔍 Job Details

### 1. `ci.yml` - Main CI Pipeline

**Purpose:** Primary quality gate ensuring code meets all standards.

#### Job: `test`

**What it does:**
1. **Matrix Testing** - Tests Python 3.10 and 3.11
2. **Installation** - Sets up Poetry with dev dependencies
3. **Linting** - Runs `ruff` for code quality (non-blocking)
4. **Testing** - Runs full test suite (61 tests)
5. **Deprecation Check** - Fails if deprecated config keys found

**Code quality check:**
```yaml
- name: Fail if deprecated config keys present
  run: |
    if grep -R --line-number --exclude-dir .git -e '\blearningrate\b' .; then
      echo "Found deprecated 'learningrate' usage. Please migrate to 'learning_rate'."
      exit 1
    fi
```

**Educational value:**
- Shows how to enforce API migrations across codebase
- `grep -R` searches recursively for deprecated patterns
- Fails CI if old patterns found, forcing cleanup

**Example output (passing):**
```
✓ Python 3.10 tests passed (61 tests)
✓ Python 3.11 tests passed (61 tests)
✓ No deprecated config keys found
```

---

#### Job: `mlflow-smoke`

**What it does:**
1. **Depends on `test`** - Only runs if main tests pass
2. **MLflow Setup** - Installs with `--with mlflow` extras
3. **MLflow Testing** - Runs 4 MLflow-specific tests

**Why separate job:**
- MLflow is optional dependency (not all users need it)
- Tests experiment tracking integration
- Verifies MLflow extras install correctly

**Tests run:**
- `test_mlflow_integration.py` - Run creation
- `test_runner_mlflow.py` - Runner integration
- `test_mlflow_smoke.py` - Basic functionality
- `test_mlflow_model_logging.py` - Model serialization

**Educational value:**
- Demonstrates optional dependency testing strategy
- Shows how to use job dependencies (`needs: test`)
- Tests subset of features requiring extra packages

---

### 2. `poetry-tests.yml` - Multi-Python Testing

**Purpose:** Ensures compatibility across Python versions.

#### Job: `tests`

**What it does:**
1. **Matrix Strategy** - Tests Python 3.9, 3.10, 3.11
2. **Caching** - Speeds up with Poetry/pip caches
3. **Full Suite** - Runs all 61 tests per version

**Caching configuration:**
```yaml
- name: Cache Poetry & pip
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pypoetry
      ~/.cache/pip
    key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock', '**/pyproject.toml') }}
```

**Educational value:**
- Shows proper CI caching strategy
- Cache key includes file hashes (invalidates when deps change)
- Reduces CI time from ~10 min to ~4 min on cache hits

**Why test multiple Python versions:**
- Users may have different Python installations
- Catches version-specific bugs (e.g., typing differences)
- Ensures backward compatibility with 3.9

---

#### Job: `lint`

**What it does:**
1. **Linting** - Runs code quality checks
2. **Fast Fail** - Uses `--maxfail=1` to stop on first error
3. **Clean Output** - Uses `--disable-warnings` for clarity

**Educational value:**
- Separate lint job allows parallel execution
- Fast-fail saves CI time when issues exist
- Always uses latest Python (3.11) for linting

---

### 3. `unit-tests-fast.yml` - Quick Validation

**Purpose:** Fast feedback on pull requests (1-2 min).

**What it does:**
1. **Subset Testing** - Only runs 2 fast test files
2. **Smart Triggers** - Only on `biolm/**` code changes
3. **Single Version** - Python 3.11 only

**Tests run:**
```yaml
poetry run pytest -q tests/test_dataset_utils.py tests/test_cross_validator.py
```

**Why these tests:**
- `test_dataset_utils.py` (3 tests) - Core data splitting
- `test_cross_validator.py` (7 tests) - CV logic
- Total: 10 tests covering critical functionality
- No external dependencies (pure Python logic)

**Educational value:**
- Shows smoke testing strategy for PRs
- Provides quick feedback (< 2 min vs full 5-10 min)
- Path filtering prevents unnecessary runs
- Trade-off: Speed vs coverage (catches ~80% of issues)

**When to use:**
- Rapid iteration during development
- Pre-push local validation
- Quick sanity check before full CI

---

### 4. `unit-tests-full.yml` - Comprehensive Testing

**Purpose:** Complete validation across multiple Python versions.

#### Job: `tests-matrix`

**What it does:**
1. **Matrix Testing** - Python 3.10, 3.11, 3.12
2. **Full Coverage** - All 61 tests
3. **In-Project venv** - Hermetic testing environment

**Configuration:**
```yaml
strategy:
  matrix:
    python-version: [3.10, 3.11, 3.12]
```

**Educational value:**
- Tests bleeding-edge Python (3.12) for future compatibility
- Matrix runs 3 parallel jobs (faster than sequential)
- Each version tests independently

---

#### Job: `nightly-heavy-tests`

**Purpose:** Weekly expensive integration tests.

**What it does:**
1. **Scheduled Run** - Sunday 3 AM UTC
2. **CPU PyTorch** - Installs CPU-only wheels
3. **Heavy Tests** - Long-running integration scenarios

**PyTorch CPU installation:**
```yaml
poetry run pip install --extra-index-url https://download.pytorch.org/whl/cpu torch --upgrade
```

**Educational value:**
- Shows conditional job execution (`if: github.event_name == 'schedule'`)
- Demonstrates PyTorch CPU installation for CI (no GPU needed)
- Uses `--extra-index-url` for alternative wheel sources
- Weekly cadence balances cost vs coverage

**Why CPU wheels:**
- GitHub Actions doesn't provide GPUs
- CPU wheels are smaller (~100 MB vs ~2 GB)
- Faster downloads and installation
- Tests still validate model architecture

**Types of heavy tests:**
- Full training loops (not mocked)
- Large dataset processing
- End-to-end pipelines
- Memory-intensive operations

---

### 5. `plugin-compat.yml` - Plugin Integration

**Purpose:** Validates plugin system with real plugins.

**What it does:**
1. **Multi-Repo Checkout** - Clones framework + 2 plugins
2. **Plugin Installation** - Installs Saluki and XLNet
3. **Integration Testing** - Runs plugin discovery tests
4. **Path Rewriting** - Adapts paths for CI environment

**Multi-repo setup:**
```yaml
- name: Checkout framework
  uses: actions/checkout@v4
  with:
    path: biolm_utils

- name: Checkout Saluki plugin
  uses: actions/checkout@v4
  with:
    repository: dieterich-lab/rna_saluki_cnn
    ref: saluki-2.0
    path: rna_saluki_cnn

- name: Checkout XLNet plugin
  uses: actions/checkout@v4
  with:
    repository: dieterich-lab/rna_protein_xlnet
    ref: main
    path: rna_protein_xlnet
```

**Educational value:**
- Shows how to test multi-repository architectures
- Demonstrates plugin installation workflow
- Uses `sed` to rewrite local paths for CI:
  ```bash
  sed -i 's|/home/pwiesenbach/rna_saluki_cnn|../rna_saluki_cnn|g' pyproject.toml
  ```
- Tests actual plugin discovery (not mocked)

**Why this matters:**
- Catches plugin compatibility issues early
- Validates entry point system works
- Ensures framework changes don't break plugins
- Simulates user installation experience

**Tests run:**
- All 10 tests in `test_plugin_discovery.py`
- Entry point registration
- Plugin loading and configuration
- No builtin plugins present

---

### 6. `lockfile-sync.yml` - Dependency Validation

**Purpose:** Ensures `poetry.lock` matches `pyproject.toml`.

**What it does:**
1. **Lockfile Generation** - Runs `poetry lock --no-update`
2. **Diff Check** - Verifies no changes occurred
3. **Fails if Mismatch** - Forces developer to commit updated lock

**Validation logic:**
```yaml
- name: Verify lockfile
  run: |
    poetry lock --no-interaction --no-update
    if [ -n "$(git status --porcelain --untracked-files=no -- poetry.lock)" ]; then
      echo "poetry.lock is out-of-sync with pyproject.toml. Run 'poetry lock' and commit the changed poetry.lock file."
      git --no-pager diff --no-color --name-only -- poetry.lock
      exit 1
    fi
```

**Educational value:**
- Shows how to enforce lockfile discipline
- `--no-update` regenerates lock without changing versions
- `git status --porcelain` checks for unstaged changes
- Prevents "works on my machine" dependency issues

**Why this is critical:**
- Lockfile ensures reproducible builds
- Mismatch means dependencies could change unexpectedly
- Common mistake: update `pyproject.toml`, forget `poetry lock`
- CI catches this before merge

**How to fix if it fails:**
```bash
cd /prj/RNA_NLP/biolm_utils
poetry lock --no-update
git add poetry.lock
git commit -m "Update poetry.lock"
```

---

## 🎓 Best Practices

### 1. Understanding Job Dependencies

```yaml
mlflow-smoke:
  needs: test  # This job waits for 'test' to complete
```

**Benefits:**
- Saves CI resources (doesn't run if tests fail)
- Logical flow (don't test integrations if core is broken)
- Faster feedback (main tests finish first)

### 2. Matrix Strategy for Compatibility

```yaml
strategy:
  matrix:
    python-version: [3.10, 3.11, 3.12]
```

**When to use:**
- Testing multiple Python versions
- Testing multiple OS (linux, macos, windows)
- Testing different configurations

**Runs:** 3 parallel jobs (faster than 3 sequential)

### 3. Caching for Speed

```yaml
uses: actions/cache@v4
with:
  path: ~/.cache/pypoetry
  key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}
```

**Cache invalidation:**
- Key includes `poetry.lock` hash
- Lock file changes → cache miss → fresh install
- Lock file unchanged → cache hit → fast restore

**Speed improvement:**
- Cold run: ~8 min (install from scratch)
- Cached run: ~4 min (restore from cache)

### 4. In-Project Virtualenvs

```yaml
poetry config virtualenvs.in-project true --local
```

**Why:**
- Consistent venv location (`.venv/`)
- Easier to cache
- Hermetic (isolated from system Python)
- Matches local development

### 5. Smart Triggers with Paths

```yaml
on:
  push:
    paths:
      - 'biolm/**'  # Only trigger on core code changes
```

**Saves CI time on:**
- Documentation updates
- README changes
- Config file tweaks
- Example updates

### 6. Scheduled Heavy Tests

```yaml
schedule:
  - cron: '0 3 * * 0'  # Sunday at 3 AM UTC
```

**When to use:**
- Expensive integration tests
- Tests requiring external services
- Long-running validation
- Compatibility checks

**Benefits:**
- Doesn't block PR merges
- Runs when resources are cheaper
- Weekly cadence catches regressions

---

## 🐛 Troubleshooting

### Job Failed: "poetry.lock is out-of-sync"

**Cause:** Modified `pyproject.toml` without updating lock file.

**Fix:**
```bash
poetry lock --no-update
git add poetry.lock
git commit -m "Update poetry.lock"
git push
```

---

### Job Failed: "Deprecated config key found"

**Cause:** Code still uses old API (e.g., `learningrate` instead of `learning_rate`).

**Fix:**
1. Find usage: `grep -R "learningrate" .`
2. Replace with new API: `learning_rate`
3. Commit and push

---

### Job Failed: Plugin compatibility test

**Cause:** Framework changes broke plugin interface.

**Fix:**
1. Check `test_plugin_discovery.py` output
2. Verify plugin entry points are correct
3. Ensure `PluginConfig` interface matches plugins
4. Update plugin repos if needed

**Common issues:**
- Changed `PluginConfig` attributes
- Removed required plugin API
- Entry point name mismatch

---

### Job Failed: MLflow tests

**Cause:** MLflow extras not installed or compatibility issue.

**Fix:**
1. Verify `pyproject.toml` has `mlflow` extras:
   ```toml
   [tool.poetry.extras]
   mlflow = ["mlflow>=2.0.0"]
   ```
2. Update MLflow version if needed
3. Check for MLflow API changes

---

### Job Takes Too Long

**Solutions:**

**1. Add caching:**
```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pypoetry
    key: poetry-${{ hashFiles('poetry.lock') }}
```

**2. Use fast subset tests:**
- Add to `unit-tests-fast.yml`
- Run only critical tests
- Use for quick PR validation

**3. Parallelize with matrix:**
```yaml
strategy:
  matrix:
    test-group: [unit, integration]
```

**4. Move heavy tests to scheduled:**
- Weekly run instead of every push
- Saves CI minutes

---

## 📊 CI Workflow Decision Tree

```
Push/PR to main or biolm-2.0
│
├─ Always runs:
│  ├─ ci.yml (main + mlflow)
│  ├─ poetry-tests.yml (multi-version)
│  ├─ unit-tests-full.yml (comprehensive)
│  ├─ plugin-compat.yml (integration)
│  └─ lockfile-sync.yml (validation)
│
├─ On code changes only:
│  └─ unit-tests-fast.yml (quick validation)
│
└─ Weekly (Sunday 3 AM):
   └─ unit-tests-full.yml (nightly-heavy-tests)
```

---

## 🔧 Local Testing

**Before pushing, run CI checks locally:**

```bash
# 1. Lint check
ruff check .

# 2. Quick tests (like unit-tests-fast)
poetry run pytest tests/test_dataset_utils.py tests/test_cross_validator.py

# 3. Full test suite
poetry run pytest tests/

# 4. Verify lockfile
poetry lock --no-update
git diff poetry.lock  # Should show no changes

# 5. Check for deprecated keys
grep -R "learningrate" . --exclude-dir=.git

# 6. Test specific Python version
poetry env use 3.11
poetry install
poetry run pytest tests/
```

---

## 🎯 Adding New CI Jobs

**Template for new workflow:**

```yaml
name: My New Check

on:
  push:
    branches: [ main, biolm-2.0 ]
  pull_request:
    branches: [ main, biolm-2.0 ]

jobs:
  my-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      
      - name: Install dependencies
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          poetry install --no-interaction --with dev
      
      - name: Run my check
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          poetry run <your-command>
```

**Checklist:**
- [ ] Clear job name
- [ ] Appropriate triggers
- [ ] Correct Python version
- [ ] Necessary dependencies
- [ ] Descriptive step names
- [ ] Proper error handling

---

## 📈 CI Metrics

**Current CI Statistics:**

| Metric | Value |
|--------|-------|
| **Total Workflows** | 6 |
| **Jobs per Push** | ~12 (parallel) |
| **Total CI Time** | ~8-10 min (parallel) |
| **Sequential Time** | ~35-40 min (if serial) |
| **Tests per Run** | 61 (framework) |
| **Python Versions** | 3.9, 3.10, 3.11, 3.12 |
| **Cache Hit Rate** | ~80% |

**Time Savings from Parallelization:**
- Sequential: ~40 min
- Parallel: ~10 min
- **Speedup: 4x**

---

## 🔗 Related Documentation

- **[Testing Guide](TESTING.md)** - Detailed test documentation
- **[Installation Guide](INSTALLATION.md)** - Setup instructions
- **[Plugin Development](PLUGIN_DEVELOPMENT.md)** - Plugin architecture

---

## 🚀 Future Improvements

### Potential Additions

1. **Coverage Reports**
   ```yaml
   - name: Upload coverage
     uses: codecov/codecov-action@v3
   ```

2. **Performance Benchmarks**
   ```yaml
   - name: Run benchmarks
     run: poetry run pytest --benchmark-only
   ```

3. **Security Scanning**
   ```yaml
   - name: Safety check
     run: poetry run safety check
   ```

4. **Documentation Build**
   ```yaml
   - name: Build docs
     run: mkdocs build
   ```

5. **Release Automation**
   ```yaml
   - name: Publish to PyPI
     if: github.event_name == 'release'
     run: poetry publish --build
   ```

---

**Last Updated:** December 6, 2025  
**CI/CD Version:** 1.0.0 (BioLM 2.0)

