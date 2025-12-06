# Testing Guide

Complete documentation of BioLM's test suite. All 67 tests are organized by category and explained in detail.

## 📊 Test Statistics

- **Total Tests:** 61
- **Test Files:** 16
- **Coverage:** Core framework and plugin system
- **Status:** ✅ All passing

> **Note:** Plugin-specific end-to-end tests live in their respective plugin repositories (XLNet, Saluki, etc.).

## 🏗️ Test Structure

```
tests/
├── integration/             # Plugin integration tests (10 tests)
│   └── test_plugin_discovery.py
└── unit/                    # Unit tests (51 tests)
    ├── test_biolm.py
    ├── test_cross_validator.py
    ├── test_dataset_utils.py
    ├── test_gpu_*.py
    ├── test_integration.py
    ├── test_loader.py
    ├── test_mlflow_*.py
    ├── test_params.py
    ├── test_runner.py
    ├── test_training_loop.py
    └── test_train_utils.py
```

## Running Tests

```bash
# Run all tests
poetry run pytest tests/

# Run specific category
poetry run pytest tests/integration/       # Integration tests
poetry run pytest tests/test_*.py          # Unit tests

# Run specific file
poetry run pytest tests/test_biolm.py

# Run with coverage
poetry run pytest tests/ --cov=biolm --cov-report=html

# Run with verbose output
poetry run pytest tests/ -v

# Run specific test
poetry run pytest tests/test_biolm.py::TestBiolm::test_set_seed
```

---

## 🔌 Integration Tests (10 tests)

These tests verify plugin discovery, registration, and compatibility.

### `tests/integration/test_plugin_discovery.py` (10 tests)

#### `test_plugin_entry_points_exist`
**Purpose:** Verifies at least one plugin is registered.

**What it tests:**
- `importlib.metadata.entry_points(group='biolm.plugins')` returns results
- Entry point group exists and is not empty

**Why it matters:** Confirms plugin system is functional

---

#### `test_saluki_plugin_registered`
**Purpose:** Confirms Saluki plugin is registered in entry points.

**What it tests:**
- Saluki appears in entry point list
- Entry point name is exactly `"saluki"`

---

#### `test_xlnet_plugin_registered`
**Purpose:** Confirms XLNet plugin is registered in entry points.

**What it tests:**
- XLNet appears in entry point list
- Entry point name is exactly `"xlnet"`

---

#### `test_plugin_loading_saluki`
**Purpose:** Tests Saluki plugin can be loaded and executed.

**What it tests:**
1. Entry point can be loaded via `.load()`
2. Config function is callable
3. Executing config function returns `PluginConfig` instance
4. Config has required attributes (`model_cls_for_finetuning`, `dataset_cls`, etc.)

**How it works:**
- Finds saluki entry point
- Calls `ep.load()()` to get config
- Validates config object structure

---

#### `test_plugin_loading_xlnet`
**Purpose:** Tests XLNet plugin can be loaded and executed.

**What it tests:**
1. Entry point can be loaded
2. Config function is callable
3. Returns valid `PluginConfig`
4. Has all required attributes

---

#### `test_no_builtin_plugins`
**Purpose:** Ensures framework doesn't contain plugin code.

**What it tests:**
- No `biolm.plugins.saluki` module exists
- No `biolm.plugins.xlnet` module exists
- Framework is plugin-free

**Why it matters:** Enforces clean plugin architecture (plugins in separate repos)

---

#### `test_plugin_discovery_without_framework_import`
**Purpose:** Verifies plugins can be discovered without loading heavy framework modules.

**What it tests:**
- Entry points can be listed without importing `biolm` core
- Plugin discovery is lightweight

**Why it matters:** Enables fast plugin inspection tools

---

#### `test_plugin_config_attributes` (parametrized: saluki, xlnet)
**Purpose:** Validates plugin configs have all required attributes.

**What it tests:** For each plugin:
- `model_cls_for_pretraining` (exists, type correct)
- `model_cls_for_finetuning` (exists, not None)
- `dataset_cls` (exists, not None)
- `tokenizer_cls` (exists, not None)
- `datacollator_cls_for_pretraining` (exists, type correct)
- `datacollator_cls_for_finetuning` (exists, not None)
- `add_special_tokens` (bool)
- `pretraining_required` (bool)

**Why it matters:** Ensures consistent plugin interface

---

#### `test_plugin_models_are_callable`
**Purpose:** Verifies plugin model classes can be instantiated.

**What it tests:**
- `model_cls_for_finetuning` is a class (has `__init__`)
- Model classes are callable (can be instantiated)

**Why it matters:** Catches plugin implementation errors early

---

## 🧪 Unit Tests (51 tests)

### `tests/test_biolm.py` (1 test)

#### `TestBiolm::test_set_seed`
**Purpose:** Tests random seed setting functionality.

**What it tests:**
- `set_seed()` can be called without errors
- Seed value is applied to random number generators

**Why it matters:** Ensures reproducible experiments

---

### `tests/test_cross_validator.py` (7 tests)

#### `test_crossval_random`
**Purpose:** Tests cross-validation with random splits.

**What it tests:**
- `CrossValidator` can create random train/val/test splits
- Split ratios are respected (e.g., [70, 15, 15])
- All data points are assigned to exactly one split

---

#### `test_crossval_predict`
**Purpose:** Tests cross-validation in prediction mode.

**What it tests:**
- Prediction mode uses correct data splits
- No training is performed (only inference)

---

#### `test_compat_parametrized_decorator`
**Purpose:** Tests legacy `parametrized_decorator` compatibility.

**What it tests:**
- Old decorator still works (deprecated but functional)
- Warning is raised about deprecation

---

#### `test_cv_true_without_splitpos_raises`
**Purpose:** Tests validation error when config is invalid.

**What it tests:**
- `crossvalidation=True` without `splitpos` raises `ValueError`
- Error message is informative

---

#### `test_cv_int_without_splitratio_raises`
**Purpose:** Tests validation for integer cross-validation.

**What it tests:**
- `crossvalidation=<int>` without `splitratio` raises error
- Prevents undefined split behavior

---

#### `test_cv_int_with_splitpos_conflict_raises`
**Purpose:** Tests conflicting cross-validation config detection.

**What it tests:**
- `crossvalidation=<int>` with `splitpos` raises error
- Can't mix pre-defined splits with k-fold

---

#### `test_splitpos_without_devsplits_raises_no_cv`
**Purpose:** Tests validation when splitpos is used incorrectly.

**What it tests:**
- `splitpos` without `devsplits` raises error (non-CV mode)
- Ensures complete split configuration

---

### `tests/test_dataset_utils.py` (3 tests)

#### `test_split_indices_two_way`
**Purpose:** Tests train/test split generation.

**What it tests:**
- `split_indices()` with 2-way split works correctly
- Indices don't overlap
- All indices are covered

---

#### `test_split_indices_three_way`
**Purpose:** Tests train/val/test split generation.

**What it tests:**
- `split_indices()` with 3-way split works correctly
- Split ratios are respected
- No data leakage between splits

---

#### `test_make_subsets_and_check_batchsize`
**Purpose:** Tests dataset subset creation and batch size validation.

**What it tests:**
- Subsets can be created from dataset
- Batch size is checked against dataset size
- Warning raised if batch too large

---

### `tests/test_gpu_autodetect.py` (6 tests)

#### `TestGPUAutodetect::test_no_torch_fallbacks_to_cpu`
**Purpose:** Tests CPU fallback when PyTorch unavailable.

**What it tests:**
- System falls back to CPU if torch not available
- No crash when GPU detection fails

---

#### `TestGPUAutodetect::test_non_power_of_two_reduced`
**Purpose:** Tests GPU count reduction for invalid values.

**What it tests:**
- Non-power-of-2 GPU counts are reduced (e.g., 3 → 2)
- Prevents distributed training issues

---

#### `TestGPUAutodetect::test_power_of_two_respected`
**Purpose:** Tests valid GPU counts are preserved.

**What it tests:**
- Power-of-2 values (1, 2, 4, 8) are kept as-is
- No unnecessary reduction

---

#### `TestGPUAutodetect::test_explicit_invalid_raises`
**Purpose:** Tests validation of explicit GPU specifications.

**What it tests:**
- Invalid GPU values raise errors
- Prevents impossible configurations

---

#### `TestGPUAutodetect::test_explicit_settings_invalid_raises`
**Purpose:** Tests validation through settings config.

**What it tests:**
- Invalid GPU settings are caught early
- Error messages guide user to fix

---

#### `TestGPUAutodetect::test_explicit_valid_is_removed`
**Purpose:** Tests that valid explicit GPU config is normalized.

**What it tests:**
- Valid GPU specifications are processed correctly
- Redundant configs are cleaned up

---

### `tests/test_gpu_detection.py` (1 test)

#### `test_gpu_detection`
**Purpose:** Tests basic GPU detection functionality.

**What it tests:**
- System can detect available GPUs
- Returns correct count (or 0 if none)

---

### `tests/test_integration.py` (5 tests)

#### `TestIntegration::test_tokenize_mode_integration`
**Purpose:** Tests tokenization mode with mocked components.

**What it tests:**
- Tokenize mode calls correct functions
- Tokenizer is created and saved
- Data flow is correct

---

#### `TestIntegration::test_fine_tune_mode_integration`
**Purpose:** Tests fine-tuning mode integration.

**What it tests:**
- Fine-tune mode calls training functions
- Model is trained and evaluated
- Results are saved

---

#### `TestIntegration::test_predict_mode_integration`
**Purpose:** Tests prediction mode integration.

**What it tests:**
- Predict mode loads model correctly
- Predictions are generated
- Results are saved

---

#### `TestIntegration::test_interpret_mode_integration`
**Purpose:** Tests interpretation mode integration.

**What it tests:**
- Interpret mode computes feature importance
- LOO (leave-one-out) scores are calculated
- Results are saved

---

#### `TestIntegration::test_set_seed_integration`
**Purpose:** Tests seed setting in integration context.

**What it tests:**
- Seed can be set during integrated workflows
- Random operations are reproducible

---

### `tests/test_loader.py` (3 tests)

#### `test_process_hydra_config_from_dictconfig`
**Purpose:** Tests Hydra config processing from DictConfig.

**What it tests:**
- DictConfig can be converted to structured config
- All fields are preserved
- Type conversions work correctly

---

#### `test_load_config_overrides_accepts_list`
**Purpose:** Tests config override with list values.

**What it tests:**
- List overrides work (e.g., `splitratio=[70,15,15]`)
- Lists are parsed correctly from command line

---

#### `test_load_config_rejects_legacy_ngpus_override`
**Purpose:** Tests that legacy `ngpus` parameter is rejected.

**What it tests:**
- `ngpus` parameter raises deprecation error
- Users are guided to use new parameter name

---

### `tests/test_mlflow_integration.py` (2 tests)

#### `test_start_mlflow_run_enabled`
**Purpose:** Tests MLflow run creation when enabled.

**What it tests:**
- MLflow run is started when `settings.mlflow.enabled=True`
- Run ID is returned
- Artifacts can be logged

---

#### `test_start_mlflow_run_disabled`
**Purpose:** Tests MLflow is skipped when disabled.

**What it tests:**
- No MLflow run created when `enabled=False`
- No errors raised
- Training proceeds normally

---

### `tests/test_mlflow_model_logging.py` (1 test)

#### `test_model_log_and_reload`
**Purpose:** Tests MLflow model logging and reloading.

**What it tests:**
- Model can be logged to MLflow
- Model can be reloaded from MLflow
- Reloaded model makes same predictions

---

### `tests/test_mlflow_smoke.py` (1 test)

#### `test_mlflow_smoke`
**Purpose:** Basic MLflow functionality smoke test.

**What it tests:**
- MLflow can be imported
- Basic operations work
- No configuration errors

---

### `tests/test_params.py` (10 tests)

#### `TestParams::test_validate_config_tokenize_mode_no_validation`
**Purpose:** Tests that tokenize mode doesn't require task parameter.

**What it tests:**
- `mode=tokenize` without `task` is valid
- No validation error raised

---

#### `TestParams::test_validate_config_fine_tune_requires_task`
**Purpose:** Tests fine-tune mode requires task.

**What it tests:**
- `mode=fine-tune` without `task` raises `ValueError`
- Error message mentions required parameter

---

#### `TestParams::test_validate_config_predict_requires_task`
**Purpose:** Tests predict mode requires task.

**What it tests:**
- `mode=predict` without `task` raises error

---

#### `TestParams::test_validate_config_interpret_requires_task`
**Purpose:** Tests interpret mode requires task.

**What it tests:**
- `mode=interpret` without `task` raises error

---

#### `TestParams::test_validate_config_fine_tune_with_task_valid`
**Purpose:** Tests valid fine-tune configuration.

**What it tests:**
- `mode=fine-tune` with `task=regression` passes validation

---

#### `TestParams::test_validate_config_splitratio_invalid_length`
**Purpose:** Tests validation of splitratio length.

**What it tests:**
- `splitratio` with wrong length raises error
- Must be [train%] or [train%, val%, test%]

---

#### `TestParams::test_validate_config_splitratio_not_sum_100`
**Purpose:** Tests splitratio sum validation.

**What it tests:**
- `splitratio` that doesn't sum to 100 raises error
- e.g., `[50, 30, 10]` is invalid

---

#### `TestParams::test_validate_config_splitratio_valid`
**Purpose:** Tests valid splitratio configurations.

**What it tests:**
- `[70, 15, 15]` passes validation
- `[80, 20]` passes validation

---

#### `TestParams::test_validate_config_splitpos_without_devsplits`
**Purpose:** Tests splitpos requires devsplits.

**What it tests:**
- `splitpos` without `devsplits` raises error

---

#### `TestParams::test_validate_config_splitpos_valid`
**Purpose:** Tests valid splitpos configuration.

**What it tests:**
- `splitpos` with `devsplits` passes validation

---

### `tests/test_runner.py` (5 tests)

#### `test_make_run_fn_invalid_mode_raises`
**Purpose:** Tests invalid mode detection.

**What it tests:**
- Unknown mode raises `ValueError`
- Error lists valid modes

---

#### `test_make_run_fn_tokenize_mode`
**Purpose:** Tests tokenize mode function creation.

**What it tests:**
- Tokenize mode creates correct run function
- Function calls tokenization code

---

#### `test_predict_delegates_to_biolm_test`
**Purpose:** Tests predict mode delegates correctly.

**What it tests:**
- Predict mode calls `biolm.test()` function
- Correct parameters are passed

---

#### `test_fine_tune_triggers_train_and_then_test`
**Purpose:** Tests fine-tune calls train then test.

**What it tests:**
- Fine-tune mode calls `biolm.train()`
- Then calls `biolm.test()` for evaluation
- Correct order is maintained

---

#### `test_interpret_delegates_to_loo_scores`
**Purpose:** Tests interpret mode delegates correctly.

**What it tests:**
- Interpret mode calls LOO (leave-one-out) functions
- Feature importance is computed

---

### `tests/test_runner_mlflow.py` (1 test)

#### `test_runner_uses_mlflow_for_results`
**Purpose:** Tests runner integrates with MLflow.

**What it tests:**
- Results are logged to MLflow when enabled
- Metrics are tracked correctly

---

### `tests/test_train_utils.py` (4 tests)

#### `TestTrainUtils::test_log_scaler`
**Purpose:** Tests log scaling for regression targets.

**What it tests:**
- Log scaler transforms values correctly
- Inverse transform recovers original values

---

#### `TestTrainUtils::test_identity_scaler`
**Purpose:** Tests identity scaler (no transformation).

**What it tests:**
- Identity scaler passes through values unchanged
- Useful for debugging

---

#### `TestTrainUtils::test_compute_metrics_for_regression`
**Purpose:** Tests regression metrics computation.

**What it tests:**
- Computes MSE, Pearson, Spearman correlations
- Metrics are correct for known values

---

#### `TestTrainUtils::test_compute_metrics_for_classification`
**Purpose:** Tests classification metrics computation.

**What it tests:**
- Computes accuracy, precision, recall, F1
- Metrics are correct for known predictions

---

### `tests/test_training_loop.py` (1 test)

#### `test_minimal_training_loop`
**Purpose:** Tests basic training loop functionality.

**What it tests:**
- Training loop can run 1 epoch
- Model parameters are updated
- Trainer is created correctly
- Forward pass works after training

---

## 🔧 Test Utilities

### Fixtures

**`tmp_path` (pytest builtin):**
- Provides temporary directory for each test
- Automatically cleaned up after test
- Used for file operations

### Mocking

Many tests use `monkeypatch` to mock heavy operations:
- Training is mocked to avoid GPU usage
- File I/O is mocked for speed
- External services (MLflow) are mocked for isolation

---

## 📈 Coverage

To generate coverage report:

```bash
poetry run pytest tests/ --cov=biolm --cov-report=html
open htmlcov/index.html  # View in browser
```

**Target Coverage:** >80% for core modules

---

## 🐛 Debugging Failed Tests

### Run with verbose output:
```bash
poetry run pytest tests/test_name.py -v -s
```

### Run with full tracebacks:
```bash
poetry run pytest tests/test_name.py --tb=long
```

### Run with PDB on failure:
```bash
poetry run pytest tests/test_name.py --pdb
```

### Run only failed tests from last run:
```bash
poetry run pytest --lf
```

---

## ✅ Test Checklist for New Features

When adding new functionality, ensure you add tests for:

- [ ] Happy path (normal operation)
- [ ] Error cases (invalid inputs)
- [ ] Edge cases (empty data, extreme values)
- [ ] Integration with existing features
- [ ] Plugin compatibility (if applicable)

---

## 🚀 Continuous Integration

Tests run automatically on:
- Every commit to `biolm-2.0` branch
- Every pull request
- Before releases

**CI Requirements:**
- All 67 tests must pass
- No new warnings
- Coverage must not decrease

---

## 📝 Writing New Tests

### Test Naming Convention
```python
def test_<component>_<behavior>_<expected_result>():
    """Brief description of what this tests."""
    pass
```

### Test Structure (Arrange-Act-Assert)
```python
def test_example():
    # Arrange: Set up test data
    data = create_test_data()
    
    # Act: Execute the code being tested
    result = function_under_test(data)
    
    # Assert: Verify the result
    assert result == expected_value
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubling(input, expected):
    assert double(input) == expected
```

---

## 🎯 Test Categories Summary

| Category | Count | Focus | Duration |
|----------|-------|-------|----------|
| **Integration** | 10 | Plugin system | 1-2 min |
| **Unit** | 51 | Individual functions | 10-30 sec |
| **Total** | **61** | Framework coverage | **~5 min** |

> **Note:** Plugin-specific end-to-end tests are maintained in their respective plugin repositories and test complete workflows (tokenization → training → evaluation).

---

## 🔗 Related Documentation

- **[Configuration Reference](CONFIGURATION.md)** - Parameter details tested
- **[Plugin Development](PLUGIN_DEVELOPMENT.md)** - Plugin interface tested
- **[Installation Guide](INSTALLATION.md)** - Setup for running tests

---

**Last Updated:** December 6, 2025  
**Test Suite Version:** 1.0.0 (BioLM 2.0)

