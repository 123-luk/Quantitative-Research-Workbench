# Dependency Policy

## 1. Purpose

`requirements.txt` declares the packages that this project uses directly.
`constraints-v3-core.txt` records the exact core package versions in the
environment that has passed the V3 test suite. The constraints file improves
reproducibility but does not replace the direct dependency declaration.

## 2. Validated Environment

The current validation environment is Windows with Python 3.12.2.

| Package | Validated version |
| --- | --- |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scipy | 1.17.1 |
| statsmodels | 0.14.6 |
| scikit-learn | 1.9.0 |
| PyYAML | 6.0.3 |
| pyarrow | 24.0.0 |
| pytest | 9.0.3 |

## 3. Installation

The following PowerShell command is an example for the current development
machine. The `E:` path is not required for other users:

```powershell
& "E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe" -m pip install -r requirements.txt -c constraints-v3-core.txt
```

The portable form is:

```text
python -m pip install -r requirements.txt -c constraints-v3-core.txt
```

## 4. Core and Optional Dependencies

The core runtime dependencies covered by the validated constraints are:

- numpy
- pandas
- scipy
- statsmodels
- scikit-learn
- PyYAML
- pyarrow

`pytest` is a development and test dependency.

LightGBM and XGBoost are optional model dependencies that have not yet been
installed, declared, or validated. V3-D2 will validate installation,
Python 3.12 compatibility, and tests before recording them in a separate
optional dependency file. Unverified versions must not be added to the core
requirements.

## 5. joblib Policy

joblib is currently installed as a transitive dependency of scikit-learn.
V3-C does not save or load models and production code does not directly import
joblib. It therefore remains a transitive dependency rather than a project
direct dependency. If V3-H production code directly imports joblib, joblib must
then be added to the direct dependency declaration and the appropriate
validated constraints.

## 6. Version Policy

`requirements.txt` expresses direct project dependencies, while
`constraints-v3-core.txt` preserves the exact core versions that were tested.
Any version upgrade requires a new full test run. Version files must not be
updated without validation, and a user's complete `pip freeze` output must not
be committed as the project dependency definition.

## 7. Optional Model Dependency Policy

Future LightGBM and XGBoost support must:

- use a separate optional requirements file;
- use lazy imports;
- leave Ridge, ElasticNet, and HistGradientBoosting usable when optional
  packages are absent;
- raise a clear error only when the corresponding unavailable adapter is
  created;
- record a version only after installation in the real target environment and
  successful compatibility and test validation.

No LightGBM or XGBoost version is guessed or supported by the current stage.

## 8. Reproducibility Limitations

The exact Python version is currently documented rather than enforced through
a `pyproject.toml` `requires-python` declaration. These constraints primarily
describe the validated Windows and Python 3.12 environment; other platforms
require separate validation. The constraints file covers selected core
packages and is not a complete lock file.
