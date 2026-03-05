# model-builder

A Python framework for building, evaluating, and analysing machine-learning pipelines. The project provides a configuration-driven workflow that takes a pandas DataFrame through exploratory data analysis (EDA), preprocessing, training, and evaluation — producing structured reports alongside the trained models.

---

## Table of Contents

1. [High-level goal](#high-level-goal)
2. [Setup](#setup)
3. [Configuration](#configuration)
4. [EDA Analyzer](#eda-analyzer)

---

## High-level goal

`model-builder` aims to reduce the boilerplate required to go from raw tabular data to a validated model. Core responsibilities:

- **EDA** — automatic statistical summaries, distribution classification, correlation heatmaps, outlier detection, and per-feature histograms.
- **Preprocessing** — configurable feature transformations and train/test splitting.
- **Training** — pluggable model back-ends (scikit-learn, XGBoost, PyTorch).
- **Evaluation** — metric reports saved alongside predictions.

All outputs are written to a namespaced directory (`output/<run-name>/`) so results from different experimental runs never collide.

---

## Setup

**Requirements:** Python ≥ 3.11

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone and enter the repo
git clone <repo-url>
cd model_builder

# Create a virtual environment and install all dependencies
uv sync

# Install dev extras (linting, testing)
uv sync --group dev
```

### Common commands

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_config.py

# Run the example script (Iris EDA demo)
uv run python main.py

# Lint and enforce code style (ruff)
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Install and run pre-commit hooks on all files
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## Configuration

Pipelines are driven by a YAML config file validated against `PipelineConfig`.

### Minimal config

```yaml
base:
  name: 'my_run'       # Run name — used as the output sub-directory
  output_dir: 'output' # Root output directory (default: "output")
  seed: 42             # Global random seed (default: 524)

data:
  target_column: 'target'  # Column name to treat as the label (optional)
```

### Field reference

#### `base` (required)

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Unique run identifier. Output is written to `output_dir/name/`. |
| `output_dir` | `str` | `"output"` | Root directory for all run outputs. |
| `seed` | `int` | `524` | Random seed for reproducibility. |

#### `data` (optional)

| Field | Type | Default | Description |
|---|---|---|---|
| `target_column` | `str \| null` | `null` | Name of the label column in the DataFrame. When set, it is plotted first in histogram pages. |

### Loading the config in code

```python
from model_builder.utils.utils import load_config

config = load_config("configs/config.yaml")
print(config.base.name)        # "my_run"
print(config.data.target_column)  # "target"
```

---

## EDA Analyzer

`DataAnalyzer` runs a full exploratory analysis on any pandas DataFrame and writes all outputs under `output/<name>/eda/`.

### Usage

```python
import pandas as pd
from model_builder import DataAnalyzer
from model_builder.utils.utils import load_config

config = load_config("configs/config.yaml")
analyzer = DataAnalyzer(config)
analyzer.run(df)
```

### Output files

| File | Description |
|---|---|
| `eda/eda.pdf` | Single PDF containing all tables and charts |
| `eda/describe.csv` | Per-column descriptive statistics + distribution label |
| `eda/correlations.csv` | Pearson correlation matrix (numeric columns only) |
| `eda/outliers.csv` | IQR 1.5× outlier counts and fence values per column |
| `eda/plots/<col>_histogram.png` | Individual histogram PNG for every numeric column |

### Distribution labels

The analyzer automatically classifies each numeric column:

| Label | Condition |
|---|---|
| `binary` | ≤ 2 unique values |
| `ordinal/categorical` | 3–10 unique values |
| `gaussian` | D'Agostino–Pearson normality test p > 0.05 |
| `uniform` | Chi-square uniformity test p > 0.05 |
| `right-skewed` | Skewness > 1.0 |
| `left-skewed` | Skewness < −1.0 |
| `non-gaussian` | None of the above |

### Sample output

Below is a sample `describe.csv` produced from the Iris dataset (`main.py`):

```
column,nulls,count,mean,std,min,25%,50%,75%,max,distribution
sepal length (cm),0,150,5.843,0.828,4.3,5.1,5.8,6.4,7.9,gaussian
sepal width (cm),0,150,3.057,0.436,2.0,2.8,3.0,3.3,4.4,gaussian
petal length (cm),0,150,3.758,1.765,1.0,1.6,4.35,5.1,6.9,non-gaussian
petal width (cm),0,150,1.199,0.762,0.1,0.3,1.3,1.8,2.5,non-gaussian
target,0,150,1.0,0.819,0.0,0.0,1.0,2.0,2.0,ordinal/categorical
```

Sample `outliers.csv`:

```
column,method,n_outliers,pct_outliers,lower_fence,upper_fence
sepal length (cm),IQR (1.5x),0,0.0,3.15,8.35
sepal width (cm),IQR (1.5x),4,2.67,2.05,4.05
petal length (cm),IQR (1.5x),0,0.0,-3.65,10.35
petal width (cm),IQR (1.5x),0,0.0,-1.95,4.05
target,IQR (1.5x),0,0.0,-3.0,5.0
```

Sample `correlations.csv` (truncated):

```
,sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target
sepal length (cm),1.00,-0.12,0.87,0.82,0.78
sepal width (cm),-0.12,1.00,-0.43,-0.37,-0.43
petal length (cm),0.87,-0.43,1.00,0.96,0.95
petal width (cm),0.82,-0.37,0.96,1.00,0.96
target,0.78,-0.43,0.95,0.96,1.00
```

The PDF (`eda.pdf`) consolidates all of the above into a single document:

- **Data Summary** — paginated table of the `describe.csv` statistics.
- **Feature Correlations** — colour-coded heatmap (red = negative, blue = positive).
- **Histograms** — one plot per numeric column, labelled with its distribution type. Priority order: target column → non-normal columns → remaining columns.
- **Outlier Summary** — table with outlier counts and IQR fences; rows with outliers highlighted in red.
- **Outlier Boxplots** — grid of boxplots (6 per page) with outlier points overlaid in red.
