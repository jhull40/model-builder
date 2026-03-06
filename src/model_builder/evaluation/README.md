# Evaluation

This package implements binary classification evaluation for `model-builder`. The single public class, `BinaryClassificationEvaluator`, scores a trained model across all data splits and persists a CSV metrics table and a multi-page PDF report.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Usage](#usage)
3. [Threshold strategy](#threshold-strategy)
4. [Output files](#output-files)
   - [metrics.csv](#metricscsv)
   - [report.pdf](#reportpdf)
5. [Metric reference](#metric-reference)
6. [Random baseline](#random-baseline)

---

## Architecture

`BinaryClassificationEvaluator` is initialised with a `PipelineConfig` and the integer `model_id` produced by `Pipeline.run()`. Calling `evaluate()` drives all computation and I/O; the result is the evaluator instance itself (fluent API).

```
BinaryClassificationEvaluator
├── evaluate(model, train, test, val=None)   ← public entry point
├── _write_csv(...)                          ← metrics.csv
└── _write_pdf(...)                          ← report.pdf
    ├── _page_summary_table(...)             ← PDF page 1
    ├── _page_roc_pr_curves(...)             ← PDF page 2
    ├── _page_histograms(...)                ← PDF page 3
    ├── _page_calibration(...)               ← PDF page 4
    └── _page_confusion_matrices(...)        ← PDF page 5
```

Two module-level helpers are also importable for unit testing:

| Helper | Description |
|---|---|
| `_compute_metrics(y_true, y_proba, threshold=None)` | Full metric set for one split |
| `_optimal_threshold_f1(y_true, y_proba)` | Precision-recall curve threshold that maximises F1 |

---

## Usage

```python
from model_builder import BinaryClassificationEvaluator
from model_builder.pipeline import Pipeline
from model_builder.utils.utils import load_config

config = load_config("configs/config.yaml")
pipeline = Pipeline(config)
pipeline.run(df)   # fits scaler + model, sets pipeline.model_id

evaluator = BinaryClassificationEvaluator(config, model_id=pipeline.model_id)
evaluator.evaluate(
    model,
    train=(X_train, y_train),
    test=(X_test, y_test),
    val=(X_val, y_val),   # optional
)
```

`X_*` are preprocessed `pd.DataFrame`s; `y_*` are `pd.Series` of binary integer labels. The model must expose a `predict_proba(X) -> np.ndarray` method returning a 1-D array of positive-class probabilities.

---

## Threshold strategy

The optimal decision threshold is derived **from the training split only**:

1. Build the precision-recall curve for the training predictions.
2. Select the threshold that maximises F1 on that curve.
3. Apply the same threshold to test and validation splits.

This prevents target leakage: the threshold is never tuned on held-out data.

---

## Output files

All artifacts are written to:

```
output/<name>/evaluation/<model_id>/
├── metrics.csv
└── report.pdf
```

### metrics.csv

One row per evaluated group:

| Row | Description |
|---|---|
| `train` | Training split |
| `test` | Test split |
| `val` | Validation split (omitted when `val=None`) |
| `overall` | All splits concatenated |
| `random` | Naive baseline — constant predictor at the training positive rate |

#### Column reference

| Column | Description |
|---|---|
| `split` | Split name (see above) |
| `n_rows` | Number of samples |
| `actual_positive_rate` | Fraction of true positives |
| `sum_predicted_probs` | Sum of predicted probabilities (sanity check vs. n_rows × positive_rate) |
| `log_loss` | Cross-entropy loss |
| `brier_score` | Mean squared error of predicted probabilities |
| `auc_pr` | Area under the precision-recall curve (average precision) |
| `auc_roc` | Area under the ROC curve |
| `optimal_threshold_f1` | F1-optimal threshold derived from the training split |
| `accuracy` | Fraction of correctly classified samples at the optimal threshold |
| `precision` | Positive predictive value at the optimal threshold |
| `recall` | True positive rate at the optimal threshold |
| `f1` | Harmonic mean of precision and recall at the optimal threshold |

All numeric values are rounded to 6 decimal places.

### report.pdf

A 5-page PDF named `report.pdf` in the same directory.

| Page | Title | Contents |
|---|---|---|
| 1 | Binary Classification Summary | Table of N Samples, Log Loss, AUC ROC, AUC PR for each split + random baseline (highlighted) |
| 2 | AUC Curves | Side-by-side ROC curve and Precision-Recall curve, one line per split with AUC annotations |
| 3 | Prediction Distributions by True Class | Probability histograms split by true label (class 0 vs. class 1), one subplot per data split |
| 4 | Calibration | Calibration curves (quantile binning, 10 bins) plotted against the perfect-calibration diagonal |
| 5 | Confusion Matrices | Confusion matrix for each split at the F1-optimal threshold from training |

---

## Metric reference

### Probability metrics (threshold-independent)

| Metric | Ideal | Notes |
|---|---|---|
| `log_loss` | → 0 | Lower is better; penalises confident wrong predictions heavily |
| `brier_score` | → 0 | Lower is better; MSE of probabilities — good overall calibration indicator |
| `auc_roc` | → 1 | 0.5 = random; measures rank-ordering ability across all thresholds |
| `auc_pr` | → 1 | Equal to the random baseline positive rate; more informative on imbalanced data |

### Threshold-dependent metrics

These are computed at the F1-optimal threshold derived from the training split.

| Metric | Ideal | Notes |
|---|---|---|
| `accuracy` | → 1 | Can be misleading on imbalanced classes |
| `precision` | → 1 | Fraction of positive predictions that are correct |
| `recall` | → 1 | Fraction of actual positives that are detected |
| `f1` | → 1 | Harmonic mean of precision and recall |

---

## Random baseline

The `random` row simulates a caller that knows only the training positive rate `r` and always outputs `predict_proba = r`. With a decision threshold of 0.5:

- When `r < 0.5` the baseline predicts all-negative (zero precision, zero recall, F1 = 0).
- When `r ≥ 0.5` the baseline predicts all-positive (recall = 1, precision = positive rate).

Probability metrics (log_loss, brier_score, AUC PR, AUC ROC) reflect the calibration quality of a constant predictor, giving a lower bound for what a useful model must beat.
