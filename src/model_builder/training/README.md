# Training

This package implements the training layer of `model-builder`. All models share the `Model` abstract base class defined in [`base.py`](base.py) and are stored under [`models/`](models/).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Model selection](#model-selection)
3. [Models](#models)
   - [Logistic Regression (`logr`)](#logistic-regression-logr)
4. [Adding a new model](#adding-a-new-model)

---

## Architecture

Every model must implement the five methods defined in `Model`:

| Method | Signature | Description |
|---|---|---|
| `fit` | `(X: DataFrame, y: Series) → Model` | Train on the provided features and labels |
| `predict` | `(X: DataFrame) → ndarray` | Return class predictions |
| `score` | `(X: DataFrame, y: Series) → float` | Return accuracy (or equivalent metric) |
| `save` | `(path: Path) → None` | Serialise the model to disk with `joblib` |
| `load` | `(path: Path) → Model` *(classmethod)* | Deserialise a previously saved model |

Models are serialised by `joblib` and written to `output/<name>/models/<model_id>/model_<model_id>`.

---

## Model selection

The active model is specified in the YAML config under `model.type`:

```yaml
model:
  type: logr   # identifier string for the desired backend
  logr:        # sub-block name matches the type string
    ...
```

`model.type` is a discriminated literal — only values listed in `ModelConfig` are accepted. Adding a new model requires updating [`config/schema.py`](../config/schema.py) as well as providing the implementation (see [Adding a new model](#adding-a-new-model)).

---

## Models

### Logistic Regression (`logr`)

**File:** [`models/logr.py`](models/logr.py)
**Backed by:** [`sklearn.linear_model.LogisticRegressionCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)

`LogisticRegressionModel` wraps scikit-learn's cross-validated logistic regression, which automatically selects the best regularisation strength `C` from the candidate set during fitting.

#### Config block

```yaml
model:
  type: logr
  logr:
    Cs: 10
    cv: 5
    solver: lbfgs
    max_iter: 200
    l1_ratios: [0.0]
    n_jobs: -1
```

#### Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Cs` | `int \| list[float]` | `10` | Candidate regularisation strengths. When an integer *n* is given, *n* values are sampled log-uniformly between `1e-4` and `1e4`. Pass an explicit list (e.g. `[0.01, 0.1, 1.0]`) for manual control. Larger values of `C` mean less regularisation. |
| `cv` | `int` | `5` | Number of cross-validation folds used to select the best `C`. |
| `solver` | `str` | `"lbfgs"` | Optimisation algorithm. Choices: `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga`. See the solver/penalty compatibility table below. |
| `max_iter` | `int` | `100` | Maximum number of iterations for the solver to converge. Increase if you see convergence warnings. |
| `l1_ratios` | `list[float]` | `[0.0]` | Elastic-net mixing parameter(s), relevant only when `solver="saga"` and penalty is `"elasticnet"`. `0.0` is pure L2; `1.0` is pure L1. Ignored for other solvers. |
| `n_jobs` | `int` | `-1` | Number of parallel jobs used for the cross-validation loop. `-1` uses all available CPU cores. |

#### Solver / penalty compatibility

| Solver | Supported penalties |
|---|---|
| `lbfgs` | L2, none |
| `liblinear` | L1, L2 |
| `newton-cg` | L2, none |
| `newton-cholesky` | L2, none |
| `sag` | L2, none |
| `saga` | L1, L2, elastic-net, none |

> **Tip:** For sparse, high-dimensional data or when you need L1 regularisation (feature selection), use `solver: saga` and set `l1_ratios: [1.0]`.

#### Accessing cross-validation results

After a run the underlying `LogisticRegressionCV` estimator is accessible inside the saved model:

```python
from model_builder.training.models.logr import LogisticRegressionModel

model = LogisticRegressionModel.load("output/my_run/models/3/model_3")
print(model._clf.C_)          # best C chosen per class
print(model._clf.scores_)     # CV scores for every C candidate
```

---

## Adding a new model

1. **Implement the interface** — create a new file under `models/` that subclasses `Model` and implements all five abstract methods.

2. **Add a config schema** — add a `<Name>Config` Pydantic model to `config/schema.py` with fields for every hyperparameter, then extend `ModelConfig`:

   ```python
   class ModelConfig(BaseModel):
       type: Literal["logr", "mymodel"] = "logr"
       logr: LogisticRegressionConfig = LogisticRegressionConfig()
       mymodel: MyModelConfig = MyModelConfig()
   ```

3. **Wire up the factory** — update whichever factory or dispatch logic instantiates models from the config so that `type: mymodel` creates an instance of your new class.

4. **Document it here** — add a section to this file following the same structure as the `logr` section above.
