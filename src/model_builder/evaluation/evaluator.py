"""Binary classification evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from model_builder.config.schema import PipelineConfig
from model_builder.training.base import Model

# (split_name, y_true, y_proba)
_SplitData = List[Tuple[str, np.ndarray, np.ndarray]]

_COLORS: Dict[str, str] = {
    "train": "#1f77b4",
    "test": "#ff7f0e",
    "val": "#2ca02c",
    "overall": "#9467bd",
}


# ---------------------------------------------------------------------------
# Module-level metric helpers (importable for unit testing)
# ---------------------------------------------------------------------------


def _optimal_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return the decision threshold on *y_proba* that maximises F1 on *y_true*."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # arrays have length n+1; thresholds has length n
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_vals = np.where(
            (precision[:-1] + recall[:-1]) > 0,
            2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
            0.0,
        )
    return float(thresholds[int(np.argmax(f1_vals))])


def _compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """Full binary classification metric set for one split.

    If *threshold* is ``None`` it is derived by maximising F1 via
    ``_optimal_threshold_f1``.
    """
    if threshold is None:
        threshold = _optimal_threshold_f1(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "n_rows": int(len(y_true)),
        "actual_positive_rate": round(float(y_true.mean()), 6),
        "sum_predicted_probs": round(float(y_proba.sum()), 4),
        "log_loss": round(float(log_loss(y_true, y_proba)), 6),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 6),
        "auc_pr": round(float(average_precision_score(y_true, y_proba)), 6),
        "auc_roc": round(float(roc_auc_score(y_true, y_proba)), 6),
        "optimal_threshold_f1": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def _random_baseline_metrics(
    y_true: np.ndarray,
    base_rate: float,
) -> Dict[str, object]:
    """Metrics for a naive predictor that always outputs *base_rate* (threshold=0.5).

    Simulates a caller who knows nothing but the training positive rate.
    When base_rate < 0.5 the classifier predicts all-negative; when ≥ 0.5
    it predicts all-positive.  Probability metrics (log_loss, brier, AUC)
    reflect the calibration quality of a constant predictor.
    """
    y_proba = np.full(len(y_true), base_rate)
    return _compute_metrics(y_true, y_proba, threshold=0.5)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class BinaryClassificationEvaluator:
    """Evaluate a trained binary classifier and persist CSV + PDF artifacts."""

    def __init__(self, config: PipelineConfig, model_id: int) -> None:
        self._config = config
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Model,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
        val: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "BinaryClassificationEvaluator":
        """Run evaluation and write artifacts.

        Parameters
        ----------
        model:
            Fitted model with a ``predict_proba`` method.
        train / test / val:
            ``(X, y)`` tuples **after** preprocessing; *val* may be ``None``.
        """
        splits: _SplitData = [
            ("train", train[1].to_numpy().astype(int), model.predict_proba(train[0])),
            ("test", test[1].to_numpy().astype(int), model.predict_proba(test[0])),
        ]
        if val is not None:
            splits.append(
                (
                    "val",
                    val[1].to_numpy().astype(int),
                    model.predict_proba(val[0]),
                )
            )

        # Derive threshold from train only — no leakage into test/val.
        train_y, train_p = splits[0][1], splits[0][2]
        train_threshold = _optimal_threshold_f1(train_y, train_p)
        train_base_rate = float(train[1].mean())
        self._write_csv(splits, train_base_rate, train_threshold)
        self._write_pdf(splits, train_base_rate, train_threshold)
        return self

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _eval_dir(self) -> Path:
        return (
            Path(self._config.base.output_dir)
            / self._config.base.name
            / "evaluation"
            / str(self._model_id)
        )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _write_csv(
        self, splits: _SplitData, train_base_rate: float, train_threshold: float
    ) -> None:
        y_all = np.concatenate([s[1] for s in splits])
        p_all = np.concatenate([s[2] for s in splits])

        rows = []
        for name, y, p in splits:
            rows.append(
                {"split": name, **_compute_metrics(y, p, threshold=train_threshold)}
            )  # type: ignore[arg-type]
        rows.append(
            {
                "split": "overall",
                **_compute_metrics(y_all, p_all, threshold=train_threshold),
            }
        )  # type: ignore[arg-type]
        rows.append(
            {"split": "random", **_random_baseline_metrics(y_all, train_base_rate)}
        )  # type: ignore[arg-type]

        out = self._eval_dir()
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _write_pdf(
        self, splits: _SplitData, train_base_rate: float, train_threshold: float
    ) -> None:
        out = self._eval_dir()
        out.mkdir(parents=True, exist_ok=True)
        with PdfPages(out / "report.pdf") as pdf:
            self._page_summary_table(pdf, splits, train_base_rate)
            self._page_roc_pr_curves(pdf, splits)
            self._page_histograms(pdf, splits)
            self._page_calibration(pdf, splits)
            self._page_confusion_matrices(pdf, splits, train_threshold)

    def _title(self) -> str:
        return f"{self._config.base.name}  ·  model {self._model_id}"

    # --- Page 1: summary table ---

    def _page_summary_table(
        self,
        pdf: PdfPages,
        splits: _SplitData,
        train_base_rate: float,
    ) -> None:
        y_all = np.concatenate([s[1] for s in splits])
        p_all = np.concatenate([s[2] for s in splits])
        display: _SplitData = splits + [("overall", y_all, p_all)]

        col_headers = ["Split", "N Samples", "Log Loss", "AUC ROC", "AUC PR"]
        rows = []
        for name, y, p in display:
            m = _compute_metrics(y, p)
            rows.append(
                [
                    name,
                    int(m["n_rows"]),  # type: ignore[arg-type]
                    f"{m['log_loss']:.4f}",
                    f"{m['auc_roc']:.4f}",
                    f"{m['auc_pr']:.4f}",
                ]
            )

        # single random row using overall data
        rm = _random_baseline_metrics(y_all, train_base_rate)
        rows.append(
            [
                "random (overall)",
                int(rm["n_rows"]),  # type: ignore[arg-type]
                f"{rm['log_loss']:.4f}",
                f"{rm['auc_roc']:.4f}",
                f"{rm['auc_pr']:.4f}",
            ]
        )

        n_data_rows = len(rows)
        fig, ax = plt.subplots(figsize=(9, max(3.0, 0.55 * (n_data_rows + 1) + 1.5)))
        ax.axis("off")
        tbl = ax.table(
            cellText=rows,
            colLabels=col_headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.1, 2.2)

        for j in range(len(col_headers)):
            tbl[0, j].set_facecolor("#2c3e50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, n_data_rows + 1):
            shade = "#f0f0f0" if i % 2 == 0 else "white"
            for j in range(len(col_headers)):
                tbl[i, j].set_facecolor(shade)
        # highlight random row (last)
        for j in range(len(col_headers)):
            tbl[n_data_rows, j].set_facecolor("#fff3cd")

        fig.suptitle(
            f"{self._title()}  ·  Binary Classification Summary",
            fontsize=12,
            fontweight="bold",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # --- Page 2: ROC + PR curves ---

    def _page_roc_pr_curves(self, pdf: PdfPages, splits: _SplitData) -> None:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 6))

        for name, y, p in splits:
            color = _COLORS.get(name, "#555555")
            fpr, tpr, _ = roc_curve(y, p)
            ax_roc.plot(
                fpr,
                tpr,
                color=color,
                lw=2,
                label=f"{name}  (AUC={roc_auc_score(y, p):.3f})",
            )
            prec, rec, _ = precision_recall_curve(y, p)
            ax_pr.plot(
                rec,
                prec,
                color=color,
                lw=2,
                label=f"{name}  (AP={average_precision_score(y, p):.3f})",
            )

        # random reference lines
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves")
        ax_roc.legend(loc="lower right", fontsize=9)
        ax_roc.set_xlim([-0.01, 1.01])
        ax_roc.set_ylim([-0.01, 1.01])
        ax_roc.grid(True, alpha=0.3)

        overall_pos_rate = float(np.concatenate([s[1] for s in splits]).mean())
        ax_pr.axhline(
            overall_pos_rate,
            color="black",
            lw=1,
            ls="--",
            alpha=0.5,
            label=f"random  (AP={overall_pos_rate:.3f})",
        )
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curves")
        ax_pr.legend(loc="upper right", fontsize=9)
        ax_pr.set_xlim([-0.01, 1.01])
        ax_pr.set_ylim([-0.01, 1.01])
        ax_pr.grid(True, alpha=0.3)

        fig.suptitle(f"{self._title()}  ·  AUC Curves", fontsize=13, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # --- Page 3: prediction histograms (one subplot per split) ---

    def _page_histograms(self, pdf: PdfPages, splits: _SplitData) -> None:
        n = len(splits)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]

        bins = np.linspace(0, 1, 31)
        for ax, (name, y, p) in zip(axes, splits):
            ax.hist(p[y == 0], bins=bins, alpha=0.65, color="#2196F3", label="Class 0")
            ax.hist(p[y == 1], bins=bins, alpha=0.65, color="#F44336", label="Class 1")
            ax.set_title(name)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{self._title()}  ·  Prediction Distributions by True Class",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # --- Page 4: calibration curves ---

    def _page_calibration(self, pdf: PdfPages, splits: _SplitData) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))

        for name, y, p in splits:
            frac_pos, mean_pred = calibration_curve(
                y, p, n_bins=10, strategy="quantile"
            )
            ax.plot(
                mean_pred,
                frac_pos,
                marker="o",
                lw=2,
                markersize=5,
                color=_COLORS.get(name, "#555555"),
                label=name,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curves")
        ax.legend(fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"{self._title()}  ·  Calibration", fontsize=13, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # --- Page 5: confusion matrices ---

    def _page_confusion_matrices(
        self, pdf: PdfPages, splits: _SplitData, train_threshold: float
    ) -> None:
        n = len(splits)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]

        for ax, (name, y, p) in zip(axes, splits):
            y_pred = (p >= train_threshold).astype(int)
            ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax, colorbar=False)
            ax.set_title(f"{name}  (thresh={train_threshold:.3f})")

        fig.suptitle(
            f"{self._title()}  ·  Confusion Matrices  (Optimal F1 Threshold)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
