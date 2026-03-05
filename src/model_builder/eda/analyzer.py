import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from model_builder.config.schema import PipelineConfig


def _describe_distribution(series: pd.Series) -> str:
    """Return a brief label for the distribution shape of a numeric series."""
    clean = series.dropna()
    n_unique = clean.nunique()

    if n_unique <= 2:
        return "binary"
    if n_unique <= 10:
        return "ordinal/categorical"

    # Normality test (requires ≥8 observations)
    if len(clean) >= 8:
        _, p_normal = stats.normaltest(clean)
        if p_normal > 0.05:
            return "gaussian"

    # Uniformity: chi-square test on histogram bins
    counts, _ = np.histogram(clean, bins=10)
    _, p_uniform = stats.chisquare(counts)
    if p_uniform > 0.05:
        return "uniform"

    skew = stats.skew(clean)
    if skew > 1.0:
        return "right-skewed"
    if skew < -1.0:
        return "left-skewed"

    return "non-gaussian"


class DataAnalyzer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_path = os.path.join(config.base.output_dir, config.base.name, "eda")
        os.makedirs(self.output_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> None:
        plots_dir = os.path.join(self.output_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        pdf_path = os.path.join(self.output_path, "eda.pdf")
        with PdfPages(pdf_path) as pdf:
            self._add_describe(df, pdf)
            self._add_correlations(df, pdf)
            self._add_histograms(df, plots_dir, pdf)
            self._add_outliers(df, pdf)

    # ------------------------------------------------------------------
    # Describe table
    # ------------------------------------------------------------------

    def _add_describe(self, df: pd.DataFrame, pdf: PdfPages) -> None:
        desc = df.describe(include="all").T
        desc.insert(0, "nulls", df.isnull().sum())

        dist_labels: List[str] = []
        for col in desc.index:
            if pd.api.types.is_numeric_dtype(df[col]):
                dist_labels.append(_describe_distribution(df[col]))
            else:
                dist_labels.append("categorical")
        desc["distribution"] = dist_labels

        desc.to_csv(os.path.join(self.output_path, "describe.csv"))

        # Render as a table in the PDF, chunked so rows stay readable
        chunk_size = 20
        index_list = list(desc.index)
        for start in range(0, len(index_list), chunk_size):
            chunk = desc.loc[index_list[start : start + chunk_size]]
            fig, ax = plt.subplots(figsize=(14, max(4, len(chunk) * 0.45 + 1.5)))
            ax.axis("off")

            tbl_df = chunk.reset_index().rename(columns={"index": "column"})

            # Format values for display
            def _fmt(x: object) -> str:
                if isinstance(x, float):
                    return f"{x:.4g}"
                try:
                    if pd.isna(x):  # type: ignore[arg-type]
                        return ""
                except (TypeError, ValueError):
                    pass
                return str(x)

            display = tbl_df.apply(lambda col: col.map(_fmt))

            tbl = ax.table(
                cellText=display.values.tolist(),
                colLabels=list(display.columns),
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.auto_set_column_width(col=list(range(len(display.columns))))

            title = "Data Summary"
            if start > 0:
                end = min(start + chunk_size, len(index_list))
                title += f" (rows {start + 1}–{end})"
            ax.set_title(title, fontsize=10, pad=12)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------

    def _add_correlations(self, df: pd.DataFrame, pdf: PdfPages) -> None:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return

        corr = numeric_df.corr()
        corr.to_csv(os.path.join(self.output_path, "correlations.csv"))

        n = len(corr)
        fig_size = max(6, n * 0.7)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        # RdBu: red = negative, white = 0, blue = positive
        im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)

        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                text_color = "white" if abs(val) >= 0.7 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

        ax.set_title("Feature Correlations", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------

    def _add_histograms(self, df: pd.DataFrame, plots_dir: str, pdf: PdfPages) -> None:
        target_col: Optional[str] = getattr(self.config.data, "target_column", None)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        def _is_categorical(s: pd.Series) -> bool:
            return s.nunique() <= 10

        def _is_normal(s: pd.Series) -> bool:
            clean = s.dropna()
            if len(clean) < 8:
                return True
            _, p = stats.normaltest(clean)
            return p > 0.05

        cols_to_plot: List[str] = []

        # Target column first (if numeric)
        if target_col and target_col in numeric_cols:
            cols_to_plot.append(target_col)

        # Then non-normal, non-categorical columns
        for col in numeric_cols:
            if col in cols_to_plot:
                continue
            if not _is_categorical(df[col]) and not _is_normal(df[col]):
                cols_to_plot.append(col)

        # Finally, remaining numeric columns not yet included
        for col in numeric_cols:
            if col not in cols_to_plot:
                cols_to_plot.append(col)

        for col in cols_to_plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            clean = df[col].dropna()
            ax.hist(clean, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            label = _describe_distribution(df[col])
            ax.set_title(f"{col}  [{label}]")
            fig.tight_layout()

            png_path = os.path.join(plots_dir, f"{col}_histogram.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def _add_outliers(self, df: pd.DataFrame, pdf: PdfPages) -> None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return

        records = []
        for col in numeric_cols:
            clean = df[col].dropna()
            q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (clean < lower) | (clean > upper)
            n_out = int(mask.sum())
            records.append(
                {
                    "column": col,
                    "method": "IQR (1.5x)",
                    "n_outliers": n_out,
                    "pct_outliers": (
                        round(100.0 * n_out / len(clean), 2) if len(clean) else 0.0
                    ),
                    "lower_fence": round(lower, 4),
                    "upper_fence": round(upper, 4),
                }
            )

        outlier_df = pd.DataFrame(records)
        outlier_df.to_csv(os.path.join(self.output_path, "outliers.csv"), index=False)

        # --- Summary table page ---
        fig, ax = plt.subplots(figsize=(12, max(3, len(outlier_df) * 0.45 + 1.5)))
        ax.axis("off")

        def _fmt(x: object) -> str:
            if isinstance(x, float):
                return f"{x:.4g}"
            return str(x)

        display = outlier_df.apply(lambda c: c.map(_fmt))
        tbl = ax.table(
            cellText=display.values.tolist(),
            colLabels=list(display.columns),
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(len(display.columns))))

        # Highlight rows with outliers
        for row_idx, n_out in enumerate(outlier_df["n_outliers"], start=1):
            if n_out > 0:
                for col_idx in range(len(display.columns)):
                    tbl[row_idx, col_idx].set_facecolor("#ffe0e0")

        ax.set_title("Outlier Summary (IQR 1.5× fences)", fontsize=11, pad=12)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Boxplot grid page(s) ---
        cols_per_page = 6
        for page_start in range(0, len(numeric_cols), cols_per_page):
            page_cols = numeric_cols[page_start : page_start + cols_per_page]
            ncols = min(3, len(page_cols))
            nrows = -(-len(page_cols) // ncols)  # ceiling division
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
            )

            for ax_idx, col in enumerate(page_cols):
                ax = axes[ax_idx // ncols][ax_idx % ncols]
                clean = df[col].dropna()
                q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = clean[(clean < lower) | (clean > upper)]

                ax.boxplot(
                    clean,
                    orientation="vertical",
                    patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6),
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="", linestyle="none"),  # hide default fliers
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                )
                # Overlay outliers as red dots with jitter for visibility
                if len(outliers):
                    rng = np.random.default_rng(self.config.base.seed)
                    jitter = rng.uniform(-0.05, 0.05, size=len(outliers))
                    ax.scatter(
                        np.ones(len(outliers)) + jitter,
                        outliers.values,
                        color="red",
                        s=18,
                        zorder=5,
                        label=f"{len(outliers)} outlier(s)",
                    )
                    ax.legend(fontsize=7, loc="upper right")

                ax.set_title(col, fontsize=9)
                ax.set_xticks([])

            # Hide unused axes
            for ax_idx in range(len(page_cols), nrows * ncols):
                axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

            title = "Outlier Boxplots"
            if len(numeric_cols) > cols_per_page:
                title += f" ({page_start + 1}–{page_start + len(page_cols)})"
            fig.suptitle(title, fontsize=12, y=1.01)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
