"""Reusable chart functions for analysis notebooks."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set by notebooks: import charts as _charts; _charts.FIGURES_DIR = Path(...)
FIGURES_DIR: Path | None = None


def horizontal_bar(
    series: pd.Series,
    title: str,
    xlabel: str = "Count",
    top_n: int = 20,
    figsize: tuple = (10, 6),
    save_as: str | None = None,
) -> None:
    """Horizontal bar chart of a value-counted Series (top_n entries).

    Args:
        series: Series of categorical values to count.
        title: Chart title.
        xlabel: X-axis label.
        top_n: Maximum number of bars to show (sorted descending).
        figsize: Figure size (width, height) in inches.
        save_as: Filename to save under FIGURES_DIR (skipped if None or FIGURES_DIR unset).
    """
    data = series.value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(list(data.index), data.values.tolist())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def heatmap(
    data: pd.DataFrame,
    title: str,
    figsize: tuple = (12, 8),
    fmt: str = ".0f",
    save_as: str | None = None,
) -> None:
    """Annotated heatmap from a pivot-table DataFrame.

    Args:
        data: 2-D DataFrame (rows = y-axis, columns = x-axis).
        title: Chart title.
        figsize: Figure size in inches.
        fmt: Format string for cell annotations.
        save_as: Filename to save under FIGURES_DIR.

    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            ax.text(j, i, format(data.values[i, j], fmt), ha="center", va="center", fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def time_series(
    series: pd.Series,
    title: str,
    ylabel: str = "Count",
    figsize: tuple = (12, 4),
    save_as: str | None = None,
) -> None:
    """Line chart for a time-indexed Series.

    Args:
        series: Series with a date/datetime index and numeric values.
        title: Chart title.
        ylabel: Y-axis label.
        figsize: Figure size in inches.
        save_as: Filename to save under FIGURES_DIR.

    """
    fig, ax = plt.subplots(figsize=figsize)
    series.sort_index().plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def box_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    figsize: tuple = (12, 6),
    save_as: str | None = None,
) -> None:
    """Box plot of numeric column `y` grouped by categorical `x`.

    Args:
        df: Source DataFrame.
        x: Categorical grouping column name.
        y: Numeric value column name.
        title: Chart title.
        figsize: Figure size in inches.
        save_as: Filename to save under FIGURES_DIR.

    """
    order = sorted(df[x].dropna().unique())
    groups = [df[df[x] == k][y].dropna().values for k in order]
    labels = [str(k) for k in order]
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(groups, tick_labels=labels)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def stacked_bar(
    df: pd.DataFrame,
    title: str,
    figsize: tuple = (12, 6),
    save_as: str | None = None,
) -> None:
    """Stacked bar chart from a cross-tabulation DataFrame.

    Args:
        df: DataFrame with numeric values (rows = x categories, columns = stack segments).
        title: Chart title.
        figsize: Figure size in inches.
        save_as: Filename to save under FIGURES_DIR.

    """
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def value_bar(
    index: "pd.Index | list",
    values: "pd.Series | list",
    title: str,
    xlabel: str = "",
    figsize: tuple = (10, 6),
    fmt: str = ",.0f",
    save_as: str | None = None,
) -> None:
    """Horizontal bar chart from pre-computed values (not value_counts).

    Use this instead of horizontal_bar when you already have the bar heights
    (e.g. median salary by city, benefit percentages).

    Args:
        index: Category labels for the y-axis.
        values: Numeric bar lengths (one per label).
        title: Chart title.
        xlabel: X-axis label (e.g. 'Annual salary (EUR)').
        figsize: Figure size (width, height) in inches.
        fmt: Python format spec for value annotations (e.g. ',.0f' or '€,.0f').
        save_as: Filename to save under FIGURES_DIR (skipped if None or FIGURES_DIR unset).

    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(index, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for i, v in enumerate(values):
        ax.text(v, i, format(v, fmt), va="center", fontsize=9)
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def histogram(
    series: pd.Series,
    title: str,
    xlabel: str = "",
    bins: int = 30,
    figsize: tuple = (10, 5),
    save_as: str | None = None,
) -> None:
    """Histogram of a numeric Series.

    Args:
        series: Numeric values to plot.
        title: Chart title.
        xlabel: X-axis label.
        bins: Number of histogram bins.
        figsize: Figure size in inches.
        save_as: Filename to save under FIGURES_DIR.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(series.dropna(), bins=bins, edgecolor="white", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def dual_histogram(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str,
    label_b: str,
    title: str,
    xlabel: str = "",
    bins: int = 30,
    figsize: tuple = (10, 5),
    save_as: str | None = None,
) -> None:
    """Overlaid histogram of two numeric Series.

    Args:
        series_a: First numeric Series.
        series_b: Second numeric Series.
        label_a: Legend label for series_a.
        label_b: Legend label for series_b.
        title: Chart title.
        xlabel: X-axis label.
        bins: Number of histogram bins.
        figsize: Figure size in inches.
        save_as: Filename to save under FIGURES_DIR.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(series_a.dropna(), bins=bins, alpha=0.6, label=label_a, edgecolor="white")
    ax.hist(series_b.dropna(), bins=bins, alpha=0.6, label=label_b, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()
    plt.close(fig)


def radar_chart(
    data: pd.DataFrame,
    title: str,
    figsize: tuple = (8, 8),
    save_as: str | None = None,
) -> None:
    """Radar (spider) chart comparing multiple categories across shared axes.

    Args:
        data: DataFrame with categories as index, axes as columns.
            Values should be normalised to 0–100 for best visual results.
        title: Chart title.
        figsize: Figure size in inches.
        save_as: Filename to save under FIGURES_DIR.
    """
    categories = list(data.columns)
    n_axes = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    ax.set_title(title, y=1.08, fontsize=14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)

    colors = plt.cm.Set2(np.linspace(0, 1, len(data.index)))
    for (label, row), color in zip(data.iterrows(), colors):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    if save_as and FIGURES_DIR:
        fig.savefig(FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
