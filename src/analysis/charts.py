"""Reusable chart functions for analysis notebooks."""
from pathlib import Path

import matplotlib.pyplot as plt
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
    ax.barh(data.index, data.values)
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
    ax.boxplot(groups, labels=labels)
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
