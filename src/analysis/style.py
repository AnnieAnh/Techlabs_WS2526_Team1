"""Matplotlib/seaborn theme configuration for analysis notebooks.

Usage in notebooks:
    from analysis.style import set_style
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def set_style() -> None:
    """Configure matplotlib/seaborn theme for consistent notebook styling."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })
