"""Shared plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt


def annotate_bar_values(bar_patches, values: list[float], y_offset: float = 0.002) -> None:
    """Annotate vertical bars with numeric values."""
    for patch, value in zip(bar_patches, values):
        x_center = patch.get_x() + patch.get_width() * 0.5
        y_value = value + y_offset
        plt.text(
            x_center,
            y_value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
