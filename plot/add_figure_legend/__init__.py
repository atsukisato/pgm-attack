"""Multi-panel figure legend helpers (single fig.legend for subplot grids)."""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def collect_subplot_grid_legend_handles_labels(
    axes: Any,
    num_datasets: int,
    num_cols: int,
) -> Tuple[List, List]:
    """Handles/labels for fig.legend: prefer last dataset panel, then fallbacks."""
    if num_datasets <= 0:
        return [], []
    axes_arr = np.atleast_2d(axes)
    last_i = num_datasets - 1
    legend_ax_row, legend_ax_col = last_i // num_cols, last_i % num_cols
    ax_legend = axes_arr[legend_ax_row, legend_ax_col]
    handles, labels = ax_legend.get_legend_handles_labels()
    if not handles:
        handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if not handles:
        for idx in range(num_datasets):
            r, c = idx // num_cols, idx % num_cols
            h, lab = axes_arr[r, c].get_legend_handles_labels()
            if h:
                handles, labels = h, lab
                break
    return handles, labels


def add_figure_legend_outside_right(
    fig: Any,
    axes: Any,
    num_datasets: int,
    num_cols: int,
    *,
    ncol: int = 2,
    fontsize: int = 14,
) -> None:
    """
    Place a single figure legend to the right of the subplot grid.
    Call after plt.tight_layout(rect=[0, 0, right, 1]).
    """
    handles, labels = collect_subplot_grid_legend_handles_labels(
        axes, num_datasets, num_cols
    )
    if not handles:
        return
    axes_arr = np.atleast_2d(axes)
    last_i = num_datasets - 1
    legend_ax_row, legend_ax_col = last_i // num_cols, last_i % num_cols
    ax_legend = axes_arr[legend_ax_row, legend_ax_col]
    box = ax_legend.get_position()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(box.x1 + 0.02, box.y0 + box.height / 2),
        loc="center left",
        fontsize=fontsize,
        ncol=ncol,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.2,
        columnspacing=1.0,
    )


def add_figure_legend_top_horizontal(
    fig: Any,
    axes: Any,
    num_datasets: int,
    num_cols: int,
    *,
    pad: float = 0.05,
    legend_ncol: int = 3,
    fontsize: int = 14,
) -> None:
    """One horizontal legend centered just above the subplot grid (figure coordinates)."""
    handles, labels = collect_subplot_grid_legend_handles_labels(
        axes, num_datasets, num_cols
    )
    if not handles:
        return
    axes_arr = np.atleast_2d(axes)
    ymax = 0.0
    for ax in axes_arr.ravel():
        if ax.get_visible():
            ymax = max(ymax, ax.get_position().y1)
    y_legend_bottom = min(0.999, ymax + pad)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y_legend_bottom),
        bbox_transform=fig.transFigure,
        ncol=legend_ncol,
        fontsize=fontsize,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.2,
        columnspacing=1.2,
    )


__all__ = [
    "add_figure_legend_outside_right",
    "add_figure_legend_top_horizontal",
    "collect_subplot_grid_legend_handles_labels",
]
