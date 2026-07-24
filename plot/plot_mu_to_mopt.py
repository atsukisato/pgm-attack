import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from load_results_mopt import (
    discover_datasets,
    discover_epsilons,
    load_lambda0_baseline,
    load_mu_to_mopt_data,
)
from plot_config import DATASET_IDS_1M, DATASET_IDS_1M_DUPLICATES, dataset_display_name
from add_figure_legend import add_figure_legend_top_horizontal

# Figure settings
FIG_WIDTH_PER_COL = 4.0
FIG_HEIGHT_PER_ROW = 3.5
GRID_NUM_COLS = 3

# Plot settings
PLOT_MARKERSIZE = 8
PLOT_LINEWIDTH = 2.5
# (color, marker, linestyle) per lambda
LAMBDA_STYLES = [
    ("#1f77b4", "o", "-"),   # blue, circle
    ("#ff7f0e", "s", "--"),  # orange, square
    ("#2ca02c", "^", "-."),  # green, triangle
    ("#d62728", "x", ":"),   # red, x
    ("#9467bd", "D", "-"),   # purple, diamond
]
# (color, marker, linestyle) per epsilon for fixed-lambda overlay (order: 16, 32, 64, 128)
EPSILON_STYLES = [
    ("#1f77b4", "o", "-"),   # blue, circle
    ("#ff7f0e", "s", "--"),  # orange, square
    ("#2ca02c", "^", "-."),  # green, triangle
    ("#d62728", "D", ":"),   # red, diamond
]

# Grid settings
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 1.0
GRID_COLOR = "gray"

# Font sizes
FONTSIZE_XLABEL = 24
FONTSIZE_YLABEL = 24
FONTSIZE_TITLE = 24
FONTSIZE_TICKS = 19
FONTSIZE_TEXT = 24
FONTSIZE_LEGEND = 28

# Text settings
TEXT_COLOR = "gray"


def _set_log_axis_plain_tick_labels(ax: plt.Axes, axis: str) -> None:
    """
    Log-scaled axis:
      - Keep default tick locations (1.1, 2, 3, ...)
      - Show plain numbers instead of 1.1×10^0
      - Hide minor tick labels
    """
    if axis not in ("x", "y"):
        raise ValueError(axis)

    def formatter_func(v, _p):
        cands = [
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            1.8, 2, 2.2,
            3, 5, 10, 20, 50
        ]
        for cand in cands:
            if abs(v - cand) < 1e-6:
                return f"{cand:g}"
        return ""

    formatter = mticker.FuncFormatter(formatter_func)

    if axis == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(formatter)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(formatter)


def plot_single_subplot(
    ax,
    base_dir: str,
    dataset: str,
    epsilon: int,
    lambda_values: List[int],
    show_legend: bool = True,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """
    Plot mu vs m_opt for one dataset.
    - Plot curves for each lambda in one ax.
    - Plot lambda=0 baseline as horizontal dashed line.
    """
    has_data = False

    # Draw lambda=0 baseline as horizontal dashed line
    m_opt_baseline, m_opt_baseline_std = load_lambda0_baseline(base_dir, dataset, epsilon)
    if m_opt_baseline is not None:
        ax.axhline(
            y=m_opt_baseline,
            linestyle="--",
            color="black",
            linewidth=3.0,
            zorder=0
        )

    # Plot mu vs m_opt for each lambda
    for i, lambda_val in enumerate(lambda_values):
        data = load_mu_to_mopt_data(base_dir, dataset, lambda_val, epsilon)
        if not data:
            continue

        mus = [x[0] for x in data]
        m_opts = [x[1] for x in data]

        has_data = True
        style = LAMBDA_STYLES[i % len(LAMBDA_STYLES)]
        color, marker, linestyle = style
        ax.plot(
            mus,
            m_opts,
            marker=marker,
            linestyle=linestyle,
            markersize=PLOT_MARKERSIZE,
            linewidth=PLOT_LINEWIDTH,
            color=color,
            label=rf"$\lambda$={lambda_val}",
        )

    if has_data:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1, base=10)
        ax.set_xlim(-0.2, None)
        _set_log_axis_plain_tick_labels(ax, "y")
        ax.grid(
            True,
            which="both",
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            color=GRID_COLOR,
        )
        if show_xlabel:
            ax.set_xlabel(r"$\mu$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(r"$m_{\mathrm{opt}}$", fontsize=FONTSIZE_YLABEL)
        else:
            ax.set_ylabel("")
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND)
    else:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=FONTSIZE_TEXT,
            transform=ax.transAxes,
            color=TEXT_COLOR,
        )
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(r"$\mu$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(r"$m_{\mathrm{opt}}$", fontsize=FONTSIZE_YLABEL)
        else:
            ax.set_ylabel("")


def plot_single_subplot_fixed_lambda_epsilons(
    ax,
    base_dir: str,
    dataset: str,
    lambda_fixed: int,
    epsilon_values: List[int],
    show_legend: bool = True,
    y_axis_ratio: bool = True,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """
    Plot mu vs m_opt (or ratio) for one dataset with lambda fixed and multiple epsilons overlaid.
    - Fix lambda_fixed and plot curves for each epsilon in one ax.
    - When y_axis_ratio=True, the Y-axis label is m_opt increase (value is m_opt(λ)/m_opt(λ=0)), and lambda=0 is a horizontal dashed line at y=1.
    - When y_axis_ratio=False, the Y-axis is the absolute value of m_opt, and the lambda=0 baseline for each epsilon is a horizontal dashed line.
    """
    has_data = False

    if y_axis_ratio:
        # Ratio plot: baseline is y=1
        ax.axhline(
            y=1.0,
            linestyle="--",
            color="black",
            linewidth=3.0,
            zorder=0
        )

    for i, eps in enumerate(epsilon_values):
        m_opt_baseline, _ = load_lambda0_baseline(base_dir, dataset, eps)
        if not y_axis_ratio and m_opt_baseline is not None:
            style = EPSILON_STYLES[i % len(EPSILON_STYLES)]
            color, _, _ = style
            ax.axhline(
                y=m_opt_baseline,
                linestyle="--",
                color=color,
                linewidth=3.0,
                zorder=0,
                alpha=0.6
            )

    for i, eps in enumerate(epsilon_values):
        m_opt_baseline, _ = load_lambda0_baseline(base_dir, dataset, eps)
        data = load_mu_to_mopt_data(base_dir, dataset, lambda_fixed, eps)
        if not data:
            continue
        if y_axis_ratio and (m_opt_baseline is None or m_opt_baseline == 0):
            continue

        mus = [x[0] for x in data]
        m_opts_raw = [x[1] for x in data]
        if y_axis_ratio:
            m_opts = [m / m_opt_baseline for m in m_opts_raw]
        else:
            m_opts = m_opts_raw

        has_data = True
        style = EPSILON_STYLES[i % len(EPSILON_STYLES)]
        color, marker, linestyle = style
        ax.plot(
            mus,
            m_opts,
            marker=marker,
            linestyle=linestyle,
            markersize=PLOT_MARKERSIZE,
            linewidth=PLOT_LINEWIDTH,
            color=color,
            label=rf"$\varepsilon$={eps}",
        )

    if has_data:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1, base=10)
        ax.set_xlim(-0.2, None)
        _set_log_axis_plain_tick_labels(ax, "y")
        ax.grid(
            True,
            which="both",
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            color=GRID_COLOR,
        )
        if show_xlabel:
            ax.set_xlabel(r"$\mu$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.set_xlabel("")
        y_lbl = (
            r"$m_{\mathrm{opt}}$ increase"
            if y_axis_ratio
            else r"$m_{\mathrm{opt}}$"
        )
        if show_ylabel:
            ax.set_ylabel(y_lbl, fontsize=FONTSIZE_YLABEL)
        else:
            ax.set_ylabel("")
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND)
    else:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=FONTSIZE_TEXT,
            transform=ax.transAxes,
            color=TEXT_COLOR,
        )
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(r"$\mu$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.set_xlabel("")
        y_lbl = (
            r"$m_{\mathrm{opt}}$ increase"
            if y_axis_ratio
            else r"$m_{\mathrm{opt}}$"
        )
        if show_ylabel:
            ax.set_ylabel(y_lbl, fontsize=FONTSIZE_YLABEL)
        else:
            ax.set_ylabel("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mu vs m_opt (swing_lambda_with_theta).")
    parser.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Only duplicate-key 1M datasets; write under fig/mu_to_mopt_dup/.",
    )
    args = parser.parse_args()

    base_dir = "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
    output_dir = Path("fig/mu_to_mopt_dup" if args.duplicate_only else "fig/mu_to_mopt")
    out_pdf_suffix = "_dup" if args.duplicate_only else ""

    LAMBDA_VALUES = [10000, 20000, 50000, 100000, 200000]

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(DATASET_IDS_1M_DUPLICATES if args.duplicate_only else DATASET_IDS_1M)
    # Filter to datasets that exist
    base_path = Path(base_dir)
    datasets = [ds for ds in datasets if (base_path / ds).exists()]
    if not datasets:
        print(f"No datasets found in {base_dir}")
        exit(1)

    # Discover epsilons from first available lambda+dataset
    # epsilons = []
    # for lam in LAMBDA_VALUES:
    #     def has_lambda(ds: str) -> bool:
    #         p = Path(base_dir) / ds
    #         if (p / f"lambda{lam}").exists():
    #             return True
    #         for sd in p.iterdir():
    #             if sd.is_dir() and sd.name.startswith("seed") and (sd / f"lambda{lam}").exists():
    #                 return True
    #         return False
    #     if any(has_lambda(ds) for ds in datasets):
    #         for ds in datasets:
    #             epsilons.extend(discover_epsilons(base_dir, ds, lam))
    #         if epsilons:
    #             epsilons = sorted(set(epsilons))
    #             break

    # if not epsilons:
    #     print("No epsilon data found")
    #     exit(1)

    num_cols = GRID_NUM_COLS
    num_rows = max(1, (len(datasets) + num_cols - 1) // num_cols)

    # for eps in epsilons:
    #     fig, axes = plt.subplots(
    #         num_rows,
    #         num_cols,
    #         figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
    #         squeeze=False,
    #     )

    #     for i, dataset in enumerate(datasets):
    #         row, col = i // num_cols, i % num_cols
    #         ax = axes[row, col]
    #         plot_single_subplot(
    #             ax,
    #             base_dir,
    #             dataset,
    #             eps,
    #             LAMBDA_VALUES,
    #             show_legend=False,
    #             show_xlabel=row == num_rows - 1,
    #             show_ylabel=col == 0,
    #         )
    #     for i in range(len(datasets), num_rows * num_cols):
    #         row, col = i // num_cols, i % num_cols
    #         axes[row, col].set_visible(False)

    #     plt.tight_layout(rect=[0, 0, 1, 0.94])
    #     add_figure_legend_top_horizontal(
    #         fig,
    #         axes,
    #         len(datasets),
    #         num_cols,
    #         legend_ncol=3,
    #         fontsize=FONTSIZE_TICKS,
    #     )
    #     output_file = output_dir / f"mu_to_mopt_epsilon{eps}.pdf"
    #     plt.savefig(output_file, bbox_inches="tight")
    #     plt.close()
    #     print(f"Saved plot: {output_file}", file=sys.stderr)

    # Lambda=100000 fixed, epsilon=16,32,64,128 overlaid
    lambda_fixed = 100000
    epsilon_values = [16, 32, 64, 128]
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
        squeeze=False,
    )
    for i, dataset in enumerate(datasets):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col]
        plot_single_subplot_fixed_lambda_epsilons(
            ax,
            base_dir,
            dataset,
            lambda_fixed,
            epsilon_values,
            show_legend=False,
            y_axis_ratio=True,
            show_xlabel=row == num_rows - 1,
            show_ylabel=col == 0,
        )
    for i in range(len(datasets), num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    add_figure_legend_top_horizontal(
        fig,
        axes,
        len(datasets),
        num_cols,
        legend_ncol=5,
        fontsize=FONTSIZE_LEGEND,
    )
    output_file = output_dir / f"mu_to_mopt_lambda100000_epsilon_overlay{out_pdf_suffix}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}", file=sys.stderr)
