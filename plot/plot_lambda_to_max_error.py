"""
Plot max error vs lambda (PGM poisoning experiments).
Also mean poisoning generation time (generation_time_sec) vs lambda for n=100.
Writes PDFs under fig/lambda_to_max_error/ (or fig/lambda_to_max_error_dup/ with --duplicate-only).
"""
import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from add_figure_legend import (
    add_figure_legend_outside_right,
    add_figure_legend_top_horizontal,
)
from load_results_max_error import (
    load_all_generation_times,
    load_all_max_errors,
    load_max_error_results,
)
from plot_config import (
    DATASET_IDS_800M_DUPLICATES,
    DATASET_IDS_800M_SWING,
    dataset_display_name,
)

# Figure settings
FIG_WIDTH_PER_COL = 6.0
FIG_HEIGHT_PER_ROW = 3.5
# Multi-dataset panels: 3 columns; rows = ceil(n_datasets / 3) (3×3 when n=9)
GRID_NUM_COLS = 3


def subplot_grid_shape(n_datasets: int) -> tuple:
    """Return (num_rows, num_cols) for laying out n_datasets panels."""
    num_cols = GRID_NUM_COLS
    if n_datasets <= 0:
        return (1, num_cols)
    num_rows = (n_datasets + num_cols - 1) // num_cols
    return (num_rows, num_cols)


# Plot settings
PLOT_MARKER = "o"
PLOT_LINESTYLE = "-"
PLOT_MARKERSIZE = 10
PLOT_LINEWIDTH = 2.5
PLOT_COLOR_BASELINE = "black"
PLOT_COLORS = {
    "random": "blue",
    "random_adjacent": "orange",
    "greedy": "green",
    "consec": "red",
    "optimal": "orange",
    "dup_optimal": "purple"
}
PLOT_MARKERS = {
    "random": "o",
    "random_adjacent": "p",
    "greedy": "s",
    "consec": "^",
    "optimal": "D",
    "dup_optimal": "v"
}
# Seed0 line styles (baseline E_L uses ":"; methods vary as much as possible)
SEED0_LINESTYLES = {
    "random": "-",
    "random_adjacent": (0, (1, 1)),
    "greedy": "--",
    "consec": "-.",
    "optimal": (0, (5, 1)),
    "dup_optimal": (0, (3, 1, 1, 1)),
}

# Method label map: method_key -> display label. Only methods in this dict are plotted.
# Add/remove keys to control which methods appear in plots.
LABEL_MAP = {
    "random": "Random",
    "random_adjacent": "Random-Adjacent",
    "greedy": "Greedy",
    "consec": "Consecutive",
    "optimal": "Optimal",
    "dup_optimal": "Duplicate-OPT",
}
# Methods to plot for n=100 (n=16 uses all keys in LABEL_MAP). Edit to include/exclude methods.
PLOT_METHODS_N100 = [
    "random",
    "random_adjacent",
    "greedy",
    "consec",
    "dup_optimal"
]
# Generation-time vs λ (n=100): omit Random / Random-Adjacent.
PLOT_METHODS_N100_GENERATION_TIME = [
    m for m in PLOT_METHODS_N100 if m not in ("random", "random_adjacent")
]

# Comparison boxplots (ratio / diff): (comparison_key, ratio_label, diff_label). Edit to choose which to plot.
COMPARISON_CONFIGS_N16 = [
    # ("E_C_E_L", "E_C / E_L", "E_C - E_L"),
    # ("E_C_E_R", "E_C / E_R", "E_C - E_R"),
    # ("E_A_E_R", "E_A / E_R", "E_A - E_R"),
    ("E_G_E_C", "E_C / E_G", "E_C - E_G"),
    ("E_C_E_O", "E_C / E_O", "E_C - E_O"),
    # ("E_O_E_D", "E_O / E_D", "E_O - E_D"),
    # ("E_R_E_L", "E_R / E_L", "E_R - E_L"),
]
COMPARISON_CONFIGS_N100 = [
    # ("E_C_E_L", "E_C / E_L", "E_C - E_L"),
    # ("E_C_E_R", "E_C / E_R", "E_C - E_R"),
    # ("E_A_E_R", "E_A / E_R", "E_A - E_R"),
    ("E_G_E_C", "E_C / E_G", "E_C - E_G"),
    # ("E_C_E_D", "E_D / E_C", "E_D - E_C"),
    # ("E_R_E_L", "E_R / E_L", "E_R - E_L"),
]

# Grid settings
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 1.0
GRID_COLOR = "gray"

# Font sizes
FONTSIZE_XLABEL = 36
FONTSIZE_YLABEL = 36
FONTSIZE_TITLE = 36
FONTSIZE_TICKS = 20
FONTSIZE_TEXT = 24
FONTSIZE_LEGEND = 40

# Text settings
TEXT_COLOR = "gray"

# Baseline line style
BASELINE_LINESTYLE = ":"
BASELINE_LINEWIDTH = 3

# Axis label for lambda (math)
XLABEL_LAMBDA = r"$\lambda$"
YLABEL_GENERATION_TIME_SEC = "Time [s]"


def format_E_label(s: str) -> str:
    if "E_" in s:
        """Convert E_X / E_XY to math: $E_{\mathrm{X}}$ (subscript roman). Handles 'E_C / E_L', 'E_C - E_L', etc."""
        return "$" + re.sub(r"E_([A-Z]+)", r"E_{\\mathrm{\1}}", s) + "$"
    else:
        return s


def set_shared_ylim_with_padding(axes, padding_ratio: float = 0.05):
    """Set the same ylim on all axes from the range of plotted data, with padding. Ignores axes with no data."""
    axes_with_data = [ax for ax in axes if len(ax.get_lines()) > 0 or len(ax.collections) > 0]
    if not axes_with_data:
        return
    y_min = min(ax.get_ylim()[0] for ax in axes_with_data)
    y_max = max(ax.get_ylim()[1] for ax in axes_with_data)
    r = y_max - y_min
    if r <= 0:
        r = abs(y_min) * 0.01 if y_min != 0 else 1.0
    margin = r * padding_ratio
    for ax in axes:
        ax.set_ylim(y_min - margin, y_max + margin)


def plot_single_dataset_seed0_n16(ax, results_dir: str, dataset: str, show_ylabel: bool = True, show_legend: bool = True):
    """Plot single dataset for seed=0, n=16 max_error vs lambda."""
    epsilon = 64
    seed = 0
    n = 16
    
    # Load baseline (lambda=0)
    baseline_df = load_max_error_results(
        results_dir, "random", dataset, epsilon, n, seed
    )
    row0 = baseline_df[baseline_df["lambda"] == 0]
    baseline_max_error = float(row0.iloc[0]["max_error"]) if not row0.empty else None

    if baseline_max_error is None:
        print(f"Warning: Could not find baseline (lambda=0) for {dataset}, n={n}, seed={seed}")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT, 
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return

    E_L = baseline_max_error
    print(f"seed0 n={n} {dataset}: E_L={E_L}, E_C(lambda=20)=N/A (n=16 uses lambda 1-5)")

    # Load data for each method (driven by LABEL_MAP)
    methods = list(LABEL_MAP.keys())
    lambdas = [1, 2, 3, 4, 5]
    
    # Plot baseline line
    ax.axhline(
        y=baseline_max_error,
        color=PLOT_COLOR_BASELINE,
        linestyle=":",
        linewidth=BASELINE_LINEWIDTH,
        label=format_E_label("Legitimate")
    )
    
    # Plot each method
    for method in methods:
        data = load_max_error_results(
            results_dir, method, dataset, epsilon, n, seed
        )
        
        # Filter data for specified lambdas
        method_lambdas = []
        method_max_errors = []
        
        for lambda_val in lambdas:
            sub = data[data["lambda"] == lambda_val]
            if not sub.empty:
                method_lambdas.append(lambda_val)
                method_max_errors.append(float(sub.iloc[0]["max_error"]))
        
        if method_lambdas:
            ax.plot(
                method_lambdas,
                method_max_errors,
                marker=PLOT_MARKERS[method],
                linestyle=SEED0_LINESTYLES[method],
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH,
                color=PLOT_COLORS[method],
                label=format_E_label(LABEL_MAP[method])
            )
    
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
    ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel("Max Error", fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
    if show_legend:
        ax.legend(
            fontsize=round(FONTSIZE_LEGEND),
            loc="upper left",
            ncol=2,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.2,
            columnspacing=1.0,
        )


def plot_seed0_n16(results_dir: str, datasets: List[str], output_dir: Path):
    """Plot seed=0, n=16 max_error vs lambda for multiple datasets."""
    seed = 0
    n = 16
    
    num_rows, num_cols = subplot_grid_shape(len(datasets))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8))
    axes = np.atleast_2d(axes)
    
    for i, dataset in enumerate(datasets):
        row, col = i // num_cols, i % num_cols
        plot_single_dataset_seed0_n16(axes[row, col], results_dir, dataset, show_ylabel=(col == 0), show_legend=False)
    
    for i in range(len(datasets), num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].set_visible(False)
    
    axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
    set_shared_ylim_with_padding(axes_flat)
    for row in range(num_rows):
        for col in range(1, num_cols):
            axes[row, col].tick_params(axis="y", labelleft=False)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    add_figure_legend_outside_right(
        fig, axes, len(datasets), num_cols, fontsize=round(FONTSIZE_LEGEND)
    )
    output_file = output_dir / "lambda_to_max_error_seed0_n16.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}", file=sys.stderr)


def plot_single_dataset_seed0_n100(ax, results_dir: str, dataset: str, show_ylabel: bool = True, show_legend: bool = True):
    """Plot single dataset for seed=0, n=100 max_error vs lambda."""
    epsilon = 64
    seed = 0
    n = 100
    
    # Load baseline (lambda=0)
    baseline_df = load_max_error_results(
        results_dir, "random", dataset, epsilon, n, seed
    )
    row0 = baseline_df[baseline_df["lambda"] == 0]
    baseline_max_error = float(row0.iloc[0]["max_error"]) if not row0.empty else None

    if baseline_max_error is None:
        print(f"Warning: Could not find baseline (lambda=0) for {dataset}, n={n}, seed={seed}")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT, 
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return
    
    E_L = baseline_max_error
    consec_df = load_max_error_results(results_dir, "consec", dataset, epsilon, n, seed)
    sub20 = consec_df[consec_df["lambda"] == 20]
    E_C_lambda20 = float(sub20.iloc[0]["max_error"]) if not sub20.empty else None
    print(f"seed0 n={n} {dataset}: E_L={E_L}, E_C(lambda=20)={E_C_lambda20}")
    
    # Load data for each method (driven by PLOT_METHODS_N100)
    methods = PLOT_METHODS_N100
    lambdas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Plot baseline line
    ax.axhline(
        y=baseline_max_error,
        color=PLOT_COLOR_BASELINE,
        linestyle=":",
        linewidth=BASELINE_LINEWIDTH,
        label=format_E_label("Legitimate")
    )
    
    # Plot each method
    for method in methods:
        data = load_max_error_results(
            results_dir, method, dataset, epsilon, n, seed
        )
        
        # Filter data for specified lambdas
        method_lambdas = []
        method_max_errors = []
        
        for lambda_val in lambdas:
            sub = data[data["lambda"] == lambda_val]
            if not sub.empty:
                method_lambdas.append(lambda_val)
                method_max_errors.append(float(sub.iloc[0]["max_error"]))
        
        if method_lambdas:
            ax.plot(
                method_lambdas,
                method_max_errors,
                marker=PLOT_MARKERS[method],
                linestyle=SEED0_LINESTYLES[method],
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH,
                color=PLOT_COLORS[method],
                label=format_E_label(LABEL_MAP[method])
            )
    
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
    ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel("Max Error", fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
    if show_legend:
        ax.legend(
            fontsize=round(FONTSIZE_LEGEND),
            loc="upper left",
            ncol=2,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.2,
            columnspacing=1.0,
        )


def plot_seed0_n100(results_dir: str, datasets: List[str], output_dir: Path):
    """Plot seed=0, n=100 max_error vs lambda for multiple datasets."""
    seed = 0
    n = 100
    
    num_rows, num_cols = subplot_grid_shape(len(datasets))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8))
    axes = np.atleast_2d(axes)
    
    for i, dataset in enumerate(datasets):
        row, col = i // num_cols, i % num_cols
        plot_single_dataset_seed0_n100(axes[row, col], results_dir, dataset, show_ylabel=(col == 0), show_legend=False)
    
    for i in range(len(datasets), num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].set_visible(False)
    
    axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
    set_shared_ylim_with_padding(axes_flat)
    for row in range(num_rows):
        for col in range(1, num_cols):
            axes[row, col].tick_params(axis="y", labelleft=False)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    add_figure_legend_outside_right(
        fig, axes, len(datasets), num_cols, fontsize=round(FONTSIZE_LEGEND)
    )
    output_file = output_dir / "lambda_to_max_error_seed0_n100.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}", file=sys.stderr)


def plot_single_dataset_mean_n100(
    ax,
    results_dir: str,
    dataset: str,
    cache: Dict[str, Dict[int, Dict[int, Optional[float]]]],
    show_ylabel: bool = True,
    show_legend: bool = True,
    *,
    show_xlabel: bool = True,
):
    """Plot single dataset for mean across 100 seeds, n=100 max_error vs lambda."""
    epsilon = 64
    n = 100
    seeds = list(range(100))
    lambdas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    methods = PLOT_METHODS_N100

    # Baseline E_L: mean of random method at lambda=0 across seeds
    baseline_values = []
    for seed in seeds:
        val = cache.get("random", {}).get(seed, {}).get(0)
        if val is not None:
            baseline_values.append(val)
    baseline_max_error = np.mean(baseline_values) if baseline_values else None

    if baseline_max_error is None:
        print(f"Warning: Could not find baseline (lambda=0) for {dataset}, n={n} (mean over seeds)")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        return

    print(f"mean n={n} {dataset}: E_L={baseline_max_error:.4f}")

    # Plot baseline line
    ax.axhline(
        y=baseline_max_error,
        color=PLOT_COLOR_BASELINE,
        linestyle=":",
        linewidth=BASELINE_LINEWIDTH,
        label=format_E_label("Legitimate")
    )

    # Plot each method (mean across seeds)
    for method in methods:
        method_lambdas = []
        method_max_errors = []

        for lambda_val in lambdas:
            values = []
            for seed in seeds:
                val = cache.get(method, {}).get(seed, {}).get(lambda_val)
                if val is not None:
                    values.append(val)
            if values:
                method_lambdas.append(lambda_val)
                method_max_errors.append(np.mean(values))

        if method_lambdas:
            ax.plot(
                method_lambdas,
                method_max_errors,
                marker=PLOT_MARKERS[method],
                linestyle=SEED0_LINESTYLES[method],
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH,
                color=PLOT_COLORS[method],
                label=format_E_label(LABEL_MAP[method])
            )

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
    if show_xlabel:
        ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("Max Error", fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
    if show_legend:
        ax.legend(
            fontsize=round(FONTSIZE_LEGEND),
            loc="lower right",
            ncol=3,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.2,
            columnspacing=1.0,
        )


def plot_mean_n100(results_dir: str, datasets: List[str], output_dir: Path):
    """Plot mean across 100 seeds, n=100 max_error vs lambda for multiple datasets."""
    n = 100
    epsilon = 64
    seeds = list(range(100))
    lambdas = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    methods = PLOT_METHODS_N100

    # Pre-load data for all datasets
    dataset_caches = {}
    for dataset in datasets:
        dataset_caches[dataset] = load_all_max_errors(
            results_dir, methods, dataset, epsilon, n, seeds, lambdas
        )

    num_rows, num_cols = subplot_grid_shape(len(datasets))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.9))
    axes = np.atleast_2d(axes)

    for i, dataset in enumerate(datasets):
        row, col = i // num_cols, i % num_cols
        show_ylabel = (col == 0)
        show_xlabel = row == num_rows - 1
        plot_single_dataset_mean_n100(
            axes[row, col],
            results_dir,
            dataset,
            dataset_caches[dataset],
            show_ylabel=show_ylabel,
            show_legend=False,
            show_xlabel=show_xlabel,
        )

    for i in range(len(datasets), num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].set_visible(False)

    axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
    set_shared_ylim_with_padding(axes_flat)
    for row in range(num_rows):
        for col in range(1, num_cols):
            axes[row, col].tick_params(axis="y", labelleft=False)
    # tight_layout first; legend placed from axis positions in add_figure_legend_top_horizontal
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    add_figure_legend_top_horizontal(
        fig,
        axes,
        len(datasets),
        num_cols,
        legend_ncol=3,
        fontsize=round(FONTSIZE_LEGEND),
    )
    output_file = output_dir / "lambda_to_max_error_mean_n100.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}", file=sys.stderr)


def _is_valid_gen_time(v: Optional[float]) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    return True


def plot_single_dataset_generation_time_mean_n100(
    ax,
    results_dir: str,
    dataset: str,
    cache: Dict[str, Dict[int, Dict[int, Optional[float]]]],
    show_ylabel: bool = True,
    show_legend: bool = True,
    *,
    show_xlabel: bool = True,
):
    """Mean poisoning generation time (generation_time_sec) vs lambda; n=100, mean over seeds 0..99."""
    n = 100
    seeds = list(range(100))
    lambdas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    methods = PLOT_METHODS_N100_GENERATION_TIME

    any_point = False
    for method in methods:
        for lambda_val in lambdas:
            for seed in seeds:
                val = cache.get(method, {}).get(seed, {}).get(lambda_val)
                if _is_valid_gen_time(val):
                    any_point = True
                    break
            if any_point:
                break
        if any_point:
            break

    if not any_point:
        print(
            f"Warning: No generation_time_sec for {dataset}, n={n} (mean over seeds)",
            file=sys.stderr,
        )
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
            ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        return

    for method in methods:
        method_lambdas: List[int] = []
        method_times: List[float] = []

        for lambda_val in lambdas:
            values = []
            for seed in seeds:
                val = cache.get(method, {}).get(seed, {}).get(lambda_val)
                if _is_valid_gen_time(val):
                    values.append(float(val))
            if values:
                method_lambdas.append(lambda_val)
                method_times.append(float(np.mean(values)))

        if method_lambdas:
            ax.plot(
                method_lambdas,
                method_times,
                marker=PLOT_MARKERS[method],
                linestyle=SEED0_LINESTYLES[method],
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH,
                color=PLOT_COLORS[method],
                label=format_E_label(LABEL_MAP[method]),
            )

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
    if show_xlabel:
        ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(YLABEL_GENERATION_TIME_SEC, fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
    ax.set_yscale("log")
    if show_legend:
        ax.legend(
            fontsize=round(FONTSIZE_LEGEND),
            loc="best",
            ncol=2,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.2,
            columnspacing=1.0,
        )


def plot_generation_time_mean_n100(results_dir: str, datasets: List[str], output_dir: Path):
    """Mean generation_time_sec vs lambda; n=100, averaged over seeds 0..99."""
    n = 100
    epsilon = 64
    seeds = list(range(100))
    lambdas = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    methods = PLOT_METHODS_N100_GENERATION_TIME

    dataset_caches = {}
    for dataset in datasets:
        dataset_caches[dataset] = load_all_generation_times(
            results_dir, methods, dataset, epsilon, n, seeds, lambdas
        )

    num_rows, num_cols = subplot_grid_shape(len(datasets))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.9),
    )
    axes = np.atleast_2d(axes)

    for i, dataset in enumerate(datasets):
        row, col = i // num_cols, i % num_cols
        show_ylabel = col == 0
        show_xlabel = row == num_rows - 1
        plot_single_dataset_generation_time_mean_n100(
            axes[row, col],
            results_dir,
            dataset,
            dataset_caches[dataset],
            show_ylabel=show_ylabel,
            show_legend=False,
            show_xlabel=show_xlabel,
        )

    for i in range(len(datasets), num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].set_visible(False)

    axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
    set_shared_ylim_with_padding(axes_flat)
    for row in range(num_rows):
        for col in range(1, num_cols):
            axes[row, col].tick_params(axis="y", labelleft=False)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    add_figure_legend_top_horizontal(
        fig,
        axes,
        len(datasets),
        num_cols,
        legend_ncol=3,
        fontsize=round(FONTSIZE_LEGEND),
    )
    output_file = output_dir / "lambda_to_generation_time_mean_n100.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}", file=sys.stderr)


def get_max_error_for_lambda(
    results_dir: str,
    method: str,
    dataset: str,
    epsilon: int,
    n: int,
    seed: int,
    lambda_val: int
) -> Optional[float]:
    """Get max_error for a specific lambda value."""
    data = load_max_error_results(
        results_dir, method, dataset, epsilon, n, seed
    )
    sub = data[data["lambda"] == lambda_val]
    if sub.empty:
        return None
    return float(sub.iloc[0]["max_error"])


def plot_single_boxplot_n16(ax, results_dir: str, dataset: str, comparison_key: str, comparison_label: str, calc_type: str, cache: Dict[str, Dict[int, Dict[int, Optional[float]]]], show_ylabel: bool = True, plot_type: str = "box", show_xlabel: bool = True):
    """Plot single boxplot or violinplot for seed=0-99, n=16 with multiple lambda values.
    
    Args:
        comparison_key: e.g., "E_C_E_L", "E_C_E_R", etc.
        comparison_label: e.g., "E_C / E_L", "E_C - E_L", etc.
        calc_type: "ratio" or "diff"
        cache: Pre-loaded max_error cache {method: {seed: {lambda: max_error}}}
        plot_type: "box" for boxplot, "violin" for violinplot
    """
    epsilon = 64
    n = 16
    lambdas = [1, 2, 3, 4, 5]  # All lambda values for n=16
    
    seeds = list(range(100))
    
    # Collect values for each lambda value
    all_values = []
    lambda_labels = []
    
    for lambda_val in lambdas:
        values = []
        
        for seed in seeds:
            # Get max_errors from cache
            E_L = cache.get("random", {}).get(seed, {}).get(0)
            E_R = cache.get("random", {}).get(seed, {}).get(lambda_val)
            E_A = cache.get("random_adjacent", {}).get(seed, {}).get(lambda_val)
            E_G = cache.get("greedy", {}).get(seed, {}).get(lambda_val)
            E_C = cache.get("consec", {}).get(seed, {}).get(lambda_val)
            E_O = cache.get("optimal", {}).get(seed, {}).get(lambda_val)
            E_D = cache.get("dup_optimal", {}).get(seed, {}).get(lambda_val)
            
            # Calculate value based on comparison_key and calc_type
            value = None
            if comparison_key == "E_A_E_R":
                if E_A is not None and E_R is not None:
                    if calc_type == "ratio":
                        value = E_A / E_R
                    elif calc_type == "diff":
                        value = E_A - E_R
            elif comparison_key == "E_C_E_L":
                if E_C is not None and E_L is not None:
                    if calc_type == "ratio":
                        value = E_C / E_L
                    elif calc_type == "diff":
                        value = E_C - E_L
            elif comparison_key == "E_C_E_R":
                if E_C is not None and E_R is not None:
                    if calc_type == "ratio":
                        value = E_C / E_R
                    elif calc_type == "diff":
                        value = E_C - E_R
            elif comparison_key == "E_G_E_C":
                if E_G is not None and E_C is not None:
                    if calc_type == "ratio":
                        value = E_C / E_G
                    elif calc_type == "diff":
                        value = E_C - E_G
            elif comparison_key == "E_C_E_O":
                if E_C is not None and E_O is not None:
                    if E_C > E_O:
                        print("ERROR: E_C > E_O (consec max_error > optimal max_error).", file=sys.stderr)
                        print(
                            f"  dataset={dataset}, n={n}, epsilon={epsilon}, seed={seed}, lambda={lambda_val}, "
                            f"E_C={E_C}, E_O={E_O}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    if calc_type == "ratio":
                        value = E_C / E_O
                    elif calc_type == "diff":
                        value = E_C - E_O
            elif comparison_key == "E_O_E_D":
                if E_O is not None and E_D is not None:
                    if calc_type == "ratio":
                        value = E_O / E_D
                    elif calc_type == "diff":
                        value = E_O - E_D
            elif comparison_key == "E_R_E_L":
                if E_R is not None and E_L is not None:
                    if calc_type == "ratio":
                        value = E_R / E_L
                    elif calc_type == "diff":
                        value = E_R - E_L
            
            if value is not None:
                values.append(value)
        
        if values:
            all_values.append(values)
            lambda_labels.append(str(lambda_val))
    
    if not all_values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT, 
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        return
    
    positions = list(range(1, len(all_values) + 1))
    if plot_type == "violin":
        vp = ax.violinplot(
            all_values,
            positions=positions,
            showmeans=True,
            showmedians=True,
            widths=0.6
        )
        # NOTE: Do not override facecolor to white; it makes violins look invisible.
    else:
        bp = ax.boxplot(
            all_values,
            labels=lambda_labels,
            patch_artist=True,
            widths=0.6
        )
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_alpha(1.0)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(lambda_labels)
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis='y')
    if show_xlabel:
        ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(format_E_label(comparison_label), fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")


def plot_boxplot_n16(results_dir: str, datasets: List[str], output_dir: Path):
    """Plot boxplots for seed=0-99, n=16 for multiple datasets."""
    n = 16
    epsilon = 64
    seeds = list(range(100))
    lambdas = [0, 1, 2, 3, 4, 5]  # Include lambda=0 for E_L
    methods = list(LABEL_MAP.keys())
    
    # Pre-load data for all datasets
    dataset_caches = {}
    for dataset in datasets:
        dataset_caches[dataset] = load_all_max_errors(
            results_dir, methods, dataset, epsilon, n, seeds, lambdas
        )
    
    for comparison_key, ratio_label, diff_label in COMPARISON_CONFIGS_N16:
        if comparison_key == "E_G_E_C":
            count_gt = count_eq = count_lt = count_none = 0
            box_lambdas = [1, 2, 3, 4, 5]
            for dataset in datasets:
                cache = dataset_caches[dataset]
                for seed in range(100):
                    for lambda_val in box_lambdas:
                        E_G = cache.get("greedy", {}).get(seed, {}).get(lambda_val)
                        E_C = cache.get("consec", {}).get(seed, {}).get(lambda_val)
                        if E_G is not None and E_C is not None:
                            if E_C > E_G:
                                count_gt += 1
                            elif E_C == E_G:
                                count_eq += 1
                            else:
                                count_lt += 1
                        else:
                            count_none += 1
            print(f"E_G_E_C n={n}: E_C > E_G: {count_gt}, E_C = E_G: {count_eq}, E_G > E_C: {count_lt}, None: {count_none}")
        if comparison_key == "E_C_E_D":
            count_gt = count_eq = count_lt = count_none = 0
            box_lambdas = [1, 2, 3, 4, 5]
            for dataset in datasets:
                cache = dataset_caches[dataset]
                for seed in range(100):
                    for lambda_val in box_lambdas:
                        E_C = cache.get("consec", {}).get(seed, {}).get(lambda_val)
                        E_D = cache.get("dup_optimal", {}).get(seed, {}).get(lambda_val)
                        if E_C is not None and E_D is not None:
                            if E_C > E_D:
                                count_gt += 1
                            elif E_C == E_D:
                                count_eq += 1
                            else:
                                count_lt += 1
                        else:
                            count_none += 1
            print(f"E_C_E_D n={n}: E_C > E_D: {count_gt}, E_C = E_D: {count_eq}, E_D > E_C: {count_lt}, None: {count_none}")
        # Plot ratio
        num_rows, num_cols = subplot_grid_shape(len(datasets))
        for plot_type in ["box"]:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8))
            axes = np.atleast_2d(axes)
            for i, dataset in enumerate(datasets):
                row, col = i // num_cols, i % num_cols
                plot_single_boxplot_n16(
                    axes[row, col], results_dir, dataset, comparison_key, ratio_label, "ratio", dataset_caches[dataset],
                    show_ylabel=(col == 0), plot_type=plot_type, show_xlabel=(row == num_rows - 1),
                )
            for i in range(len(datasets), num_rows * num_cols):
                row, col = i // num_cols, i % num_cols
                axes[row, col].set_visible(False)
            axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
            set_shared_ylim_with_padding(axes_flat)
            for row in range(num_rows):
                for col in range(1, num_cols):
                    axes[row, col].tick_params(axis="y", labelleft=False)
            plt.tight_layout()
            prefix = "boxplot" if plot_type == "box" else "violinplot"
            output_file = output_dir / f"lambda_to_max_error_{prefix}_{comparison_key}_ratio_n16.pdf"
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {output_file}", file=sys.stderr)
        
        # Plot diff
        for plot_type in ["box"]:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8))
            axes = np.atleast_2d(axes)
            for i, dataset in enumerate(datasets):
                row, col = i // num_cols, i % num_cols
                plot_single_boxplot_n16(
                    axes[row, col], results_dir, dataset, comparison_key, diff_label, "diff", dataset_caches[dataset],
                    show_ylabel=(col == 0), plot_type=plot_type, show_xlabel=(row == num_rows - 1),
                )
            for i in range(len(datasets), num_rows * num_cols):
                row, col = i // num_cols, i % num_cols
                axes[row, col].set_visible(False)
            axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
            set_shared_ylim_with_padding(axes_flat)
            for row in range(num_rows):
                for col in range(1, num_cols):
                    axes[row, col].tick_params(axis="y", labelleft=False)
            plt.tight_layout()
            prefix = "boxplot" if plot_type == "box" else "violinplot"
            output_file = output_dir / f"lambda_to_max_error_{prefix}_{comparison_key}_diff_n16.pdf"
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {output_file}", file=sys.stderr)


def plot_single_boxplot_n100(ax, results_dir: str, dataset: str, comparison_key: str, comparison_label: str, calc_type: str, cache: Dict[str, Dict[int, Dict[int, Optional[float]]]], show_ylabel: bool = True, plot_type: str = "box", show_xlabel: bool = True):
    """Plot single boxplot or violinplot for seed=0-99, n=100 with multiple lambda values.
    
    Args:
        comparison_key: e.g., "E_C_E_L", "E_C_E_R", etc.
        comparison_label: e.g., "E_C / E_L", "E_C - E_L", etc.
        calc_type: "ratio" or "diff"
        cache: Pre-loaded max_error cache {method: {seed: {lambda: max_error}}}
        plot_type: "box" for boxplot, "violin" for violinplot
    """
    epsilon = 64
    n = 100
    lambdas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # All lambda values for n=100
    
    seeds = list(range(100))
    
    # Collect values for each lambda value
    all_values = []
    lambda_labels = []
    
    for lambda_val in lambdas:
        values = []
        
        for seed in seeds:
            # Get max_errors from cache
            E_L = cache.get("random", {}).get(seed, {}).get(0)
            E_R = cache.get("random", {}).get(seed, {}).get(lambda_val)
            E_A = cache.get("random_adjacent", {}).get(seed, {}).get(lambda_val)
            E_G = cache.get("greedy", {}).get(seed, {}).get(lambda_val)
            E_C = cache.get("consec", {}).get(seed, {}).get(lambda_val)
            E_D = cache.get("dup_optimal", {}).get(seed, {}).get(lambda_val)
            
            # Calculate value based on comparison_key and calc_type
            value = None
            if comparison_key == "E_A_E_R":
                if E_A is not None and E_R is not None:
                    if calc_type == "ratio":
                        value = E_A / E_R
                    elif calc_type == "diff":
                        value = E_A - E_R
            elif comparison_key == "E_C_E_L":
                if E_C is not None and E_L is not None:
                    if calc_type == "ratio":
                        value = E_C / E_L
                    elif calc_type == "diff":
                        value = E_C - E_L
            elif comparison_key == "E_C_E_R":
                if E_C is not None and E_R is not None:
                    if calc_type == "ratio":
                        value = E_C / E_R
                    elif calc_type == "diff":
                        value = E_C - E_R
            elif comparison_key == "E_G_E_C":
                if E_G is not None and E_C is not None:
                    if calc_type == "ratio":
                        value = E_C / E_G
                    elif calc_type == "diff":
                        value = E_C - E_G
            elif comparison_key == "E_C_E_D":
                if E_C is not None and E_D is not None:
                    if E_C != E_D:
                        print(
                            f"E_C != E_D: dataset={dataset}, n={n}, epsilon={epsilon}, seed={seed}, lambda={lambda_val}, E_C={E_C}, E_D={E_D}",
                            file=sys.stderr,
                        )
                    if calc_type == "ratio":
                        value = E_D / E_C
                    elif calc_type == "diff":
                        value = E_D - E_C
            elif comparison_key == "E_R_E_L":
                if E_R is not None and E_L is not None:
                    if calc_type == "ratio":
                        value = E_R / E_L
                    elif calc_type == "diff":
                        value = E_R - E_L
            
            if value is not None:
                values.append(value)
        
        if values:
            all_values.append(values)
            lambda_labels.append(str(lambda_val))
    
    if not all_values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT, 
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        return
    
    positions = list(range(1, len(all_values) + 1))
    if plot_type == "violin":
        vp = ax.violinplot(
            all_values,
            positions=positions,
            showmeans=True,
            showmedians=True,
            widths=0.6
        )
        # NOTE: Do not override facecolor to white; it makes violins look invisible.
    else:
        bp = ax.boxplot(
            all_values,
            labels=lambda_labels,
            patch_artist=True,
            widths=0.6
        )
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_alpha(1.0)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(lambda_labels)
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis='y')
    if show_xlabel:
        ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(format_E_label(comparison_label), fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")


def plot_boxplot_n100(results_dir: str, datasets: List[str], output_dir: Path):
    """Plot boxplots for seed=0-99, n=100 for multiple datasets."""
    n = 100
    epsilon = 64
    seeds = list(range(100))
    lambdas = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Include lambda=0 for E_L
    methods = PLOT_METHODS_N100
    
    # Pre-load data for all datasets
    dataset_caches = {}
    for dataset in datasets:
        dataset_caches[dataset] = load_all_max_errors(
            results_dir, methods, dataset, epsilon, n, seeds, lambdas
        )
    
    for comparison_key, ratio_label, diff_label in COMPARISON_CONFIGS_N100:
        if comparison_key == "E_G_E_C":
            count_gt = count_eq = count_lt = count_none = 0
            box_lambdas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            for dataset in datasets:
                cache = dataset_caches[dataset]
                for seed in range(100):
                    for lambda_val in box_lambdas:
                        E_G = cache.get("greedy", {}).get(seed, {}).get(lambda_val)
                        E_C = cache.get("consec", {}).get(seed, {}).get(lambda_val)
                        if E_G is not None and E_C is not None:
                            if E_C > E_G:
                                count_gt += 1
                            elif E_C == E_G:
                                count_eq += 1
                            else:
                                count_lt += 1
                        else:
                            count_none += 1
            print(f"E_G_E_C n={n}: E_C > E_G: {count_gt}, E_C = E_G: {count_eq}, E_G > E_C: {count_lt}, None: {count_none}")
        if comparison_key == "E_C_E_D":
            count_gt = count_eq = count_lt = count_none = 0
            box_lambdas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            for dataset in datasets:
                cache = dataset_caches[dataset]
                for seed in range(100):
                    for lambda_val in box_lambdas:
                        E_C = cache.get("consec", {}).get(seed, {}).get(lambda_val)
                        E_D = cache.get("dup_optimal", {}).get(seed, {}).get(lambda_val)
                        if E_C is not None and E_D is not None:
                            if E_C > E_D:
                                count_gt += 1
                            elif E_C == E_D:
                                count_eq += 1
                            else:
                                count_lt += 1
                        else:
                            count_none += 1
            print(f"E_C_E_D n={n}: E_C > E_D: {count_gt}, E_C = E_D: {count_eq}, E_D > E_C: {count_lt}, None: {count_none}")
        # Plot ratio
        num_rows, num_cols = subplot_grid_shape(len(datasets))
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows))
        
        # if num_cols == 1:
        #     axes = [axes]
        
        # for i, dataset in enumerate(datasets):
        #     plot_single_boxplot_n100(axes[i], results_dir, dataset, comparison_key, ratio_label, "ratio", dataset_caches[dataset], show_ylabel=(i == 0))
        
        # set_shared_ylim_with_padding(axes)
        # for i in range(1, len(axes)):
        #     axes[i].tick_params(axis="y", labelleft=False)
        # plt.tight_layout()
        # output_file = output_dir / f"lambda_to_max_error_boxplot_{comparison_key}_ratio_n100.pdf"
        # plt.savefig(output_file, bbox_inches="tight")
        # plt.close()
        # print(f"Saved plot: {output_file}", file=sys.stderr)
        
        # Plot diff
        for plot_type in ["box"]:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8))
            axes = np.atleast_2d(axes)
            for i, dataset in enumerate(datasets):
                row, col = i // num_cols, i % num_cols
                plot_single_boxplot_n100(
                    axes[row, col], results_dir, dataset, comparison_key, diff_label, "diff", dataset_caches[dataset],
                    show_ylabel=(col == 0), plot_type=plot_type, show_xlabel=(row == num_rows - 1),
                )
            for i in range(len(datasets), num_rows * num_cols):
                row, col = i // num_cols, i % num_cols
                axes[row, col].set_visible(False)
            axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
            set_shared_ylim_with_padding(axes_flat)
            for row in range(num_rows):
                for col in range(1, num_cols):
                    axes[row, col].tick_params(axis="y", labelleft=False)
            plt.tight_layout()
            prefix = "boxplot" if plot_type == "box" else "violinplot"
            output_file = output_dir / f"lambda_to_max_error_{prefix}_{comparison_key}_diff_n100.pdf"
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {output_file}", file=sys.stderr)


def plot_single_maxerror_boxplot_n16(ax, results_dir: str, dataset: str, method: str, method_label: str, cache: Dict[str, Dict[int, Dict[int, Optional[float]]]], show_ylabel: bool = True, plot_type: str = "box", show_xlabel: bool = True):
    """Plot single maxerror boxplot or violinplot for seed=0-99, n=16 with multiple lambda values.
    
    Args:
        method: e.g., "random", "greedy", "consec", "optimal", "dup_optimal"
        method_label: e.g., "E_R", "E_G", "E_C", "E_O", "E_D"
        cache: Pre-loaded max_error cache {method: {seed: {lambda: max_error}}}
        plot_type: "box" for boxplot, "violin" for violinplot
    """
    epsilon = 64
    n = 16
    lambdas = [0, 1, 2, 3, 4, 5]  # All lambda values for n=16, including lambda=0
    
    seeds = list(range(100))
    
    # Collect max_errors for each lambda value
    all_max_errors = []
    lambda_labels = []
    
    for lambda_val in lambdas:
        max_errors = []
        
        for seed in seeds:
            max_error = cache.get(method, {}).get(seed, {}).get(lambda_val)
            if max_error is not None:
                max_errors.append(max_error)
        
        if max_errors:
            all_max_errors.append(max_errors)
            lambda_labels.append(str(lambda_val))
    
    if not all_max_errors:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT, 
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        return
    
    positions = list(range(1, len(all_max_errors) + 1))
    if plot_type == "violin":
        vp = ax.violinplot(
            all_max_errors,
            positions=positions,
            showmeans=True,
            showmedians=True,
            widths=0.6
        )
        # NOTE: Do not override facecolor to white; it makes violins look invisible.
    else:
        bp = ax.boxplot(
            all_max_errors,
            labels=lambda_labels,
            patch_artist=True,
            widths=0.6
        )
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_alpha(1.0)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(lambda_labels)
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis='y')
    if show_xlabel:
        ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(format_E_label(method_label), fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")


def plot_maxerror_boxplot_n16(results_dir: str, datasets: List[str], output_dir: Path):
    """Plot maxerror boxplots for seed=0-99, n=16 for multiple datasets."""
    n = 16
    epsilon = 64
    seeds = list(range(100))
    lambdas = [0, 1, 2, 3, 4, 5]
    methods = list(LABEL_MAP.keys())
    
    # Pre-load data for all datasets
    dataset_caches = {}
    for dataset in datasets:
        dataset_caches[dataset] = load_all_max_errors(
            results_dir, methods, dataset, epsilon, n, seeds, lambdas
        )
    
    # Plot each method (driven by LABEL_MAP)
    num_rows, num_cols = subplot_grid_shape(len(datasets))
    for method, method_label in LABEL_MAP.items():
        for plot_type in ["box"]:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8))
            axes = np.atleast_2d(axes)
            for i, dataset in enumerate(datasets):
                row, col = i // num_cols, i % num_cols
                plot_single_maxerror_boxplot_n16(
                    axes[row, col], results_dir, dataset, method, method_label, dataset_caches[dataset],
                    show_ylabel=(col == 0), plot_type=plot_type, show_xlabel=(row == num_rows - 1),
                )
            for i in range(len(datasets), num_rows * num_cols):
                row, col = i // num_cols, i % num_cols
                axes[row, col].set_visible(False)
            axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
            set_shared_ylim_with_padding(axes_flat)
            for row in range(num_rows):
                for col in range(1, num_cols):
                    axes[row, col].tick_params(axis="y", labelleft=False)
            plt.tight_layout()
            prefix = "boxplot" if plot_type == "box" else "violinplot"
            output_file = output_dir / f"lambda_to_max_error_{prefix}_{method_label}_n16.pdf"
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {output_file}", file=sys.stderr)


def plot_single_maxerror_boxplot_n100(ax, results_dir: str, dataset: str, method: str, method_label: str, cache: Dict[str, Dict[int, Dict[int, Optional[float]]]], show_ylabel: bool = True, plot_type: str = "box", show_xlabel: bool = True):
    """Plot single maxerror boxplot or violinplot for seed=0-99, n=100 with multiple lambda values.
    
    Args:
        method: e.g., "random", "greedy", "consec", "dup_optimal"
        method_label: e.g., "E_R", "E_G", "E_C", "E_D"
        cache: Pre-loaded max_error cache {method: {seed: {lambda: max_error}}}
        plot_type: "box" for boxplot, "violin" for violinplot
    """
    epsilon = 64
    n = 100
    lambdas = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # All lambda values for n=100, including lambda=0
    
    seeds = list(range(100))
    
    # Collect max_errors for each lambda value
    all_max_errors = []
    lambda_labels = []
    
    for lambda_val in lambdas:
        max_errors = []
        
        for seed in seeds:
            max_error = cache.get(method, {}).get(seed, {}).get(lambda_val)
            if max_error is not None:
                max_errors.append(max_error)
        
        if max_errors:
            all_max_errors.append(max_errors)
            lambda_labels.append(str(lambda_val))
    
    if not all_max_errors:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT, 
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        return
    
    positions = list(range(1, len(all_max_errors) + 1))
    if plot_type == "violin":
        vp = ax.violinplot(
            all_max_errors,
            positions=positions,
            showmeans=True,
            showmedians=True,
            widths=0.6
        )
        # NOTE: Do not override facecolor to white; it makes violins look invisible.
    else:
        bp = ax.boxplot(
            all_max_errors,
            labels=lambda_labels,
            patch_artist=True,
            widths=0.6
        )
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_alpha(1.0)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(lambda_labels)
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis='y')
    if show_xlabel:
        ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(format_E_label(method_label), fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")


def plot_maxerror_boxplot_n100(results_dir: str, datasets: List[str], output_dir: Path):
    """Plot maxerror boxplots for seed=0-99, n=100 for multiple datasets."""
    n = 100
    epsilon = 64
    seeds = list(range(100))
    lambdas = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    methods = PLOT_METHODS_N100
    
    # Pre-load data for all datasets
    dataset_caches = {}
    for dataset in datasets:
        dataset_caches[dataset] = load_all_max_errors(
            results_dir, methods, dataset, epsilon, n, seeds, lambdas
        )
    
    # Plot each method (driven by PLOT_METHODS_N100 and LABEL_MAP)
    num_rows, num_cols = subplot_grid_shape(len(datasets))
    for method in PLOT_METHODS_N100:
        method_label = LABEL_MAP[method]
        for plot_type in ["box"]:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8))
            axes = np.atleast_2d(axes)
            for i, dataset in enumerate(datasets):
                row, col = i // num_cols, i % num_cols
                plot_single_maxerror_boxplot_n100(
                    axes[row, col], results_dir, dataset, method, method_label, dataset_caches[dataset],
                    show_ylabel=(col == 0), plot_type=plot_type, show_xlabel=(row == num_rows - 1),
                )
            for i in range(len(datasets), num_rows * num_cols):
                row, col = i // num_cols, i % num_cols
                axes[row, col].set_visible(False)
            axes_flat = [axes[i // num_cols, i % num_cols] for i in range(len(datasets))]
            set_shared_ylim_with_padding(axes_flat)
            for row in range(num_rows):
                for col in range(1, num_cols):
                    axes[row, col].tick_params(axis="y", labelleft=False)
            plt.tight_layout()
            prefix = "boxplot" if plot_type == "box" else "violinplot"
            output_file = output_dir / f"lambda_to_max_error_{prefix}_{method_label}_n100.pdf"
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot max error vs lambda (PGM poisoning experiments)."
    )
    parser.add_argument(
        "--duplicate-only",
        action="store_true",
        help=(
            "Only duplicate-key datasets (bench2/bench3/wiki_ts_200M/zipf); "
            "write under fig/lambda_to_max_error_dup/."
        ),
    )
    args = parser.parse_args()

    results_dir = "results/poisoning_maxerror"
    datasets = list(
        DATASET_IDS_800M_DUPLICATES if args.duplicate_only else DATASET_IDS_800M_SWING
    )
    output_base_dir = (
        "fig/lambda_to_max_error_dup"
        if args.duplicate_only
        else "fig/lambda_to_max_error"
    )

    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # # Plot seed=0, n=16
    # plot_seed0_n16(results_dir, datasets, output_base_path)
    
    # # Plot seed=0, n=100
    # plot_seed0_n100(results_dir, datasets, output_base_path)

    # Plot mean across 100 seeds, n=100
    plot_mean_n100(results_dir, datasets, output_base_path)

    # Mean poisoning generation time vs lambda (n=100, mean over seeds)
    plot_generation_time_mean_n100(results_dir, datasets, output_base_path)

    # # Plot boxplots for n=16
    # plot_boxplot_n16(results_dir, datasets, output_base_path)
    
    # Plot boxplots for n=100
    plot_boxplot_n100(results_dir, datasets, output_base_path)
    
    # # Plot maxerror boxplots for n=16
    # plot_maxerror_boxplot_n16(results_dir, datasets, output_base_path)
    
    # Plot maxerror boxplots for n=100
    plot_maxerror_boxplot_n100(results_dir, datasets, output_base_path)
