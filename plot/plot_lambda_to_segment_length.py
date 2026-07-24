"""
Plot segment length vs lambda (PGM poisoning experiments).
Writes PDFs under fig/lambda_to_segment_length/ (or *_dup/ with --duplicate-only).
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from load_results_segment_length import load_all_segment_lengths, load_segment_length_results
from plot_config import (
    DATASET_IDS_800M_DUPLICATES,
    DATASET_IDS_800M_SWING,
    dataset_display_name,
)
from add_figure_legend import add_figure_legend_top_horizontal

# Figure settings (align with plot_lambda_to_max_error)
FIG_WIDTH_PER_COL = 5.5
FIG_HEIGHT_PER_ROW = 3.5
# Multi-dataset panels: 3 columns; rows = ceil(n_datasets / 3) (3×3 when n=9)
GRID_NUM_COLS = 3
PLOT_MARKERSIZE = 12

def subplot_grid_rows(n_datasets: int, num_cols: int = GRID_NUM_COLS) -> int:
    """Number of subplot rows for n_datasets panels in a fixed num_cols-wide grid."""
    if n_datasets <= 0:
        return 1
    return (n_datasets + num_cols - 1) // num_cols


# Grid / style
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 1.0
GRID_COLOR = "gray"
FONTSIZE_XLABEL = 32
FONTSIZE_YLABEL = 32
FONTSIZE_TITLE = 32
FONTSIZE_TICKS = 22
FONTSIZE_LEGEND = 36
FONTSIZE_TEXT = 24
TEXT_COLOR = "gray"
XLABEL_LAMBDA = r"$\lambda$"
XTICK_LABEL_EVERY = 6  # show x-tick label every N positions to avoid overlap

# Epsilon sets
EPSILONS_CC = [16, 32, 64]  # for minimize_segment_length (C_C): 9 graphs
EPSILONS_CS = [16, 32, 64, 128]  # for minimize_segment_length_swing (C_S): 12 graphs (up to epsilon 128)
EPSILONS_CC_MINUS_CS = [16, 32, 64]  # for diff (C_S - C_C) and (C_S - C_C) / C_L: 9 graphs

def format_C_label(s: str) -> str:
    if "C_" in s:
        """Convert C_X / C_XY to math: $C_{\mathrm{X}}$ (subscript roman)."""
        return "$" + re.sub(r"C_([A-Z]+)", r"C_{\\mathrm{\1}}", s) + "$"
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


def apply_ax_y_power10_mantissa_ticks(ax, *, label_fontsize: Optional[int] = None) -> None:
    """Place $\\times 10^{m}$ at upper-left and show y-tick labels as one-decimal mantissa $y/10^m$ (per-axis). Skips if $m=0$ or ylim invalid."""
    ylo, yhi = ax.get_ylim()
    if not (np.isfinite(ylo) and np.isfinite(yhi)):
        return
    ref = max(abs(ylo), abs(yhi))
    if ref <= 0 or not np.isfinite(ref):
        return
    m = int(np.floor(np.log10(ref)))
    if m == 0:
        return
    scale = 10.0 ** m
    fs = label_fontsize if label_fontsize is not None else FONTSIZE_TICKS - 2
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, min_n_ticks=4))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos, s=scale: f"{y / s:.1f}")
    )
    # Upper-left corner of the axes (tight to panel edge; clip_on=False avoids tight_layout clipping)
    ax.text(
        -0.05,
        1.2,
        rf"$\times 10^{{{m}}}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fs + 4,
        clip_on=False,
    )


# C mean plot: methods and styles (align with lambda_to_max_error_mean_n100)
CONSEC_LABEL_NAME = "Consecutive"
SWING_CONSEC_LABEL_NAME = "DI-Consecutive"
RAND_LABEL_NAME = "Random"
RAND_ADJACENT_LABEL_NAME = "Random-Adjacent"

C_METHODS = [
    ("minimize_segment_length", CONSEC_LABEL_NAME, "C0", "o", "-"),
    ("minimize_segment_length_swing", SWING_CONSEC_LABEL_NAME, "C1", "X", (0, (1, 1))),
    ("minimize_segment_length_random", RAND_LABEL_NAME, "C2", "p", (0, (1, 1))),
    ("minimize_segment_length_random_adjacent", RAND_ADJACENT_LABEL_NAME, "C3", "D", (0, (5, 1))),
]
PLOT_COLOR_L = "black"
PLOT_LINESTYLE_L = ":"
PLOT_LINEWIDTH_L = 4.0


def plot_single_dataset_mean_C(
    ax,
    results_dir: str,
    dataset: str,
    cache: Dict[str, Dict[int, Dict[int, Optional[float]]]],
    seeds: List[int],
    lambdas: List[int],
    epsilon: int,
    show_ylabel: bool = True,
    show_legend: bool = True,
    *,
    show_epsilon_in_subplot_title: bool = True,
    show_xlabel: bool = True,
):
    """Plot single dataset: mean C across seeds vs lambda. C_C, C_S, C_R, C_A, C_AI as lines. C_L as black dashed."""
    # C_L: mean of baseline (lambda=0) across seeds. Use any method that has lambda=0.
    baseline_values = []
    for method in [m[0] for m in C_METHODS]:
        for seed in seeds:
            val = cache.get(method, {}).get(seed, {}).get(0)
            if val is not None:
                baseline_values.append(val)
        if baseline_values:
            break

    C_L_mean = np.mean(baseline_values) if baseline_values else None

    if C_L_mean is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        _ttl = (
            f"{dataset_display_name(dataset)} ($\\varepsilon={epsilon}$)"
            if show_epsilon_in_subplot_title
            else dataset_display_name(dataset)
        )
        ax.set_title(_ttl, fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        return

    # Plot C_L baseline (black dashed)
    ax.axhline(
        y=C_L_mean,
        color=PLOT_COLOR_L,
        linestyle=PLOT_LINESTYLE_L,
        linewidth=PLOT_LINEWIDTH_L,
        label=format_C_label("Legitimate")
    )

    # Plot each method (mean across seeds)
    lambdas_for_plot = [lam for lam in lambdas if lam > 0]  # exclude 0 for method lines
    for method, label, color, marker, linestyle in C_METHODS:
        if method not in cache:
            continue
        method_lambdas = []
        method_means = []

        for lambda_val in lambdas_for_plot:
            values = []
            for seed in seeds:
                val = cache.get(method, {}).get(seed, {}).get(lambda_val)
                if val is not None:
                    values.append(val)
            if values:
                method_lambdas.append(lambda_val)
                method_means.append(np.mean(values))

        if method_lambdas:
            ax.plot(
                method_lambdas,
                method_means,
                marker=marker,
                linestyle=linestyle,
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH_L,
                color=color,
                label=format_C_label(label)
            )

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
    if show_xlabel:
        ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(r"$C$", fontsize=FONTSIZE_YLABEL)
    _ttl = (
        f"{dataset_display_name(dataset)} ($\\varepsilon={epsilon}$)"
        if show_epsilon_in_subplot_title
        else dataset_display_name(dataset)
    )
    ax.set_title(_ttl, fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
    if show_legend:
        ax.legend(
            fontsize=round(FONTSIZE_LEGEND),
            loc="upper right",
            ncol=3,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.2,
            columnspacing=1.0,
        )

    apply_ax_y_power10_mantissa_ticks(ax)


def plot_lambda_to_C_mean(
    results_dir: str,
    datasets: List[str],
    output_dir: Path,
    epsilons: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
):
    """Plot mean C across seeds vs lambda. One PDF per epsilon; 3 columns × ceil(n/3) rows."""
    seeds = seeds if seeds is not None else list(range(20))
    eps_list = epsilons if epsilons is not None else EPSILONS_CC
    methods = [m[0] for m in C_METHODS]

    num_cols = GRID_NUM_COLS
    n_datasets = len(datasets)
    num_rows_per_eps = subplot_grid_rows(n_datasets, num_cols)

    # Pre-load data for all (dataset, epsilon) pairs
    dataset_epsilon_caches: Dict[tuple, Dict] = {}
    all_lambdas = set()
    for epsilon in eps_list:
        for dataset in datasets:
            for method in methods:
                data = load_segment_length_results(
                    results_dir, method, dataset, epsilon, seed=None,
                    intercept_candidate_num=None,
                    allow_intercept_fallback=True,
                )
                for d in data.to_dict("records"):
                    all_lambdas.add(d["lambda"])
    lambdas = sorted(all_lambdas) if all_lambdas else [0, 16, 32, 48, 64, 80, 96, 112, 128]

    for epsilon in eps_list:
        for dataset in datasets:
            dataset_epsilon_caches[(dataset, epsilon)] = load_all_segment_lengths(
                results_dir, methods, dataset, epsilon, seeds, lambdas
            )

    for epsilon in eps_list:
        fig, axes = plt.subplots(
            num_rows_per_eps,
            num_cols,
            figsize=(
                FIG_WIDTH_PER_COL * num_cols,
                FIG_HEIGHT_PER_ROW * num_rows_per_eps * 0.85,
            ),
            squeeze=False,
        )
        axes = np.atleast_2d(axes)

        for i, dataset in enumerate(datasets):
            row, col = i // num_cols, i % num_cols
            show_ylabel = (col == 0)
            show_xlabel = row == num_rows_per_eps - 1
            plot_single_dataset_mean_C(
                axes[row, col],
                results_dir,
                dataset,
                dataset_epsilon_caches[(dataset, epsilon)],
                seeds=seeds,
                lambdas=lambdas,
                epsilon=epsilon,
                show_ylabel=show_ylabel,
                show_legend=False,
                show_epsilon_in_subplot_title=False,
                show_xlabel=show_xlabel,
            )

        for j in range(len(datasets), num_rows_per_eps * num_cols):
            row, col = j // num_cols, j % num_cols
            axes[row, col].set_visible(False)

        plt.subplots_adjust(wspace=0.10, hspace=0.15)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        add_figure_legend_top_horizontal(
            fig,
            axes,
            len(datasets),
            num_cols,
            legend_ncol=3,
            fontsize=round(FONTSIZE_LEGEND),
        )
        out_file = output_dir / f"lambda_to_C_mean_eps{epsilon}.pdf"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {out_file}", file=sys.stderr)


def plot_single_boxplot_CC(
    ax,
    results_dir: str,
    dataset: str,
    epsilon: int,
    show_ylabel: bool = True,
    show_legend: bool = False,
    intercept_candidate_num: Optional[int] = None,
):
    """Boxplot of C_C = covered_keys_after_attack vs lambda for minimize_segment_length."""
    data = load_segment_length_results(
        results_dir, "minimize_segment_length", dataset, epsilon, seed=None,
        intercept_candidate_num=None,
    )
    # Group by lambda: list of C_C = covered_keys_after_attack (over seeds)
    by_lambda: Dict[int, List[float]] = {}
    for d in data.to_dict("records"):
        lam = d["lambda"]
        if lam not in by_lambda:
            by_lambda[lam] = []
        by_lambda[lam].append(d["covered_keys_after_attack"])

    if not by_lambda:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return

    lambdas_sorted = sorted(by_lambda.keys())
    all_values = [by_lambda[lam] for lam in lambdas_sorted]
    lambda_labels = [str(lam) for lam in lambdas_sorted]

    bp = ax.boxplot(
        all_values,
        labels=lambda_labels,
        patch_artist=True,
        widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    # Mean line (red) connecting mean values at each lambda
    mean_values = [np.mean(vals) for vals in all_values]
    x_positions = list(range(1, len(mean_values) + 1))
    if mean_values:
        ax.plot(x_positions, mean_values, marker="x", color="red", linestyle="--",
                linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label="Mean", zorder=10)
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND)

    # x-tick labels: every XTICK_LABEL_EVERY-th to avoid overlap
    xticklabels_display = [lambda_labels[i] if i % XTICK_LABEL_EVERY == 0 else "" for i in range(len(lambda_labels))]
    ax.set_xticklabels(xticklabels_display)
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis="y")
    ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel(r"$C_{\mathrm{C}}$", fontsize=FONTSIZE_YLABEL)
    ax.set_title(f"{dataset_display_name(dataset)} ($\\varepsilon={epsilon}$)", fontsize=FONTSIZE_TITLE, fontweight="bold")

    apply_ax_y_power10_mantissa_ticks(ax)


def plot_single_boxplot_CS(
    ax,
    results_dir: str,
    dataset: str,
    epsilon: int,
    show_ylabel: bool = True,
    show_legend: bool = False,
    intercept_candidate_num: Optional[int] = None,
):
    """Boxplot of C_S = covered_keys_after_attack vs lambda for minimize_segment_length_swing."""
    data = load_segment_length_results(
        results_dir, "minimize_segment_length_swing", dataset, epsilon, seed=None,
        intercept_candidate_num=intercept_candidate_num,
        allow_intercept_fallback=(intercept_candidate_num is None),
    )
    by_lambda: Dict[int, List[float]] = {}
    for d in data.to_dict("records"):
        lam = d["lambda"]
        if lam not in by_lambda:
            by_lambda[lam] = []
        by_lambda[lam].append(d["covered_keys_after_attack"])

    if not by_lambda:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return

    lambdas_sorted = sorted(by_lambda.keys())
    all_values = [by_lambda[lam] for lam in lambdas_sorted]
    lambda_labels = [str(lam) for lam in lambdas_sorted]

    bp = ax.boxplot(
        all_values,
        labels=lambda_labels,
        patch_artist=True,
        widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    # Mean line (red) connecting mean values at each lambda
    mean_values = [np.mean(vals) for vals in all_values]
    x_positions = list(range(1, len(mean_values) + 1))
    if mean_values:
        ax.plot(x_positions, mean_values, marker="x", color="red", linestyle="--",
                linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label="Mean", zorder=10)
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND)

    # x-tick labels: every XTICK_LABEL_EVERY-th to avoid overlap
    xticklabels_display = [lambda_labels[i] if i % XTICK_LABEL_EVERY == 0 else "" for i in range(len(lambda_labels))]
    ax.set_xticklabels(xticklabels_display)
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis="y")
    ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel(r"$C_{\mathrm{S}}$", fontsize=FONTSIZE_YLABEL)
    ax.set_title(f"{dataset_display_name(dataset)} ($\\varepsilon={epsilon}$)", fontsize=FONTSIZE_TITLE, fontweight="bold")

    apply_ax_y_power10_mantissa_ticks(ax)


def _group_generation_time_by_lambda(data: pd.DataFrame) -> Dict[int, List[float]]:
    """Group generation_time_sec by lambda, skip None."""
    by_lambda: Dict[int, List[float]] = {}
    for d in data.to_dict("records"):
        gt = d.get("generation_time_sec")
        if gt is None:
            continue
        lam = d["lambda"]
        if lam not in by_lambda:
            by_lambda[lam] = []
        by_lambda[lam].append(gt)
    return by_lambda


def plot_single_generation_time_mean_std(
    ax,
    results_dir: str,
    dataset: str,
    epsilon: int,
    show_ylabel: bool = True,
    show_legend: bool = False,
    intercept_candidate_num: Optional[int] = None,
):
    data_cc = load_segment_length_results(
        results_dir, "minimize_segment_length", dataset, epsilon, seed=None,
        intercept_candidate_num=None,
    )
    data_cs = load_segment_length_results(
        results_dir, "minimize_segment_length_swing", dataset, epsilon, seed=None,
        intercept_candidate_num=intercept_candidate_num,
        allow_intercept_fallback=(intercept_candidate_num is None),
    )
    by_lambda_cc = _group_generation_time_by_lambda(data_cc)
    by_lambda_cs = _group_generation_time_by_lambda(data_cs)

    if not by_lambda_cc and not by_lambda_cs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return

    all_lambdas = sorted(set(by_lambda_cc.keys()) | set(by_lambda_cs.keys()))
    lambda_labels = [str(lam) for lam in all_lambdas]
    x_positions = list(range(1, len(all_lambdas) + 1))

    # C_C: mean with 5%-95% percentile
    if by_lambda_cc:
        means_cc = [np.mean(by_lambda_cc.get(lam, [0])) for lam in all_lambdas]
        p5_cc = [np.percentile(v, 5) if (v := by_lambda_cc.get(lam, [])) else np.nan for lam in all_lambdas]
        p95_cc = [np.percentile(v, 95) if (v := by_lambda_cc.get(lam, [])) else np.nan for lam in all_lambdas]
        valid_cc = [i for i, lam in enumerate(all_lambdas) if lam in by_lambda_cc]
        if valid_cc:
            x_cc = [x_positions[i] for i in valid_cc]
            m_cc = [means_cc[i] for i in valid_cc]
            lo_cc = [p5_cc[i] for i in valid_cc]
            hi_cc = [p95_cc[i] for i in valid_cc]
            ax.plot(x_cc, m_cc, marker="o", color="C0", linestyle="-", linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label=CONSEC_LABEL_NAME, zorder=10)
            ax.fill_between(x_cc, np.array(lo_cc), np.array(hi_cc), color="C0", alpha=0.2)

    # C_S: mean with 5%-95% percentile
    if by_lambda_cs:
        means_cs = [np.mean(by_lambda_cs.get(lam, [0])) for lam in all_lambdas]
        p5_cs = [np.percentile(v, 5) if (v := by_lambda_cs.get(lam, [])) else np.nan for lam in all_lambdas]
        p95_cs = [np.percentile(v, 95) if (v := by_lambda_cs.get(lam, [])) else np.nan for lam in all_lambdas]
        valid_cs = [i for i, lam in enumerate(all_lambdas) if lam in by_lambda_cs]
        if valid_cs:
            x_cs = [x_positions[i] for i in valid_cs]
            m_cs = [means_cs[i] for i in valid_cs]
            lo_cs = [p5_cs[i] for i in valid_cs]
            hi_cs = [p95_cs[i] for i in valid_cs]
            ax.plot(x_cs, m_cs, marker="s", color="C1", linestyle="-", linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label=SWING_CONSEC_LABEL_NAME, zorder=10)
            ax.fill_between(x_cs, np.array(lo_cs), np.array(hi_cs), color="C1", alpha=0.2)

    if show_legend:
        ax.legend(fontsize=FONTSIZE_LEGEND)

    ax.set_xticks(x_positions)
    xticklabels_display = [lambda_labels[i] if i % XTICK_LABEL_EVERY == 0 else "" for i in range(len(lambda_labels))]
    ax.set_xticklabels(xticklabels_display)
    ax.tick_params(labelsize=FONTSIZE_TICKS, axis="x")
    ax.tick_params(labelsize=FONTSIZE_TICKS - 2, axis="y")

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis="y")
    ax.set_yscale("log")
    ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel("Time [s]", fontsize=FONTSIZE_YLABEL)
    ax.set_title(f"{dataset_display_name(dataset)} ($\\varepsilon={epsilon}$)", fontsize=FONTSIZE_TITLE, fontweight="bold")


def plot_single_boxplot_CC_minus_CS(
    ax,
    results_dir: str,
    dataset: str,
    epsilon: int,
    show_ylabel: bool = True,
    show_legend: bool = False,
    intercept_candidate_num: Optional[int] = None,
) -> Optional[float]:
    """Boxplot of (C_S - C_C) vs lambda; only (lambda, seed) where both exist. C_C/C_S = covered_keys_after_attack. Returns mean of (C_S - C_C) or None if no data."""
    data_cc = load_segment_length_results(
        results_dir, "minimize_segment_length", dataset, epsilon, seed=None,
        intercept_candidate_num=None,
    )
    data_cs = load_segment_length_results(
        results_dir, "minimize_segment_length_swing", dataset, epsilon, seed=None,
        intercept_candidate_num=intercept_candidate_num,
        allow_intercept_fallback=(intercept_candidate_num is None),
    )

    # Build (lambda, seed) -> C_C or C_S (= covered_keys_after_attack)
    cc_map = {}
    for d in data_cc.to_dict("records"):
        cc_map[(d["lambda"], d["seed"])] = d["covered_keys_after_attack"]
    cs_map = {}
    for d in data_cs.to_dict("records"):
        cs_map[(d["lambda"], d["seed"])] = d["covered_keys_after_attack"]

    # Common (lambda, seed) pairs
    common_keys = set(cc_map.keys()) & set(cs_map.keys())
    by_lambda: Dict[int, List[float]] = {}
    for (lam, se) in common_keys:
        diff = cs_map[(lam, se)] - cc_map[(lam, se)]
        if lam not in by_lambda:
            by_lambda[lam] = []
        by_lambda[lam].append(diff)

    if not by_lambda:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return None

    lambdas_sorted = sorted(by_lambda.keys())
    all_values = [by_lambda[lam] for lam in lambdas_sorted]
    lambda_labels = [str(lam) for lam in lambdas_sorted]

    bp = ax.boxplot(
        all_values,
        labels=lambda_labels,
        patch_artist=True,
        widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    # Mean line (red) connecting mean values at each lambda
    mean_values = [np.mean(vals) for vals in all_values]
    x_positions = list(range(1, len(mean_values) + 1))
    if mean_values:
        ax.plot(x_positions, mean_values, marker="x", color="red", linestyle="--", linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label="Mean", zorder=10)
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND)

    # x-tick labels: every XTICK_LABEL_EVERY-th to avoid overlap
    xticklabels_display = [lambda_labels[i] if i % XTICK_LABEL_EVERY == 0 else "" for i in range(len(lambda_labels))]
    ax.set_xticklabels(xticklabels_display)
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis="y")
    ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel(r"$C_{\mathrm{S}} - C_{\mathrm{C}}$", fontsize=FONTSIZE_YLABEL)
    ax.set_title(f"{dataset_display_name(dataset)} ($\\varepsilon={epsilon}$)", fontsize=FONTSIZE_TITLE, fontweight="bold")

    apply_ax_y_power10_mantissa_ticks(ax)

    all_diffs = [d for vals in by_lambda.values() for d in vals]
    mean_diff = float(np.mean(all_diffs))

    # Count of C_S - C_C > 0, = 0, < 0
    count_gt = sum(1 for d in all_diffs if d > 0)
    count_eq = sum(1 for d in all_diffs if d == 0)
    count_lt = sum(1 for d in all_diffs if d < 0)
    total = len(all_diffs)
    print(f"  [C_S - C_C] {dataset_display_name(dataset)}, ε={epsilon}: "
          f">0: {count_gt}, =0: {count_eq}, <0: {count_lt} (total: {total})")

    # Print when SWING is better (C_S < C_C, i.e. diff < 0)
    for lam in lambdas_sorted:
        vals = by_lambda[lam]
        better_count = sum(1 for v in vals if v < 0)
        if better_count > 0:
            better_vals = [v for v in vals if v < 0]
            mean_better = np.mean(better_vals)
            print(f"  [SWING better] {dataset_display_name(dataset)}, ε={epsilon}, λ={lam}: "
                  f"{better_count}/{len(vals)} seeds (mean diff={mean_better:.4f})")

    return mean_diff


def plot_single_boxplot_CC_minus_CS_over_CL(
    ax,
    results_dir: str,
    dataset: str,
    epsilon: int,
    show_ylabel: bool = True,
    show_legend: bool = False,
    intercept_candidate_num: Optional[int] = None,
) -> Optional[float]:
    """Boxplot of (C_S - C_C) / C_L vs lambda. C_L = covered_keys at lambda=0. Only (lambda, seed) where both exist and C_L > 0."""
    data_cc = load_segment_length_results(
        results_dir, "minimize_segment_length", dataset, epsilon, seed=None,
        intercept_candidate_num=None,
    )
    data_cs = load_segment_length_results(
        results_dir, "minimize_segment_length_swing", dataset, epsilon, seed=None,
        intercept_candidate_num=intercept_candidate_num,
        allow_intercept_fallback=(intercept_candidate_num is None),
    )

    # C_L: seed -> covered_keys at lambda=0
    cl_map: Dict[int, float] = {}
    for d in data_cc.to_dict("records"):
        if d["lambda"] == 0 and d.get("seed") is not None:
            cl_map[d["seed"]] = d["covered_keys_after_attack"]
    for d in data_cs.to_dict("records"):
        if d["lambda"] == 0 and d.get("seed") is not None:
            cl_map[d["seed"]] = d["covered_keys_after_attack"]

    cc_map = {(d["lambda"], d["seed"]): d["covered_keys_after_attack"] for d in data_cc.to_dict("records")}
    cs_map = {(d["lambda"], d["seed"]): d["covered_keys_after_attack"] for d in data_cs.to_dict("records")}

    common_keys = set(cc_map.keys()) & set(cs_map.keys())
    by_lambda: Dict[int, List[float]] = {}
    for (lam, se) in common_keys:
        if lam == 0:
            continue
        cl = cl_map.get(se)
        if cl is None or cl <= 0:
            continue
        diff = cs_map[(lam, se)] - cc_map[(lam, se)]
        ratio = diff / cl
        if lam not in by_lambda:
            by_lambda[lam] = []
        by_lambda[lam].append(ratio)

    if not by_lambda:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return None

    lambdas_sorted = sorted(by_lambda.keys())
    all_values = [by_lambda[lam] for lam in lambdas_sorted]
    lambda_labels = [str(lam) for lam in lambdas_sorted]

    bp = ax.boxplot(
        all_values,
        labels=lambda_labels,
        patch_artist=True,
        widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    mean_values = [np.mean(vals) for vals in all_values]
    x_positions = list(range(1, len(mean_values) + 1))
    if mean_values:
        ax.plot(x_positions, mean_values, marker="x", color="red", linestyle="--", linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label="Mean", zorder=10)
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND)

    xticklabels_display = [lambda_labels[i] if i % XTICK_LABEL_EVERY == 0 else "" for i in range(len(lambda_labels))]
    ax.set_xticklabels(xticklabels_display)
    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")

    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis="y")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel(XLABEL_LAMBDA, fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel(r"$(C_{\mathrm{S}} - C_{\mathrm{C}}) / C_{\mathrm{L}}$", fontsize=FONTSIZE_YLABEL)
    ax.set_title(f"{dataset_display_name(dataset)} ($\\varepsilon={epsilon}$)", fontsize=FONTSIZE_TITLE, fontweight="bold")

    apply_ax_y_power10_mantissa_ticks(ax)

    all_ratios = [r for vals in by_lambda.values() for r in vals]
    return float(np.mean(all_ratios))


def plot_CC_minus_CS_over_CL_all_epsilons(
    results_dir: str, datasets: List[str], output_dir: Path,
    intercept_candidate_num: Optional[int] = None,
    epsilons: Optional[List[int]] = None,
):
    """(C_S - C_C) / C_L: epsilon=16,32,64 vertically in one plot. 3 columns x ceil(n/3) rows (per epsilon) x len(eps)."""
    eps_list = epsilons if epsilons is not None else EPSILONS_CC_MINUS_CS
    num_cols = GRID_NUM_COLS
    num_rows_per_eps = subplot_grid_rows(len(datasets), num_cols)
    num_rows = len(eps_list) * num_rows_per_eps
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
    )
    axes = np.atleast_2d(axes)
    means_per_dataset = []
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            eps_idx = row_idx // num_rows_per_eps
            ds_idx = (row_idx % num_rows_per_eps) * num_cols + col_idx
            if ds_idx < len(datasets):
                epsilon = eps_list[eps_idx]
                dataset = datasets[ds_idx]
                show_ylabel = (col_idx == 0)
                show_legend = (row_idx == num_rows - 1 and ds_idx == len(datasets) - 1)
                mean_val = plot_single_boxplot_CC_minus_CS_over_CL(
                    axes[row_idx, col_idx], results_dir, dataset, epsilon,
                    show_ylabel=show_ylabel, show_legend=show_legend,
                    intercept_candidate_num=intercept_candidate_num,
                )
                means_per_dataset.append((dataset, epsilon, mean_val))
            else:
                axes[row_idx, col_idx].set_visible(False)
    plt.subplots_adjust(wspace=0.10, hspace=0.15)
    plt.tight_layout()
    out_file = output_dir / "lambda_to_segment_length_C_S_minus_C_C_over_C_L.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}", file=sys.stderr)
    for dataset, epsilon, mean_val in means_per_dataset:
        if mean_val is not None:
            print(f"  (C_S - C_C) / C_L mean ({dataset_display_name(dataset)}, ε={epsilon}): {mean_val:.6f}")


def plot_CC_all_epsilons(
    results_dir: str, datasets: List[str], output_dir: Path,
    intercept_candidate_num: Optional[int] = None,
    epsilons: Optional[List[int]] = None,
):
    """(1) C_C: epsilon=16,32,64 vertically in one plot. 3 columns x ceil(n/3) rows (per epsilon) x len(eps)."""
    eps_list = epsilons if epsilons is not None else EPSILONS_CC
    num_cols = GRID_NUM_COLS
    num_rows_per_eps = subplot_grid_rows(len(datasets), num_cols)
    num_rows = len(eps_list) * num_rows_per_eps
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
    )
    axes = np.atleast_2d(axes)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            eps_idx = row_idx // num_rows_per_eps
            ds_idx = (row_idx % num_rows_per_eps) * num_cols + col_idx
            if ds_idx < len(datasets):
                epsilon = eps_list[eps_idx]
                dataset = datasets[ds_idx]
                show_ylabel = (col_idx == 0)
                show_legend = (row_idx == num_rows - 1 and ds_idx == len(datasets) - 1)
                plot_single_boxplot_CC(
                    axes[row_idx, col_idx], results_dir, dataset, epsilon,
                    show_ylabel=show_ylabel, show_legend=show_legend,
                    intercept_candidate_num=intercept_candidate_num,
                )
            else:
                axes[row_idx, col_idx].set_visible(False)
    plt.subplots_adjust(wspace=0.10, hspace=0.15)
    plt.tight_layout()
    out_file = output_dir / "lambda_to_segment_length_C_C.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}", file=sys.stderr)


def plot_CS_all_epsilons(
    results_dir: str, datasets: List[str], output_dir: Path,
    intercept_candidate_num: Optional[int] = None,
    epsilons: Optional[List[int]] = None,
):
    """(2) C_S: epsilon=16,32,64,128 vertically in one plot. 3 columns x ceil(n/3) rows (per epsilon) x len(eps)."""
    eps_list = epsilons if epsilons is not None else EPSILONS_CS
    num_cols = GRID_NUM_COLS
    num_rows_per_eps = subplot_grid_rows(len(datasets), num_cols)
    num_rows = len(eps_list) * num_rows_per_eps
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
    )
    axes = np.atleast_2d(axes)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            eps_idx = row_idx // num_rows_per_eps
            ds_idx = (row_idx % num_rows_per_eps) * num_cols + col_idx
            if ds_idx < len(datasets):
                epsilon = eps_list[eps_idx]
                dataset = datasets[ds_idx]
                show_ylabel = (col_idx == 0)
                show_legend = (row_idx == num_rows - 1 and ds_idx == len(datasets) - 1)
                plot_single_boxplot_CS(
                    axes[row_idx, col_idx], results_dir, dataset, epsilon,
                    show_ylabel=show_ylabel, show_legend=show_legend,
                    intercept_candidate_num=intercept_candidate_num,
                )
            else:
                axes[row_idx, col_idx].set_visible(False)
    plt.subplots_adjust(wspace=0.10, hspace=0.15)
    plt.tight_layout()
    out_file = output_dir / "lambda_to_segment_length_C_S.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}", file=sys.stderr)


def plot_CC_minus_CS_all_epsilons(
    results_dir: str, datasets: List[str], output_dir: Path,
    intercept_candidate_num: Optional[int] = None,
    epsilons: Optional[List[int]] = None,
):
    """(3) C_S - C_C: epsilon=16,32,64 vertically in one plot. 3 columns x ceil(n/3) rows (per epsilon) x len(eps)."""
    eps_list = epsilons if epsilons is not None else EPSILONS_CC_MINUS_CS
    num_cols = GRID_NUM_COLS
    num_rows_per_eps = subplot_grid_rows(len(datasets), num_cols)
    num_rows = len(eps_list) * num_rows_per_eps
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
    )
    axes = np.atleast_2d(axes)
    means_per_dataset = []
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            eps_idx = row_idx // num_rows_per_eps
            ds_idx = (row_idx % num_rows_per_eps) * num_cols + col_idx
            if ds_idx < len(datasets):
                epsilon = eps_list[eps_idx]
                dataset = datasets[ds_idx]
                show_ylabel = (col_idx == 0)
                show_legend = (row_idx == num_rows - 1 and ds_idx == len(datasets) - 1)
                mean_val = plot_single_boxplot_CC_minus_CS(
                    axes[row_idx, col_idx], results_dir, dataset, epsilon,
                    show_ylabel=show_ylabel, show_legend=show_legend,
                    intercept_candidate_num=intercept_candidate_num,
                )
                means_per_dataset.append((dataset, epsilon, mean_val))
            else:
                axes[row_idx, col_idx].set_visible(False)
    plt.subplots_adjust(wspace=0.10, hspace=0.15)
    plt.tight_layout()
    out_file = output_dir / "lambda_to_segment_length_C_S_C_C.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}", file=sys.stderr)
    for dataset, epsilon, mean_val in means_per_dataset:
        if mean_val is not None:
            print(f"  C_S - C_C mean ({dataset_display_name(dataset)}, ε={epsilon}): {mean_val:.4f}")


def plot_single_epsilon_to_generation_time(
    ax,
    results_dir: str,
    dataset: str,
    epsilons: List[int],
    show_ylabel: bool = True,
    show_legend: bool = False,
    show_xlabel: bool = True,
    intercept_candidate_num: Optional[int] = None,
):
    by_eps_cc: Dict[int, List[float]] = {}
    by_eps_cs: Dict[int, List[float]] = {}
    for eps in epsilons:
        lam = eps  # λ = ε
        data_cc = load_segment_length_results(
            results_dir, "minimize_segment_length", dataset, eps, seed=None,
            intercept_candidate_num=None,
        )
        data_cs = load_segment_length_results(
            results_dir, "minimize_segment_length_swing", dataset, eps, seed=None,
            intercept_candidate_num=intercept_candidate_num,
            allow_intercept_fallback=(intercept_candidate_num is None),
        )
        vals_cc = [d["generation_time_sec"] for d in data_cc.to_dict("records") if d.get("generation_time_sec") is not None and d.get("lambda") == lam]
        vals_cs = [d["generation_time_sec"] for d in data_cs.to_dict("records") if d.get("generation_time_sec") is not None and d.get("lambda") == lam]
        if vals_cc:
            by_eps_cc[eps] = vals_cc
        if vals_cs:
            by_eps_cs[eps] = vals_cs

    if not by_eps_cc and not by_eps_cs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(r"$\varepsilon$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        return

    x_positions = list(range(1, len(epsilons) + 1))
    epsilon_labels = [str(eps) for eps in epsilons]

    # C_C: mean with 5%-95% percentile
    if by_eps_cc:
        means_cc = [np.mean(by_eps_cc.get(eps, [0])) for eps in epsilons]
        p5_cc = [np.percentile(v, 5) if (v := by_eps_cc.get(eps, [])) else np.nan for eps in epsilons]
        p95_cc = [np.percentile(v, 95) if (v := by_eps_cc.get(eps, [])) else np.nan for eps in epsilons]
        valid_cc = [i for i, eps in enumerate(epsilons) if eps in by_eps_cc]
        if valid_cc:
            x_cc = [x_positions[i] for i in valid_cc]
            m_cc = [means_cc[i] for i in valid_cc]
            lo_cc = [p5_cc[i] for i in valid_cc]
            hi_cc = [p95_cc[i] for i in valid_cc]
            ax.plot(x_cc, m_cc, marker="o", color="C0", linestyle="-", linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label=CONSEC_LABEL_NAME, zorder=10)
            ax.fill_between(x_cc, np.array(lo_cc), np.array(hi_cc), color="C0", alpha=0.2)

    # C_S: mean with 5%-95% percentile
    if by_eps_cs:
        means_cs = [np.mean(by_eps_cs.get(eps, [0])) for eps in epsilons]
        p5_cs = [np.percentile(v, 5) if (v := by_eps_cs.get(eps, [])) else np.nan for eps in epsilons]
        p95_cs = [np.percentile(v, 95) if (v := by_eps_cs.get(eps, [])) else np.nan for eps in epsilons]
        valid_cs = [i for i, eps in enumerate(epsilons) if eps in by_eps_cs]
        if valid_cs:
            x_cs = [x_positions[i] for i in valid_cs]
            m_cs = [means_cs[i] for i in valid_cs]
            lo_cs = [p5_cs[i] for i in valid_cs]
            hi_cs = [p95_cs[i] for i in valid_cs]
            ax.plot(x_cs, m_cs, marker="s", color="C1", linestyle="-", linewidth=PLOT_LINEWIDTH_L, markersize=PLOT_MARKERSIZE, label=SWING_CONSEC_LABEL_NAME, zorder=10)
            ax.fill_between(x_cs, np.array(lo_cs), np.array(hi_cs), color="C1", alpha=0.2)

    if show_legend:
        ax.legend(fontsize=FONTSIZE_LEGEND)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(epsilon_labels)
    ax.tick_params(labelsize=FONTSIZE_TICKS, axis="x")
    ax.tick_params(labelsize=FONTSIZE_TICKS - 1, axis="y")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)

    if show_xlabel:
        ax.set_xlabel(r"$\varepsilon$", fontsize=FONTSIZE_XLABEL)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("Time [s]", fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")


def plot_epsilon_to_generation_time(
    results_dir: str, datasets: List[str], output_dir: Path,
    intercept_candidate_num: Optional[int] = None,
    epsilons: Optional[List[int]] = None,
):
    eps_list = epsilons if epsilons is not None else [16, 32, 64, 128]
    num_cols = GRID_NUM_COLS
    num_rows = subplot_grid_rows(len(datasets), num_cols)
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.8),
    )
    axes = np.atleast_2d(axes)
    for i, dataset in enumerate(datasets):
        row, col = i // num_cols, i % num_cols
        show_ylabel = (col == 0)
        show_xlabel = row == num_rows - 1
        plot_single_epsilon_to_generation_time(
            axes[row, col], results_dir, dataset, eps_list,
            show_ylabel=show_ylabel,
            show_legend=False,
            show_xlabel=show_xlabel,
            intercept_candidate_num=intercept_candidate_num,
        )
    for i in range(len(datasets), num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].set_visible(False)
    plt.subplots_adjust(wspace=0.10, hspace=0.15)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    add_figure_legend_top_horizontal(fig, axes, len(datasets), num_cols, fontsize=FONTSIZE_LEGEND)
    out_file = output_dir / "epsilon_to_generation_time_sec.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}", file=sys.stderr)


def plot_single_generation_time_vs_segment_length_scatter(
    ax,
    results_dir: str,
    dataset: str,
    epsilons: List[int],
    show_ylabel: bool = True,
    show_legend: bool = False,
    intercept_candidate_num: Optional[int] = None,
):
    seg_len_cc: List[float] = []
    time_cc: List[float] = []
    seg_len_cs: List[float] = []
    time_cs: List[float] = []

    for eps in epsilons:
        data_cc = load_segment_length_results(
            results_dir, "minimize_segment_length", dataset, eps, seed=None,
            intercept_candidate_num=None,
        )
        data_cs = load_segment_length_results(
            results_dir, "minimize_segment_length_swing", dataset, eps, seed=None,
            intercept_candidate_num=intercept_candidate_num,
            allow_intercept_fallback=(intercept_candidate_num is None),
        )
        for d in data_cc.to_dict("records"):
            seg = d.get("covered_keys_before_attack")
            gt = d.get("generation_time_sec")
            if seg is not None and gt is not None:
                seg_len_cc.append(float(seg))
                time_cc.append(gt)
        for d in data_cs.to_dict("records"):
            seg = d.get("covered_keys_before_attack")
            gt = d.get("generation_time_sec")
            if seg is not None and gt is not None:
                seg_len_cs.append(float(seg))
                time_cs.append(gt)

    if not seg_len_cc and not seg_len_cs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")
        return

    if seg_len_cc:
        ax.scatter(seg_len_cc, time_cc, marker="o", color="C0", s=30, alpha=0.6, label=CONSEC_LABEL_NAME, zorder=5)
    if seg_len_cs:
        ax.scatter(seg_len_cs, time_cs, marker="s", color="C1", s=30, alpha=0.6, label=SWING_CONSEC_LABEL_NAME, zorder=5)

    if show_legend:
        ax.legend(fontsize=FONTSIZE_LEGEND)

    ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
    ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR, axis="y")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$C_{\mathrm{L}}$", fontsize=FONTSIZE_XLABEL)
    if show_ylabel:
        ax.set_ylabel("Time [s]", fontsize=FONTSIZE_YLABEL)
    ax.set_title(dataset_display_name(dataset), fontsize=FONTSIZE_TITLE, fontweight="bold")


def plot_generation_time_vs_segment_length_scatter(
    results_dir: str, datasets: List[str], output_dir: Path,
    intercept_candidate_num: Optional[int] = None,
    epsilons: Optional[List[int]] = None,
):
    """Scatter: generation_time_sec vs original segment length. 3 columns x ceil(n/3) rows."""
    eps_list = epsilons if epsilons is not None else [16, 32, 64, 128]
    num_cols = GRID_NUM_COLS
    num_rows = subplot_grid_rows(len(datasets), num_cols)
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
    )
    axes = np.atleast_2d(axes)
    for i, dataset in enumerate(datasets):
        row, col = i // num_cols, i % num_cols
        show_ylabel = (col == 0)
        show_legend = (i == 0)
        plot_single_generation_time_vs_segment_length_scatter(
            axes[row, col], results_dir, dataset, eps_list,
            show_ylabel=show_ylabel, show_legend=show_legend,
            intercept_candidate_num=intercept_candidate_num,
        )
    for i in range(len(datasets), num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].set_visible(False)
    plt.subplots_adjust(wspace=0.10, hspace=0.15)
    plt.tight_layout()
    out_file = output_dir / "generation_time_vs_original_segment_length.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}", file=sys.stderr)


def plot_generation_time_all_epsilons(
    results_dir: str, datasets: List[str], output_dir: Path,
    intercept_candidate_num: Optional[int] = None,
    epsilons: Optional[List[int]] = None,
):
    eps_list = epsilons if epsilons is not None else [16, 32]
    total_cells = len(eps_list) * len(datasets)
    num_cols = GRID_NUM_COLS
    num_rows = (total_cells + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows * 0.9),
    )
    axes = np.atleast_2d(axes)
    for idx in range(total_cells):
        row, col = idx // num_cols, idx % num_cols
        eps_idx = idx // len(datasets)
        ds_idx = idx % len(datasets)
        epsilon = eps_list[eps_idx]
        dataset = datasets[ds_idx]
        show_ylabel = (col == 0)
        show_legend = (idx == 0)
        plot_single_generation_time_mean_std(
            axes[row, col], results_dir, dataset, epsilon,
            show_ylabel=show_ylabel, show_legend=show_legend,
            intercept_candidate_num=intercept_candidate_num,
        )
    for idx in range(total_cells, num_rows * num_cols):
        row, col = idx // num_cols, idx % num_cols
        axes[row, col].set_visible(False)
    plt.subplots_adjust(wspace=0.10, hspace=0.15)
    plt.tight_layout()
    out_file = output_dir / "lambda_to_generation_time_sec.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot segment length vs lambda (PGM poisoning experiments)."
    )
    parser.add_argument(
        "--duplicate-only",
        action="store_true",
        help=(
            "Only duplicate-key datasets (bench2/bench3/wiki_ts_200M/zipf); "
            "write under fig/lambda_to_segment_length_dup/."
        ),
    )
    args = parser.parse_args()

    results_dir = "results/poisoning_segment_length"
    output_base_dir = (
        "fig/lambda_to_segment_length_dup"
        if args.duplicate_only
        else "fig/lambda_to_segment_length"
    )

    datasets = list(
        DATASET_IDS_800M_DUPLICATES if args.duplicate_only else DATASET_IDS_800M_SWING
    )

    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    # Produce plots for dataset groups. Non-1M (bench2/bench3/800M/200M/... etc) and 1M groups.
    out_dir = output_base_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # # (1) C_C: minimize_segment_length
    # plot_CC_all_epsilons(results_dir, datasets, out_dir)

    # # (2) C_S: minimize_segment_length_swing
    # plot_CS_all_epsilons(results_dir, datasets, out_dir)

    # # (3) C_S - C_C
    # plot_CC_minus_CS_all_epsilons(results_dir, datasets, out_dir)

    # (3b) (C_S - C_C) / C_L
    plot_CC_minus_CS_over_CL_all_epsilons(results_dir, datasets, out_dir)

    # # (4) generation_time_sec: several epsilons horizontal layout
    # plot_generation_time_all_epsilons(results_dir, datasets, out_dir)

    # (5) epsilon vs generation_time_sec
    plot_epsilon_to_generation_time(results_dir, datasets, out_dir)

    # # (6) generation_time vs original segment length (scatter)
    # plot_generation_time_vs_segment_length_scatter(results_dir, datasets, out_dir)

    # (7) lambda to C mean (like lambda_to_max_error_mean_n100): C_C, C_S, C_R, C_A, C_AI + C_L
    plot_lambda_to_C_mean(results_dir, datasets, out_dir)
