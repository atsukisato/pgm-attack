"""
Plot lambda vs m_opt for inject_poisons_to_minimize_segment_length_swing_lambda_with_theta results.
Reads from results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta.
Writes PDFs under fig/lambda_to_mopt/ (or fig/lambda_to_mopt_dup/ with --duplicate-only).
"""
import argparse
import sys

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

from load_results_mopt import load_swing_lambda_benchmark_results
from plot_config import (
    dataset_display_name,
    swing_dataset_specs_1m,
    swing_dataset_specs_1m_duplicate,
    swing_dataset_specs_800m,
)
from add_figure_legend import add_figure_legend_top_horizontal

# Figure settings
FIG_WIDTH_PER_COL = 4.0
FIG_HEIGHT_PER_ROW = 3.5
GRID_NUM_COLS = 3

# Plot settings
PLOT_MARKERSIZE = 8
PLOT_LINEWIDTH = 2.5
# (color, marker, linestyle) per epsilon for multi-epsilon overlay
EPSILON_STYLES = [
    ("#1f77b4", "o", "-"),   # blue, circle
    ("#ff7f0e", "s", "--"),  # orange, square
    ("#2ca02c", "^", "-."),  # green, triangle
    ("#d62728", "x", ":"),   # red, x
]

# Grid settings
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 1.0
GRID_COLOR = "gray"

# Font sizes
FONTSIZE_XLABEL = 24
FONTSIZE_YLABEL = 24
FONTSIZE_TITLE = 24
FONTSIZE_TICKS = 20
FONTSIZE_TEXT = 24
FONTSIZE_LEGEND = 24
# Text settings
TEXT_COLOR = "gray"

def get_y_value(result: Dict, y_metric: str) -> Optional[float]:
    """Get metric value from result dictionary."""
    return result.get(y_metric)


def print_ratios_to_lambda0(
    results_by_epsilon: dict,
    y_metric: str,
    dataset_name: str,
    epsilons: List[int],
):
    """Print each (lambda, epsilon) value as ratio to lambda=0 value."""
    print(f"dataset: {dataset_name}")
    for eps in epsilons:
        print(f"  ε={eps}")
        results = results_by_epsilon.get(eps, [])
        val_at_0 = None
        n_at_0 = 0
        for r in results:
            if r.get("lambda") == 0:
                val_at_0 = get_y_value(r, y_metric)
                n_at_0 = r.get("n", 0)
                break
        if val_at_0 is None or val_at_0 == 0:
            continue
        for r in results:
            lam = r.get("lambda")
            val = get_y_value(r, y_metric)
            if val is not None and lam is not None:
                ratio = val / val_at_0
                n = r.get("n", 0)
                print(f"    λ={lam}: {val} ({ratio:.4f}x of λ=0) [mean over {n} cases | λ=0: mean over {n_at_0} cases]")


def get_y_label(y_metric: str) -> str:
    """Get Y-axis label for metric."""
    labels = {
        "build_time_sec": "Build Time (sec)",
        "index_size_in_kb": "Index Size (KB)",
        "levels": "Levels",
        "first_level_segments_num": r"$m_{\mathrm{opt}}$",
        "first_level_segments_num_ratio": r"$m_{\mathrm{opt}}(K \cup P) / m_{\mathrm{opt}}(K)$",
        "median_avg_query_time_ns": "Query Time (ns)",
        "num_poisons_generated": "Number of Poisons",
        "m_opt_after_poisoning": "m_opt After Poisoning",
        "generation_time_sec": "Generation Time (sec)",
    }
    return labels.get(y_metric, y_metric)


def plot_single_dataset_multi_epsilon(
    ax,
    results_by_epsilon: dict,
    y_metric: str,
    dataset_name: str,
    epsilons: List[int],
    show_legend: bool = True,
    use_log_log: bool = False,
    use_log_x_exclude_lambda0: bool = False,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    """Plot single dataset on subplot with multiple epsilons overlaid (different colors/markers)."""
    # For ratio metric: m_opt(K∪P) / m_opt(K), where m_opt(K) = value at lambda=0
    use_ratio = y_metric == "first_level_segments_num_ratio"
    base_metric = "first_level_segments_num" if use_ratio else y_metric

    # When use_log_log: draw lambda=0 as horizontal dashed lines (x-axis parallel)
    # For ratio metric: baseline is 1.0. For raw metric: baseline is val_at_0.
    # When use_log_x_exclude_lambda0: x-axis log only, exclude lambda=0 (no horizontal lines)
    if use_log_log:
        for i, eps in enumerate(epsilons):
            results = results_by_epsilon.get(eps, [])
            val_at_0 = None
            for result in results:
                if result.get("lambda") == 0:
                    val_at_0 = get_y_value(result, base_metric)
                    break
            baseline = 1.0 if use_ratio else val_at_0
            if baseline is not None and baseline > 0:
                style = EPSILON_STYLES[i % len(EPSILON_STYLES)]
                color, _, _ = style
                ax.axhline(y=baseline, linestyle="--", color=color, linewidth=3.0, zorder=0)

    has_data = False
    for i, eps in enumerate(epsilons):
        results = results_by_epsilon.get(eps, [])
        y_values = []
        lambdas = []
        val_at_0 = None
        for result in results:
            if result.get("lambda") == 0:
                val_at_0 = get_y_value(result, base_metric)
                break
        if use_ratio and (val_at_0 is None or val_at_0 == 0):
            continue
        for result in results:
            y_value = get_y_value(result, base_metric)
            if y_value is not None:
                if use_ratio:
                    y_value = y_value / val_at_0
                lam = result["lambda"]
                # For log scale: exclude lambda=0 (shown as horizontal dashed line instead)
                # For use_log_x_exclude_lambda0: exclude lambda=0 (no horizontal line)
                if (use_log_log or use_log_x_exclude_lambda0) and lam == 0:
                    continue
                # For log scale: y must be > 0
                if use_log_log and y_value <= 0:
                    continue
                lambdas.append(lam)
                y_values.append(y_value)

        if y_values:
            has_data = True
            style = EPSILON_STYLES[i % len(EPSILON_STYLES)]
            color, marker, linestyle = style
            lw = 2.0 if use_log_log else PLOT_LINEWIDTH
            ms = PLOT_MARKERSIZE * 1.25 if use_log_log else PLOT_MARKERSIZE
            ax.plot(
                lambdas,
                y_values,
                marker=marker,
                linestyle=linestyle,
                markersize=ms,
                linewidth=lw,
                color=color,
                label=rf"$\varepsilon$={eps}",
            )

    if has_data:
        if use_log_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif use_log_x_exclude_lambda0:
            ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
        if show_xlabel:
            ax.set_xlabel(r"$\lambda$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(get_y_label(y_metric), fontsize=FONTSIZE_YLABEL)
        else:
            ax.set_ylabel("")
        ax.set_title(dataset_display_name(dataset_name), fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONTSIZE_TICKS, which="both")
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONTSIZE_TEXT,
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_title(dataset_display_name(dataset_name), fontsize=FONTSIZE_TITLE, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(r"$\lambda$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(get_y_label(y_metric), fontsize=FONTSIZE_YLABEL)
        else:
            ax.set_ylabel("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot lambda vs m_opt (swing_lambda_with_theta results)."
    )
    parser.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Only duplicate-key 1M datasets; write under fig/lambda_to_mopt_dup/.",
    )
    args = parser.parse_args()

    base_dir = "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
    epsilons = [16, 32, 64, 128]
    mu = 100  # Default μ for each (lambda, epsilon); loader falls back if this μ dir is missing
    output_base_dir = "fig/lambda_to_mopt_dup" if args.duplicate_only else "fig/lambda_to_mopt"
    out_suffix = "_dup" if args.duplicate_only else ""

    metrics_benchmark = [
        "index_size_in_kb",
        "first_level_segments_num",
        "first_level_segments_num_ratio",
        "median_avg_query_time_ns"
    ]

    metrics_poisons = [
        "num_poisons_generated",
        "generation_time_sec"
    ]

    metrics = metrics_benchmark + metrics_poisons

    datasets_800M = swing_dataset_specs_800m(base_dir)

    datasets_1M = (
        swing_dataset_specs_1m_duplicate(base_dir)
        if args.duplicate_only
        else swing_dataset_specs_1m(base_dir)
    )

    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    for y_metric in metrics:
        # 1M datasets — same 3-column grid
        datasets = datasets_1M
        num_cols = GRID_NUM_COLS
        num_rows = max(1, (len(datasets) + num_cols - 1) // num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows), squeeze=False)

        if y_metric == "first_level_segments_num":
            print(f"\n[{y_metric}] λ=0 is how many times (1M):")
        use_log_log_1M = y_metric in ("first_level_segments_num", "first_level_segments_num_ratio")
        use_log_x_exclude_lambda0_1M = y_metric == "generation_time_sec"
        for i, dataset in enumerate(datasets):
            row, col = i // num_cols, i % num_cols
            ax = axes[row, col]
            results_by_epsilon = {
                eps: load_swing_lambda_benchmark_results(dataset["path"], eps, mu).to_dict("records") for eps in epsilons
            }
            if y_metric == "first_level_segments_num":
                print_ratios_to_lambda0(results_by_epsilon, y_metric, dataset["name"], epsilons)
            plot_single_dataset_multi_epsilon(
                ax,
                results_by_epsilon,
                y_metric,
                dataset["name"],
                epsilons,
                show_legend=False,
                use_log_log=use_log_log_1M,
                use_log_x_exclude_lambda0=use_log_x_exclude_lambda0_1M,
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
            legend_ncol=len(epsilons),
            fontsize=FONTSIZE_LEGEND,
        )
        output_file_1M = output_base_path / f"lambda_to_{y_metric}_1M{out_suffix}.pdf"
        plt.savefig(output_file_1M, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_file_1M}", file=sys.stderr)
