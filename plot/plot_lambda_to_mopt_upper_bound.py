"""
Plot lambda vs m_opt: upper bound vs swing_lambda_with_theta (seed=0 only).

Reads from:
  - results/upper_bound/fix_w_per_block/{dataset}/.../lambda{N}/epsilon{E}/upper_bound.json
  - Swing: results/poisoning/.../swing_lambda_with_theta/{dataset}/lambda{...}/... (full, no seed dir),
    or .../{dataset}/seed0/... (1M).

Outputs:
  - fig/lambda_to_mopt_upper_bound/lambda_to_mopt_upper_bound_vs_swing_full.pdf
  - fig/lambda_to_mopt_upper_bound/lambda_to_mopt_upper_bound_vs_swing_1M.pdf
  With --duplicate-only: *_dup.pdf under fig/lambda_to_mopt_upper_bound_dup/.

LaTeX tabular: use plot/print_table_mopt_upper_bound.py.
"""
import argparse
import sys

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from load_results_mopt import load_swing_lambda_benchmark_results
from load_results_upper_bound import load_upper_bound_results
from plot_config import (
    dataset_display_name,
    upper_bound_dataset_specs,
    upper_bound_dataset_specs_duplicate_only,
    upper_bound_dataset_specs_full,
    upper_bound_dataset_specs_full_duplicate_only,
)
from add_figure_legend import add_figure_legend_top_horizontal

# Figure settings
FIG_WIDTH_PER_COL = 4.0
FIG_HEIGHT_PER_ROW = 5.0
GRID_NUM_COLS = 3

# Plot settings
PLOT_MARKERSIZE = 8
PLOT_LINEWIDTH = 2.5
# (color, marker, linestyle) per epsilon
EPSILON_STYLES = [
    ("#1f77b4", "o", "-"),   # blue, circle, solid
    ("#ff7f0e", "s", "-"),  # orange, square
    ("#2ca02c", "^", "-"),  # green, triangle
    ("#d62728", "x", "-"),  # red, x
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


def plot_single_dataset_upper_bound_vs_swing(
    ax,
    ub_results_by_epsilon: Dict[int, List[Dict]],
    swing_results_by_epsilon: Dict[int, List[Dict]],
    dataset_name: str,
    epsilons: List[int],
    show_legend: bool = False,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    """
    Plot upper bound and swing m_opt for a single dataset (seed=0).
    For each epsilon: upper_bound (solid) and swing m_opt (dashed).
    Log-log scale; lambda=0 is shown as horizontal dashed line.
    """
    has_data = False
    for i, eps in enumerate(epsilons):
        ub_results = ub_results_by_epsilon.get(eps, [])
        swing_results = swing_results_by_epsilon.get(eps, [])

        style = EPSILON_STYLES[i % len(EPSILON_STYLES)]
        color, marker, _ = style

        # Lambda=0: draw as horizontal line (log scale cannot show x=0)
        val_at_0 = None
        for r in ub_results:
            if r.get("lambda") == 0:
                val_at_0 = r.get("m_opt_upper_bound")
                break
        if val_at_0 is None:
            for r in swing_results:
                if r.get("lambda") == 0:
                    val_at_0 = r.get("first_level_segments_num") or r.get("m_opt_after_poisoning")
                    break

        # Upper bound (exclude lambda=0 for log scale)
        ub_lambdas = []
        ub_values = []
        for r in ub_results:
            v = r.get("m_opt_upper_bound")
            lam = r.get("lambda")
            if v is not None and lam is not None and lam > 0:
                ub_lambdas.append(lam)
                ub_values.append(v)
        if ub_values:
            has_data = True
            ax.plot(
                ub_lambdas,
                ub_values,
                marker=marker,
                linestyle="-",
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH,
                color=color,
                label=rf"$m_{{opt}}$ upper bound ($\varepsilon$={eps})",
            )

        # Swing m_opt (exclude lambda=0 for log scale)
        swing_lambdas = []
        swing_values = []
        for r in swing_results:
            v = r.get("first_level_segments_num")
            if v is None:
                v = r.get("m_opt_after_poisoning")
            lam = r.get("lambda")
            if v is not None and lam is not None and lam > 0:
                swing_lambdas.append(lam)
                swing_values.append(v)
        if swing_values:
            has_data = True
            ax.plot(
                swing_lambdas,
                swing_values,
                marker=marker,
                linestyle="--",
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH,
                color=color,
                label=rf"$m_{{opt}}$ after poisoning ($\varepsilon$={eps})",
            )

        if val_at_0 is not None and val_at_0 > 0:
            has_data = True
            ax.axhline(
                y=val_at_0,
                linestyle=":",
                color=color,
                linewidth=PLOT_LINEWIDTH * 2,
                zorder=0,
                label=rf"$m_{{opt}}$ before poisoning ($\varepsilon$={eps})",
            )

    if has_data:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
        if show_xlabel:
            ax.set_xlabel(r"$\lambda$", fontsize=FONTSIZE_XLABEL)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(r"$m_{\mathrm{opt}}$", fontsize=FONTSIZE_YLABEL)
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
            ax.set_ylabel(r"$m_{\mathrm{opt}}$", fontsize=FONTSIZE_YLABEL)
        else:
            ax.set_ylabel("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot upper bound vs swing_lambda_with_theta m_opt."
    )
    parser.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Only duplicate-key datasets; write *_dup.pdf under fig/lambda_to_mopt_upper_bound_dup/.",
    )
    args = parser.parse_args()

    seed = 0
    mu = 100
    # epsilons = [16, 32, 64, 128]
    epsilons = [16, 32, 64, 128]
    ub_base_dir = "results/upper_bound/fix_w_per_block"
    swing_base_dir = "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
    output_base_dir = (
        "fig/lambda_to_mopt_upper_bound_dup"
        if args.duplicate_only
        else "fig/lambda_to_mopt_upper_bound"
    )

    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    def _plot_upper_bound_grid(
        datasets: List[Dict[str, str]],
        num_cols: int,
        swing_path_for_dataset,
        output_name: str,
    ) -> None:
        num_rows = (len(datasets) + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(FIG_WIDTH_PER_COL * num_cols, FIG_HEIGHT_PER_ROW * num_rows),
            squeeze=False,
        )

        for i, dataset in enumerate(datasets):
            row, col = i // num_cols, i % num_cols
            ax = axes[row, col]

            ub_path = f"{ub_base_dir}/{dataset['path']}"
            swing_path = swing_path_for_dataset(dataset)

            ub_results_by_epsilon = {
                eps: load_upper_bound_results(ub_path, eps, seed).to_dict("records")
                for eps in epsilons
            }
            swing_results_by_epsilon = {
                eps: load_swing_lambda_benchmark_results(
                    swing_path, eps, mu, aggregate_1M=False
                ).to_dict("records")
                for eps in epsilons
            }

            plot_single_dataset_upper_bound_vs_swing(
                ax,
                ub_results_by_epsilon,
                swing_results_by_epsilon,
                dataset["name"],
                epsilons,
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
            pad=0.02,
            legend_ncol=2,
            fontsize=FONTSIZE_LEGEND,
        )
        output_file = output_base_path / output_name
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {output_file}", file=sys.stderr)

    if args.duplicate_only:
        datasets_full = upper_bound_dataset_specs_full_duplicate_only()
        datasets_1m = upper_bound_dataset_specs_duplicate_only()
        suffix = "_dup"
    else:
        datasets_full = upper_bound_dataset_specs_full()
        datasets_1m = upper_bound_dataset_specs()
        suffix = ""

    num_cols_full = GRID_NUM_COLS
    _plot_upper_bound_grid(
        datasets_full,
        num_cols_full,
        swing_path_for_dataset=lambda d: f"{swing_base_dir}/{d['path']}",
        output_name=f"lambda_to_mopt_upper_bound_vs_swing_full{suffix}.pdf",
    )

    num_cols_1m = GRID_NUM_COLS
    _plot_upper_bound_grid(
        datasets_1m,
        num_cols_1m,
        swing_path_for_dataset=lambda d: f"{swing_base_dir}/{d['path']}/seed{seed}",
        output_name=f"lambda_to_mopt_upper_bound_vs_swing_1M{suffix}.pdf",
    )
