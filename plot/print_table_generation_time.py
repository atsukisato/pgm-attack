"""
LaTeX tables: Proposed-method poisoning generation time (hours, 1 decimal).

Same experiment layout as plot/print_table_mopt_upper_bound.py (stacked $\\varepsilon$):
  results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta
  → …/lambda{L}/epsilon{E}/mu{M}/poisons_info.json → generation_time_sec.

Outputs (full-scale × $\\lambda$ frac): fig/table_generation_time/generation_time_full_lambda{0p01,0p1}.tex
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from load_results_mopt import load_swing_lambda_benchmark_results
from plot_config import (
    LAMBDA_TABLE_SPEC_1M_DUPLICATES,
    LAMBDA_TABLE_SPEC_FULL_DUPLICATES,
    dataset_display_name,
    upper_bound_dataset_specs,
    upper_bound_dataset_specs_duplicate_only,
    upper_bound_dataset_specs_full,
    upper_bound_dataset_specs_full_duplicate_only,
)

from print_table_mopt_upper_bound import (
    LAMBDA_TABLE_SPEC_1M,
    LAMBDA_TABLE_SPEC_FULL,
    TABLE_EPSILONS,
    TABLE_LAMBDA_FRACS,
    _LATEX_ROW_END,
)

DEFAULT_SWING_BASE = (
    "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
)
DEFAULT_MU = 100.0
DEFAULT_SEED = 0
DEFAULT_OUT_DIR = Path("fig/table_generation_time")
DEFAULT_OUT_DIR_DUP = Path("fig/table_generation_time_dup")


def _latex_dataset_name(dataset_id: str) -> str:
    return dataset_display_name(dataset_id).replace("_", r"\_")


def _generation_time_sec(
    swing_path: str,
    epsilon: int,
    mu: float,
    lambda_val: int,
) -> Optional[float]:
    """generation_time_sec for one (dataset path, λ, ε, μ) from swing_lambda_with_theta results."""
    df = load_swing_lambda_benchmark_results(
        swing_path,
        epsilon,
        mu,
        aggregate_1M=False,
    )
    for r in df.to_dict("records"):
        if int(r.get("lambda", -1)) != int(lambda_val):
            continue
        v = r.get("generation_time_sec")
        if v is None:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or x < 0:
            return None
        return x
    return None


def _hours_one_decimal(sec: Optional[float]) -> str:
    if sec is None:
        return r"\dots"
    h = sec / 3600.0
    return f"{h:.1f}"


def format_latex_generation_time_stacked(
    lambda_frac: str,
    print_label: str,
    resolution_lines: List[str],
    epsilon_blocks: List[Tuple[int, List[Tuple[str, str]]]],
    mu: float,
) -> str:
    """ε multirow | Dataset | Time (h); blocks = (ε, [(dataset_tex, hour_cell), ...])."""
    if not epsilon_blocks:
        return ""

    n_ds = len(epsilon_blocks[0][1])
    lines: List[str] = [
        f"% Generation time (h, 1 decimal) — {print_label}, "
        f"swing\_lambda\_with\_theta, $\\lambda={lambda_frac}\\,n$, $\\mu={mu:g}$. "
        f"Source: poisons_info.json generation_time_sec / 3600.",
        r"% Same dataset / $\lambda$ / $\varepsilon$ grid as fig/table_mopt_upper_bound.",
        r"% \usepackage{booktabs}",
        r"% \usepackage{multirow}",
        *resolution_lines,
        r"\begin{tabular}{@{}c l r@{}}",
        r"\toprule",
        " & & "
        + rf"\multicolumn{{1}}{{c}}{{$\lambda = {lambda_frac}\,n$}} \\",
        r"\cmidrule(lr){3-3}",
        r"$\varepsilon$ & Dataset & Time (h) " + _LATEX_ROW_END,
        r"\midrule",
    ]

    n_eps = len(epsilon_blocks)
    for b_idx, (eps, block_rows) in enumerate(epsilon_blocks):
        if len(block_rows) != n_ds:
            raise ValueError(
                f"Inconsistent row count for ε={eps}: expected {n_ds}, got {len(block_rows)}"
            )
        for i, (name, hour_cell) in enumerate(block_rows):
            if i == 0:
                eps_cell = rf"\multirow{{{n_ds}}}{{*}}{{{eps}}}"
            else:
                eps_cell = ""
            lines.append(
                f"{eps_cell} & {name} & {hour_cell}" + _LATEX_ROW_END
            )
        if b_idx < n_eps - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def write_generation_time_tables(
    *,
    swing_base: str,
    mu: float,
    seed: int,
    out_dir: Path,
    log_append: Optional[Path] = None,
    variants: Tuple[str, ...] = ("full",),
    duplicate_only: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_append is not None:
        log_append.parent.mkdir(parents=True, exist_ok=True)

    dup_suffix = "_dup" if duplicate_only else ""

    for variant in variants:
        if variant == "full":
            datasets = (
                upper_bound_dataset_specs_full_duplicate_only()
                if duplicate_only
                else upper_bound_dataset_specs_full()
            )
            lambda_table_spec = (
                LAMBDA_TABLE_SPEC_FULL_DUPLICATES
                if duplicate_only
                else LAMBDA_TABLE_SPEC_FULL
            )
            print_label = "full"

            def swing_path_for_dataset(d: Dict[str, str]) -> str:
                return f"{swing_base}/{d['path']}"

        elif variant == "1m":
            datasets = (
                upper_bound_dataset_specs_duplicate_only()
                if duplicate_only
                else upper_bound_dataset_specs()
            )
            lambda_table_spec = (
                LAMBDA_TABLE_SPEC_1M_DUPLICATES
                if duplicate_only
                else LAMBDA_TABLE_SPEC_1M
            )
            print_label = "1M"

            def swing_path_for_dataset(d: Dict[str, str]) -> str:
                return f"{swing_base}/{d['path']}/seed{seed}"

        else:
            raise ValueError(f"unknown variant: {variant}")

        for lambda_frac in TABLE_LAMBDA_FRACS:
            resolution_lines: List[str] = []
            for d in datasets:
                name = d["name"]
                lam = lambda_table_spec[lambda_frac][name]
                resolution_lines.append(f"% {name}: λ = {lam} (f={lambda_frac})")

            epsilon_blocks: List[Tuple[int, List[Tuple[str, str]]]] = []
            for epsilon in TABLE_EPSILONS:
                block_rows: List[Tuple[str, str]] = []
                for d in datasets:
                    swing_path = swing_path_for_dataset(d)
                    lam = lambda_table_spec[lambda_frac][d["name"]]
                    sec = _generation_time_sec(swing_path, epsilon, mu, lam)
                    block_rows.append(
                        (
                            _latex_dataset_name(d["name"]),
                            _hours_one_decimal(sec),
                        )
                    )
                epsilon_blocks.append((epsilon, block_rows))

            body = format_latex_generation_time_stacked(
                lambda_frac,
                print_label,
                resolution_lines,
                epsilon_blocks,
                mu,
            )
            suffix = lambda_frac.replace(".", "p")
            output = out_dir / f"generation_time_{variant}_lambda{suffix}{dup_suffix}.tex"
            output.write_text(body + "\n", encoding="utf-8")
            msg = f"Wrote LaTeX table: {output}"
            print(msg, file=sys.stderr)
            if log_append is not None:
                with open(log_append, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write LaTeX tables for Proposed (swing_lambda_with_theta) generation time in hours."
        )
    )
    parser.add_argument(
        "--swing-base",
        default=DEFAULT_SWING_BASE,
        help="Base dir for swing_lambda_with_theta results",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=DEFAULT_MU,
        help="μ directory (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for 1M paths (seed{S} under dataset)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for .tex files",
    )
    parser.add_argument(
        "--log-append",
        type=Path,
        default=None,
        help="Optional log file to append messages",
    )
    parser.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Duplicate-key datasets only; use *_dup.tex under table_generation_time_dup by default.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.duplicate_only and args.out_dir == DEFAULT_OUT_DIR:
        out_dir = DEFAULT_OUT_DIR_DUP

    write_generation_time_tables(
        swing_base=args.swing_base,
        mu=args.mu,
        seed=args.seed,
        out_dir=out_dir,
        log_append=args.log_append,
        duplicate_only=args.duplicate_only,
    )


if __name__ == "__main__":
    main()
