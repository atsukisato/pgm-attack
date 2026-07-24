"""
LaTeX table: poisoned m_opt vs legitimate for ε ∈ {16,32,64,128}.

Same data paths as print_table_mopt_upper_bound_small.py.
Only $\\lambda = 0.1\\,n$ (Poisoned vs Legit. at each ε).
Per cell: compact poisoned m_opt with \\footnotesize ratio vs Legit. (λ=0) at that ε.
Seed 0, μ=100.

Output: fig/table_mopt_upper_bound_various_epsilon/mopt_upper_bound_table_various_epsilon_full.tex
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from load_results_mopt import load_swing_lambda_benchmark_results
from plot_config import (
    DATASET_IDS_1M,
    LAMBDA_TABLE_SPEC_VARIOUS_EPS_1M_DUPLICATES,
    LAMBDA_TABLE_SPEC_VARIOUS_EPS_FULL_DUPLICATES,
    dataset_display_name,
    upper_bound_dataset_specs,
    upper_bound_dataset_specs_duplicate_only,
    upper_bound_dataset_specs_full,
    upper_bound_dataset_specs_full_duplicate_only,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

LAMBDA_TABLE_SPEC_FULL: Dict[str, int] = {
    "books_800M": 80_000_000,
    "osm_cellids_800M": 80_000_000,
    "fb_200M": 20_000_000,
    "ycsb": 20_000_000,
    "longitudes": 20_000_000,
    "longlat": 20_000_000,
    "uniform_range2pow63_200M": 20_000_000,
    "normal_mu0_sigma1_range2pow63_200M": 20_000_000,
    "lognormal_sigma1_range2pow63_200M": 20_000_000,
}

LAMBDA_TABLE_SPEC_1M: Dict[str, int] = {ds: 100_000 for ds in DATASET_IDS_1M}

TABLE_EPSILONS: Tuple[int, ...] = (16, 32, 64, 128)
TABLE_LAMBDA_FRAC = "0.1"


def _format_sig3(x: float) -> str:
    if x == 0 or not math.isfinite(x):
        return "0"
    ax = abs(float(x))
    m = int(math.floor(math.log10(ax)))
    nd = 2 - m
    r = round(ax, nd)
    if nd <= 0:
        return str(int(r))
    s = f"{r:.{nd}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _format_ratio_times(x: float) -> str:
    if not math.isfinite(x):
        return "0.00"
    ax = abs(float(x))
    if ax == 0:
        return "0.00"
    m = int(math.floor(math.log10(ax)))
    nd = max(0, 2 - m)
    r = round(ax, nd)
    if nd <= 0:
        return str(int(r))
    return f"{r:.{nd}f}"


def _latex_compact_num(x: Optional[float]) -> str:
    if x is None:
        return r"\dots"
    v = float(x)
    av = abs(v)
    sign = "-" if v < 0 else ""
    if av >= 1_000_000:
        coeff = av / 1_000_000
        return sign + _format_sig3(coeff) + "M"
    if av >= 1000:
        coeff = av / 1000
        return sign + _format_sig3(coeff) + "K"
    return sign + _format_sig3(av)


def _latex_poisoned_mopt_cell(prop: Optional[float], legit: Optional[float]) -> str:
    if prop is None:
        return r"\dots"
    xi = _latex_compact_num(prop)
    if legit is not None and legit > 0:
        r_l = prop / legit
        return rf"{xi} {{\footnotesize ({_format_ratio_times(r_l)}$\times$)}}"
    return xi


def _achieved_by_lambda(records: List[Dict]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for r in records:
        v = r.get("first_level_segments_num") or r.get("m_opt_after_poisoning")
        if v is not None and r.get("lambda") is not None:
            out[int(r["lambda"])] = float(v)
    return out


def build_various_epsilon_row(
    dataset_id: str,
    lambda_by_dataset: Dict[str, int],
    swing_results_by_epsilon: Dict[int, List[Dict]],
) -> Tuple[str, List[str]]:
    """One row: name, then 4 cells for ε = 16..128 at λ = 0.1·n."""
    name = dataset_display_name(dataset_id).replace("_", r"\_")
    lam = lambda_by_dataset[dataset_id]
    cells: List[str] = []
    for epsilon in TABLE_EPSILONS:
        swing_results = swing_results_by_epsilon.get(epsilon, [])
        achieved_by_lambda = _achieved_by_lambda(swing_results)
        legit = achieved_by_lambda.get(0)
        prop = achieved_by_lambda.get(lam)
        cells.append(_latex_poisoned_mopt_cell(prop, legit))
    return name, cells


def format_latex_table(
    rows: List[Tuple[str, List[str]]],
    print_label: str,
    resolution_lines: List[str],
) -> str:
    n_eps = len(TABLE_EPSILONS)
    last_col = 1 + n_eps
    eps_headers = " & ".join(rf"$\varepsilon={eps}$" for eps in TABLE_EPSILONS)
    lines: List[str] = [
        f"% Poisoned m_opt vs Legit., $\\lambda=0.1\\,n$ — ε ∈ {{16,32,64,128}} — {print_label}.",
        r"% \usepackage{booktabs}",
        *resolution_lines,
        rf"\begin{{tabular}}{{@{{}}l *{{{n_eps}}}{{r}}@{{}}}}",
        r"    \toprule",
        r"    & \multicolumn{"
        + str(n_eps)
        + r"}{c}{Poisoned} \\",
        r"    \cmidrule(lr){2-" + str(last_col) + "}",
        r"    Dataset & " + eps_headers + r" \\",
        r"    \midrule",
    ]

    for name, flat in rows:
        lines.append("    " + name + " & " + " & ".join(flat) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(
    variant: str,
    output: Path,
    log_append: Optional[Path],
    *,
    duplicate_only: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if log_append is not None:
        log_append.parent.mkdir(parents=True, exist_ok=True)

    seed = 0
    mu = 100
    epsilons = list(TABLE_EPSILONS)

    swing_base_dir = str(
        _REPO_ROOT
        / "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
    )

    if duplicate_only:
        if variant == "full":
            datasets = upper_bound_dataset_specs_full_duplicate_only()
            lambda_by_dataset = LAMBDA_TABLE_SPEC_VARIOUS_EPS_FULL_DUPLICATES
        else:
            datasets = upper_bound_dataset_specs_duplicate_only()
            lambda_by_dataset = LAMBDA_TABLE_SPEC_VARIOUS_EPS_1M_DUPLICATES
    elif variant == "full":
        datasets = upper_bound_dataset_specs_full()
        lambda_by_dataset = LAMBDA_TABLE_SPEC_FULL
    else:
        datasets = upper_bound_dataset_specs()
        lambda_by_dataset = LAMBDA_TABLE_SPEC_1M

    if variant == "full":
        print_label = "full"

        def swing_path_for_dataset(d: Dict[str, str]) -> str:
            return f"{swing_base_dir}/{d['path']}"

    else:
        print_label = "1M"

        def swing_path_for_dataset(d: Dict[str, str]) -> str:
            return f"{swing_base_dir}/{d['path']}/seed{seed}"

    table_rows: List[Tuple[str, List[str]]] = []
    resolution_lines: List[str] = []

    for dataset in datasets:
        swing_path = swing_path_for_dataset(dataset)
        swing_results_by_epsilon = {
            eps: load_swing_lambda_benchmark_results(
                swing_path, eps, mu, aggregate_1M=False
            ).to_dict("records")
            for eps in epsilons
        }

        name, cells = build_various_epsilon_row(
            dataset["name"],
            lambda_by_dataset,
            swing_results_by_epsilon,
        )
        table_rows.append((name, cells))

        lam = lambda_by_dataset[dataset["name"]]
        resolution_lines.append(
            f"% {dataset['name']}: λ={lam} (f={TABLE_LAMBDA_FRAC})"
        )

    body = format_latex_table(table_rows, print_label, resolution_lines)
    output.write_text(body, encoding="utf-8")
    msg = f"Wrote LaTeX table: {output}"
    print(msg, file=sys.stderr)
    if log_append is not None:
        with open(log_append, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _out_path_with_dup(path: Path, duplicate_only: bool) -> Path:
    if not duplicate_only:
        return path
    return path.with_name(path.stem + "_dup" + path.suffix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LaTeX table: poisoned m_opt vs legit at λ=0.1·n for several ε."
    )
    ap.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Duplicate-key datasets only; *_dup under fig/table_mopt_upper_bound_various_epsilon_dup/.",
    )
    args = ap.parse_args()
    dup = args.duplicate_only
    out_dir = _REPO_ROOT / (
        "fig/table_mopt_upper_bound_various_epsilon_dup"
        if dup
        else "fig/table_mopt_upper_bound_various_epsilon"
    )
    for variant in ("full",):
        main(
            variant,
            _out_path_with_dup(
                out_dir / f"mopt_upper_bound_table_various_epsilon_{variant}.tex", dup
            ),
            _out_path_with_dup(
                out_dir / f"print_table_various_epsilon_{variant}.log", dup
            ),
            duplicate_only=dup,
        )
