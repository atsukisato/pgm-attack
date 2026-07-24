"""
Write LaTeX tabular for m_opt upper bound vs achieved (swing_lambda_with_theta).

Fixed layout: two column groups ($\\lambda=0.01\\,n$ and $\\lambda=0.1\\,n$) only in the TeX header.
Data ε is chosen per output file. ε not written into the table header.

Output: fig/table_mopt_upper_bound_small/mopt_upper_bound_table_small_full_eps{128}.tex only
(mopt_upper_bound_table_small_full.tex is not emitted). 1M variant not emitted.
Column headers: Poisoned / Instance UB vs Legit.
Legit. uses λ=0 at that ε. Seed 0.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from load_results_mopt import load_swing_lambda_benchmark_results
from load_results_upper_bound import load_upper_bound_results
from plot_config import (
    DATASET_IDS_1M,
    LAMBDA_TABLE_SPEC_1M_DUPLICATES,
    LAMBDA_TABLE_SPEC_FULL_DUPLICATES,
    dataset_display_name,
    upper_bound_dataset_specs,
    upper_bound_dataset_specs_duplicate_only,
    upper_bound_dataset_specs_full,
    upper_bound_dataset_specs_full_duplicate_only,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-dataset λ = frac·n (full / 200M-scale ids; keys match upper_bound_dataset_specs_full).
LAMBDA_TABLE_SPEC_FULL: Dict[str, Dict[str, int]] = {
    "0.01": {
        "books_800M": 8_000_000,
        "osm_cellids_800M": 8_000_000,
        "fb_200M": 2_000_000,
        "ycsb": 2_000_000,
        "longitudes": 2_000_000,
        "longlat": 2_000_000,
        "uniform_range2pow63_200M": 2_000_000,
        "normal_mu0_sigma1_range2pow63_200M": 2_000_000,
        "lognormal_sigma1_range2pow63_200M": 2_000_000,
    },
    "0.1": {
        "books_800M": 80_000_000,
        "osm_cellids_800M": 80_000_000,
        "fb_200M": 20_000_000,
        "ycsb": 20_000_000,
        "longitudes": 20_000_000,
        "longlat": 20_000_000,
        "uniform_range2pow63_200M": 20_000_000,
        "normal_mu0_sigma1_range2pow63_200M": 20_000_000,
        "lognormal_sigma1_range2pow63_200M": 20_000_000,
    },
}

# λ = frac·n for n = 1M (keys match upper_bound_dataset_specs / DATASET_IDS_1M).
LAMBDA_TABLE_SPEC_1M: Dict[str, Dict[str, int]] = {
    "0.01": {ds: 10_000 for ds in DATASET_IDS_1M},
    "0.1": {ds: 100_000 for ds in DATASET_IDS_1M},
}

SMALL_TABLE_EPSILONS_FOR_FILES: Tuple[int, ...] = (128,)
SMALL_TABLE_LAMBDA_FRACS: Tuple[str, ...] = ("0.01", "0.1")


def _format_sig3(x: float) -> str:
    """Format a nonnegative number to 3 significant figures (table / ratios)."""
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
    """Ratio in cell \\footnotesize (...$\\times$) vs Legit."""
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
    """Compact segment counts: K/M suffix with 3 s.f. on the coefficient."""
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


def _latex_legit_cell(legit: Optional[float]) -> str:
    if legit is None:
        return r"\dots"
    return _latex_compact_num(legit)


def _latex_poisoned_mopt_cell(prop: Optional[float], legit: Optional[float]) -> str:
    """Poisoned $m_\\mathrm{opt}$ with \\footnotesize ratio $\\times$ vs Legit."""
    if prop is None:
        return r"\dots"
    xi = _latex_compact_num(prop)
    if legit is not None and legit > 0:
        r_l = prop / legit
        return rf"{xi} {{\footnotesize ({_format_ratio_times(r_l)}$\times$)}}"
    return xi


def _latex_ub_mopt_cell(ub: Optional[float], legit: Optional[float]) -> str:
    """Upper bound with \\footnotesize ratio $\\times$ vs Legit."""
    if ub is None:
        return r"\dots"
    xi = _latex_compact_num(ub)
    if legit is not None and legit > 0:
        r_l = ub / legit
        return rf"{xi} {{\footnotesize ({_format_ratio_times(r_l)}$\times$)}}"
    return xi


def _latex_dataset_name(dataset_id: str) -> str:
    s = dataset_display_name(dataset_id)
    return s.replace("_", r"\_")


def _achieved_by_lambda(records: List[Dict]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for r in records:
        v = r.get("first_level_segments_num") or r.get("m_opt_after_poisoning")
        if v is not None and r.get("lambda") is not None:
            out[int(r["lambda"])] = float(v)
    return out


def _ub_and_achieved_maps(
    ub_results: List[Dict], swing_results: List[Dict]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    ub_by_lambda = {
        int(r["lambda"]): float(r["m_opt_upper_bound"])
        for r in ub_results
        if r.get("m_opt_upper_bound") is not None and r.get("lambda") is not None
    }
    achieved_by_lambda = _achieved_by_lambda(swing_results)
    return ub_by_lambda, achieved_by_lambda


def build_small_combined_row(
    dataset_id: str,
    lambda_table_spec: Dict[str, Dict[str, int]],
    ub_results_by_epsilon: Dict[int, List[Dict]],
    swing_results_by_epsilon: Dict[int, List[Dict]],
    epsilon: int,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """One row: name, Legit. cell, two (Poisoned, Upper Bound) pairs."""
    ub_results = ub_results_by_epsilon.get(epsilon, [])
    swing_results = swing_results_by_epsilon.get(epsilon, [])
    ub_by_lambda, achieved_by_lambda = _ub_and_achieved_maps(
        ub_results, swing_results
    )
    legit = achieved_by_lambda.get(0)
    if legit is None:
        legit = ub_by_lambda.get(0)

    name = _latex_dataset_name(dataset_id)
    legit_cell = _latex_legit_cell(legit)

    groups: List[Tuple[str, str]] = []
    for frac in SMALL_TABLE_LAMBDA_FRACS:
        lam = lambda_table_spec[frac][dataset_id]
        prop = achieved_by_lambda.get(lam)
        ub = ub_by_lambda.get(lam)
        groups.append(
            (
                _latex_poisoned_mopt_cell(prop, legit),
                _latex_ub_mopt_cell(ub, legit),
            )
        )

    return name, legit_cell, groups


def format_latex_small_table(
    rows: List[Tuple[str, str, List[Tuple[str, str]]]],
    print_label: str,
    resolution_lines: List[str],
) -> str:
    lines: List[str] = [
        f"% Upper bound vs achieved — {print_label}.",
        r"% \usepackage{booktabs}",
        *resolution_lines,
        r"\begin{tabular}{@{}l r rr rr@{}}",
        r"    \toprule",
        r"        & & \multicolumn{2}{c}{$\lambda = 0.01\,n$} &"
        r" \multicolumn{2}{c}{$\lambda = 0.1\,n$} \\",
        r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"    Dataset & Legit. & Poisoned & Instance UB & Poisoned & Instance UB \\",
        r"    \midrule",
    ]
    for name, legit_cell, groups in rows:
        rest: List[str] = []
        for poison_cell, ub_cell in groups:
            rest.extend([poison_cell, ub_cell])
        lines.append(
            "    "
            + f"{name} & {legit_cell} & "
            + " & ".join(rest)
            + r" \\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(
    variant: str,
    output: Path,
    log_append: Optional[Path],
    epsilon: int,
    *,
    duplicate_only: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if log_append is not None:
        log_append.parent.mkdir(parents=True, exist_ok=True)

    seed = 0
    mu = 100
    epsilons = [epsilon]

    ub_base_dir = str(_REPO_ROOT / "results/upper_bound/fix_w_per_block")
    swing_base_dir = str(
        _REPO_ROOT
        / "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
    )

    if duplicate_only:
        if variant == "full":
            datasets = upper_bound_dataset_specs_full_duplicate_only()
            lambda_table_spec = LAMBDA_TABLE_SPEC_FULL_DUPLICATES
        else:
            datasets = upper_bound_dataset_specs_duplicate_only()
            lambda_table_spec = LAMBDA_TABLE_SPEC_1M_DUPLICATES
    elif variant == "full":
        datasets = upper_bound_dataset_specs_full()
        lambda_table_spec = LAMBDA_TABLE_SPEC_FULL
    else:
        datasets = upper_bound_dataset_specs()
        lambda_table_spec = LAMBDA_TABLE_SPEC_1M

    if variant == "full":
        print_label = "full"

        def swing_path_for_dataset(d: Dict[str, str]) -> str:
            return f"{swing_base_dir}/{d['path']}"

    else:
        print_label = "1M"

        def swing_path_for_dataset(d: Dict[str, str]) -> str:
            return f"{swing_base_dir}/{d['path']}/seed{seed}"

    table_rows: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    resolution_lines: List[str] = []

    for dataset in datasets:
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

        name, legit_cell, groups = build_small_combined_row(
            dataset["name"],
            lambda_table_spec,
            ub_results_by_epsilon,
            swing_results_by_epsilon,
            epsilon,
        )
        table_rows.append((name, legit_cell, groups))

        parts = []
        for frac in SMALL_TABLE_LAMBDA_FRACS:
            lam = lambda_table_spec[frac][dataset["name"]]
            parts.append(f"λ={lam} (f={frac})")
        resolution_lines.append(f"% {dataset['name']}: " + ", ".join(parts))

    body = format_latex_small_table(table_rows, print_label, resolution_lines)
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
    ap = argparse.ArgumentParser(description="LaTeX small tables: m_opt upper bound vs proposed.")
    ap.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Duplicate-key datasets only; *_dup.tex under fig/table_mopt_upper_bound_small_dup/.",
    )
    args = ap.parse_args()
    dup = args.duplicate_only
    out_dir = _REPO_ROOT / (
        "fig/table_mopt_upper_bound_small_dup"
        if dup
        else "fig/table_mopt_upper_bound_small"
    )
    for eps in SMALL_TABLE_EPSILONS_FOR_FILES:
        for variant in ("full",):
            main(
                variant,
                _out_path_with_dup(
                    out_dir / f"mopt_upper_bound_table_small_{variant}_eps{eps}.tex", dup
                ),
                _out_path_with_dup(
                    out_dir / f"print_table_small_{variant}_eps{eps}.log", dup
                ),
                eps,
                duplicate_only=dup,
            )
