"""
Write LaTeX tabular for m_opt upper bound vs achieved (swing_lambda_with_theta).

One table per variant: $\\lambda=0.01\\,n$ and $\\lambda=0.1\\,n$ side-by-side (Poisoning + Upper Bound
each). 1M variant not emitted. Use --duplicate-only for duplicate-key workloads only (*_dup.tex).
Stacks $\\varepsilon \\in \\{16,32,64,128\\}$ vertically; $\\varepsilon$ appears only on the first row
of each $\\varepsilon$ block. Columns: $\\varepsilon$, Dataset, Legit., then five columns per $\\lambda$.
\\midrule between $\\varepsilon$ blocks.

Output: fig/table_mopt_upper_bound/mopt_upper_bound_table_full.tex (and *_dup with --duplicate-only).

Loads Proposed (swing_lambda_with_theta) and Random / Random-Adj from inject_poisons random /
_random_adjacent. Seed 0; full-scale uses leaf_seed under mu*.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

TABLE_EPSILONS: Tuple[int, ...] = (16, 32, 64, 128)
TABLE_LAMBDA_FRACS: Tuple[str, ...] = ("0.01", "0.1")

# Tabular row terminator. Append after f-strings; do not end an f-string with \\ (one backslash out).
_LATEX_ROW_END = r" \\"


def _lambda_spec_list_for_dataset(
    table: Dict[str, Dict[str, int]], dataset_id: str
) -> List[Tuple[int, str]]:
    """Ordered (λ, frac) columns: 0.01n then 0.1n (see TABLE_LAMBDA_FRACS)."""
    return [(table[frac][dataset_id], frac) for frac in TABLE_LAMBDA_FRACS]


def _format_sig3(x: float) -> str:
    """Format a nonnegative number to 3 significant figures (table / ratios)."""
    if x == 0 or not math.isfinite(x):
        return "0"
    ax = abs(float(x))
    m = int(math.floor(math.log10(ax)))
    nd = 2 - m  # decimal places for 3 significant digits
    r = round(ax, nd)
    if nd <= 0:
        return str(int(r))
    s = f"{r:.{nd}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _format_ratio_times(x: float) -> str:
    """
    Multiplier in cell \\footnotesize (...$\\times$) vs Legit.
    Exactly 3 significant figures, keeping trailing zeros (e.g. 1.0001 -> '1.00', not '1').
    """
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
    """Compact segment counts: K/M suffix with 3 s.f. on the coefficient; <1e3 uses 3 s.f."""
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


def _latex_prop_cell(prop: Optional[float], legit: Optional[float]) -> str:
    if prop is None:
        return r"\dots"
    xi = _latex_compact_num(prop)
    if legit is not None and legit > 0:
        r_l = prop / legit
        return rf"{xi} {{\footnotesize ({_format_ratio_times(r_l)}$\times$)}}"
    return xi


def _latex_ub_cell(ub: Optional[float], legit: Optional[float]) -> str:
    if ub is None:
        return r"\dots"
    xi = _latex_compact_num(ub)
    if legit is not None and legit > 0:
        r_l = ub / legit
        return rf"{xi} {{\footnotesize ({_format_ratio_times(r_l)}$\times$)}}"
    return xi


def _latex_dataset_name(dataset_id: str) -> str:
    """Display name with LaTeX escapes for underscore etc."""
    s = dataset_display_name(dataset_id)
    return s.replace("_", r"\_")


def _achieved_by_lambda(records: List[Dict]) -> Dict[int, float]:
    """Segment count per λ from benchmark rows (PGM first level or m_opt_after_poisoning)."""
    out: Dict[int, float] = {}
    for r in records:
        v = r.get("first_level_segments_num") or r.get("m_opt_after_poisoning")
        if v is not None and r.get("lambda") is not None:
            out[int(r["lambda"])] = float(v)
    return out


def _ub_and_achieved_maps(
    ub_results: List[Dict], swing_results: List[Dict]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Returns (instance upper bound per λ, achieved m_opt per λ from swing)."""
    ub_by_lambda = {
        int(r["lambda"]): float(r["m_opt_upper_bound"])
        for r in ub_results
        if r.get("m_opt_upper_bound") is not None and r.get("lambda") is not None
    }
    achieved_by_lambda = _achieved_by_lambda(swing_results)
    return ub_by_lambda, achieved_by_lambda


def _agnostic_ub_from_legit(legit: Optional[float], lam: int) -> Optional[float]:
    """Loose / agnostic bound: legitimate m_opt + total poison budget λ (matches C++ loose UB)."""
    if legit is None:
        return None
    return float(legit) + float(lam)


def build_upper_bound_table_row(
    dataset_id: str,
    ub_results_by_epsilon: Dict[int, List[Dict]],
    swing_results_by_epsilon: Dict[int, List[Dict]],
    random_results_by_epsilon: Dict[int, List[Dict]],
    random_adj_results_by_epsilon: Dict[int, List[Dict]],
    epsilon: int,
    lambda_spec: List[Tuple[int, str]],
) -> Tuple[str, str, List[Tuple[str, str, str, str, str]], List[int]]:
    """One row: Legit., one or more λ blocks (Proposed, Random, Random-Adj., Agn.-UB, Inst.-UB)."""
    ub_results = ub_results_by_epsilon.get(epsilon, [])
    swing_results = swing_results_by_epsilon.get(epsilon, [])
    random_results = random_results_by_epsilon.get(epsilon, [])
    random_adj_results = random_adj_results_by_epsilon.get(epsilon, [])
    ub_by_lambda, achieved_by_lambda = _ub_and_achieved_maps(ub_results, swing_results)
    rand_by_lambda = _achieved_by_lambda(random_results)
    radj_by_lambda = _achieved_by_lambda(random_adj_results)

    legit = achieved_by_lambda.get(0)
    if legit is None:
        legit = ub_by_lambda.get(0)

    name = _latex_dataset_name(dataset_id)
    legit_cell = _latex_legit_cell(legit)

    targets = [lam for lam, _ in lambda_spec]
    resolved_lambdas = targets

    groups: List[Tuple[str, str, str, str, str]] = []
    for lam_resolved in resolved_lambdas:
        prop = achieved_by_lambda.get(lam_resolved)
        ub_inst = ub_by_lambda.get(lam_resolved)
        ub_agn = _agnostic_ub_from_legit(legit, lam_resolved)
        prop_cell = _latex_prop_cell(prop, legit)
        ub_agn_cell = _latex_ub_cell(ub_agn, legit)
        ub_inst_cell = _latex_ub_cell(ub_inst, legit)
        rand_cell = _latex_prop_cell(rand_by_lambda.get(lam_resolved), legit)
        radj_cell = _latex_prop_cell(radj_by_lambda.get(lam_resolved), legit)
        groups.append((prop_cell, rand_cell, radj_cell, ub_agn_cell, ub_inst_cell))

    return name, legit_cell, groups, resolved_lambdas


def format_latex_upper_bound_table(
    rows: List[Tuple[str, str, List[Tuple[str, str, str, str, str]]]],
    print_label: str,
    epsilon: int,
    lambda_fracs: List[str],
    resolution_lines: List[str],
) -> str:
    """Booktabs: one $\\varepsilon$, both λ side-by-side (legacy layout)."""
    k = len(lambda_fracs)
    if k == 0:
        return ""

    cols_per_block = 5
    col_spec = "@{}l r" + (" rrr rr" * k) + "@{}"
    frac_str = ", ".join(lambda_fracs)
    lines: List[str] = [
        f"% Upper bound vs achieved — {print_label}, $\\varepsilon={epsilon}$",
        "% $\\lambda = f\\,n$ blocks with $f \\in \\{"
        + frac_str
        + "\\}$; numeric $\\lambda$ per dataset below.",
        r"% \usepackage{booktabs}",
        r"% \setlength{\tabcolsep}{3pt}  % optional, match paper table",
        r"% Random / Random-Adj.: inject_poisons_to_minimize_segment_length_random / _random_adjacent.",
        *resolution_lines,
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & & "
        + " & ".join(
            rf"\multicolumn{{5}}{{c}}{{$\lambda = {frac}\,n$}}"
            for frac in lambda_fracs
        )
        + _LATEX_ROW_END,
    ]
    cmid_lambda = " ".join(
        f"\\cmidrule(lr){{{3 + cols_per_block * i}-{2 + cols_per_block * (i + 1)}}}"
        for i in range(k)
    )
    lines.append(cmid_lambda)
    poison_ub_block = (
        r"\multicolumn{3}{c}{Poisoning} & \multicolumn{2}{c}{Upper Bound}"
    )
    lines.append(" & & " + " & ".join([poison_ub_block] * k) + _LATEX_ROW_END)
    cmid_poison_ub = " ".join(
        piece
        for i in range(k)
        for piece in (
            f"\\cmidrule(lr){{{3 + cols_per_block * i}-{5 + cols_per_block * i}}}",
            f"\\cmidrule(lr){{{6 + cols_per_block * i}-{7 + cols_per_block * i}}}",
        )
    )
    lines.append(cmid_poison_ub)
    block_header = r"Proposed & Random & Random-Adj. & Agnostic UB & Instance UB"
    lines.append(
        "Dataset & Legit. & "
        + " & ".join([block_header] * k)
        + _LATEX_ROW_END
    )
    lines.append(r"\midrule")
    for name, legit_cell, groups in rows:
        rest: List[str] = []
        for prop_cell, rand_cell, radj_cell, ub_agn_cell, ub_inst_cell in groups:
            rest.extend(
                [
                    prop_cell,
                    rand_cell,
                    radj_cell,
                    ub_agn_cell,
                    ub_inst_cell,
                ]
            )
        lines.append(
            f"{name} & {legit_cell} & " + " & ".join(rest) + _LATEX_ROW_END
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


FiveCells = Tuple[str, str, str, str, str]
DualFiveCells = Tuple[FiveCells, FiveCells]


def format_latex_upper_bound_table_stacked_eps(
    print_label: str,
    resolution_lines: List[str],
    epsilon_blocks: List[Tuple[int, List[Tuple[str, str, DualFiveCells]]]],
) -> str:
    """
    Both λ = 0.01n and λ = 0.1n side-by-side; rows grouped by ε (16,32,64,128).
    ε appears only on the first row of each ε block.
    """
    if not epsilon_blocks:
        return ""

    n_ds = len(epsilon_blocks[0][1])
    col_spec = r"@{}c l r rrr rr rrr rr@{}"
    lines: List[str] = [
        f"% Upper bound vs achieved — {print_label}, "
        f"$\\lambda \\in \\{{0.01\\,n, 0.1\\,n\\}}$; "
        f"$\\varepsilon \\in \\{{16,32,64,128\\}}$ stacked.",
        r"% \usepackage{booktabs}",
        r"% \setlength{\tabcolsep}{3pt}  % optional",
        r"% Random / Random-Adj.: inject_poisons_to_minimize_segment_length_random / _random_adjacent.",
        *resolution_lines,
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r" & & & \multicolumn{5}{c}{$\lambda = 0.01\,n$} & \multicolumn{5}{c}{$\lambda = 0.1\,n$} \\",
        r"\cmidrule(lr){4-8} \cmidrule(lr){9-13}",
        r" & & & \multicolumn{3}{c}{Poisoning} & \multicolumn{2}{c}{Upper Bound}"
        r" & \multicolumn{3}{c}{Poisoning} & \multicolumn{2}{c}{Upper Bound} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-8} \cmidrule(lr){9-11} \cmidrule(lr){12-13}",
        r"$\varepsilon$ & Dataset & Legit. & Proposed & Random & Random-Adj. & Agnostic UB & Instance UB"
        r" & Proposed & Random & Random-Adj. & Agnostic UB & Instance UB \\",
        r"\midrule",
    ]

    n_eps = len(epsilon_blocks)
    for b_idx, (eps, block_rows) in enumerate(epsilon_blocks):
        if len(block_rows) != n_ds:
            raise ValueError(
                f"Inconsistent row count for ε={eps}: expected {n_ds}, got {len(block_rows)}"
            )
        for row_idx, (name, legit_cell, g_pair) in enumerate(block_rows):
            g0, g1 = g_pair
            p1, p2, p3, p4, p5 = g0
            q1, q2, q3, q4, q5 = g1
            eps_cell = str(eps) if row_idx == 0 else ""
            lines.append(
                f"{eps_cell} & {name} & {legit_cell} & {p1} & {p2} & {p3} & {p4} & {p5} & "
                f"{q1} & {q2} & {q3} & {q4} & {q5}"
                + _LATEX_ROW_END
            )
        if b_idx < n_eps - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main_stacked_tables(
    variant: str,
    log_append: Optional[Path],
    *,
    duplicate_only: bool = False,
) -> None:
    """Write `mopt_upper_bound_table_{variant}.tex` (both λ side-by-side, stacked ε)."""
    seed = 0
    mu = 100
    epsilons = list(TABLE_EPSILONS)
    ub_base_dir = "results/upper_bound/fix_w_per_block"
    swing_base_dir = (
        "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
    )
    random_base_dir = "results/poisoning/inject_poisons_to_minimize_segment_length_random"
    random_adj_base_dir = (
        "results/poisoning/inject_poisons_to_minimize_segment_length_random_adjacent"
    )

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
        random_leaf_seed: Optional[int] = seed

        def swing_path_for_dataset(d: Dict[str, str]) -> str:
            return f"{swing_base_dir}/{d['path']}"

        def inject_random_path_for_dataset(d: Dict[str, str], base: str) -> str:
            return f"{base}/{d['path']}"

    else:
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
        random_leaf_seed = None

        def swing_path_for_dataset(d: Dict[str, str]) -> str:
            return f"{swing_base_dir}/{d['path']}/seed{seed}"

        def inject_random_path_for_dataset(d: Dict[str, str], base: str) -> str:
            return f"{base}/{d['path']}/seed{seed}"

    dataset_payloads: List[Dict[str, Any]] = []
    for dataset in datasets:
        ub_path = f"{ub_base_dir}/{dataset['path']}"
        swing_path = swing_path_for_dataset(dataset)
        random_path = inject_random_path_for_dataset(dataset, random_base_dir)
        random_adj_path = inject_random_path_for_dataset(dataset, random_adj_base_dir)

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
        random_results_by_epsilon = {
            eps: load_swing_lambda_benchmark_results(
                random_path,
                eps,
                mu,
                aggregate_1M=False,
                leaf_seed=random_leaf_seed,
            ).to_dict("records")
            for eps in epsilons
        }
        random_adj_results_by_epsilon = {
            eps: load_swing_lambda_benchmark_results(
                random_adj_path,
                eps,
                mu,
                aggregate_1M=False,
                leaf_seed=random_leaf_seed,
            ).to_dict("records")
            for eps in epsilons
        }

        dataset_payloads.append(
            {
                "name": dataset["name"],
                "ub_results_by_epsilon": ub_results_by_epsilon,
                "swing_results_by_epsilon": swing_results_by_epsilon,
                "random_results_by_epsilon": random_results_by_epsilon,
                "random_adj_results_by_epsilon": random_adj_results_by_epsilon,
            }
        )

    out_dir = (
        Path("fig/table_mopt_upper_bound_dup")
        if duplicate_only
        else Path("fig/table_mopt_upper_bound")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_append is not None:
        log_append.parent.mkdir(parents=True, exist_ok=True)

    resolution_lines: List[str] = []
    for pl in dataset_payloads:
        name = pl["name"]
        lam_001 = lambda_table_spec["0.01"][name]
        lam_01 = lambda_table_spec["0.1"][name]
        resolution_lines.append(
            f"% {name}: λ = {lam_001} (f=0.01); λ = {lam_01} (f=0.1)"
        )

    epsilon_blocks: List[Tuple[int, List[Tuple[str, str, DualFiveCells]]]] = []
    for epsilon in epsilons:
        block_rows: List[Tuple[str, str, DualFiveCells]] = []
        for pl in dataset_payloads:
            name = pl["name"]
            lambda_spec = _lambda_spec_list_for_dataset(lambda_table_spec, name)
            name_tex, legit_cell, groups, _ = build_upper_bound_table_row(
                name,
                pl["ub_results_by_epsilon"],
                pl["swing_results_by_epsilon"],
                pl["random_results_by_epsilon"],
                pl["random_adj_results_by_epsilon"],
                epsilon,
                lambda_spec,
            )
            if len(groups) != 2:
                raise ValueError(f"expected two λ blocks, got {len(groups)}")
            block_rows.append((name_tex, legit_cell, (groups[0], groups[1])))
        epsilon_blocks.append((epsilon, block_rows))

    body = format_latex_upper_bound_table_stacked_eps(
        print_label,
        resolution_lines,
        epsilon_blocks,
    )
    dup_suffix = "_dup" if duplicate_only else ""
    output = out_dir / f"mopt_upper_bound_table_{variant}{dup_suffix}.tex"
    output.write_text(body + "\n", encoding="utf-8")
    msg = f"Wrote LaTeX table: {output}"
    print(msg, file=sys.stderr)
    if log_append is not None:
        with open(log_append, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaTeX tables: m_opt upper bound vs proposed/random.")
    parser.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Duplicate-key datasets only; write *_dup.tex under fig/table_mopt_upper_bound_dup/.",
    )
    args = parser.parse_args()

    base_log = "fig/table_mopt_upper_bound_dup" if args.duplicate_only else "fig/table_mopt_upper_bound"
    for variant in ("full",):
        main_stacked_tables(
            variant,
            Path(f"{base_log}/print_table_{variant}_stacked.log"),
            duplicate_only=args.duplicate_only,
        )
