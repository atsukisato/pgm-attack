"""
LaTeX tabular: m_opt for duplicate-key workloads only (no upper bound).

$\\lambda \\in \\{0.01\\,n, 0.1\\,n\\}$ side-by-side; $\\varepsilon \\in \\{16,32,64,128\\}$ stacked.
Columns: $\\varepsilon$, Dataset, Legit., then per-$\\lambda$ block: Proposed, Random, Random-Adj.

Output: fig/table_mopt_dup/mopt_dup_table_full.tex

Loads Proposed (swing_lambda_with_theta) and Random / Random-Adj from inject_poisons random /
_random_adjacent. Seed 0; full-scale uses leaf_seed under mu*.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from load_results_mopt import load_swing_lambda_benchmark_results
from plot_config import (
    LAMBDA_TABLE_SPEC_FULL_DUPLICATES,
    dataset_display_name,
    upper_bound_dataset_specs_full_duplicate_only,
)

TABLE_EPSILONS: Tuple[int, ...] = (16, 32, 64, 128)
TABLE_LAMBDA_FRACS: Tuple[str, ...] = ("0.01", "0.1")

_LATEX_ROW_END = r" \\"


def _lambda_spec_list_for_dataset(
    table: Dict[str, Dict[str, int]], dataset_id: str
) -> List[Tuple[int, str]]:
    return [(table[frac][dataset_id], frac) for frac in TABLE_LAMBDA_FRACS]


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


def _latex_compact_num(x: float | None) -> str:
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


def _latex_legit_cell(legit: float | None) -> str:
    if legit is None:
        return r"\dots"
    return _latex_compact_num(legit)


def _latex_prop_cell(prop: float | None, legit: float | None) -> str:
    if prop is None:
        return r"\dots"
    xi = _latex_compact_num(prop)
    if legit is not None and legit > 0:
        r_l = prop / legit
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


ThreeCells = Tuple[str, str, str]
DualThreeCells = Tuple[ThreeCells, ThreeCells]


def build_mopt_dup_row(
    dataset_id: str,
    swing_results_by_epsilon: Dict[int, List[Dict]],
    random_results_by_epsilon: Dict[int, List[Dict]],
    random_adj_results_by_epsilon: Dict[int, List[Dict]],
    epsilon: int,
    lambda_spec: List[Tuple[int, str]],
) -> Tuple[str, str, DualThreeCells, List[int]]:
    swing_results = swing_results_by_epsilon.get(epsilon, [])
    random_results = random_results_by_epsilon.get(epsilon, [])
    random_adj_results = random_adj_results_by_epsilon.get(epsilon, [])
    achieved_by_lambda = _achieved_by_lambda(swing_results)
    rand_by_lambda = _achieved_by_lambda(random_results)
    radj_by_lambda = _achieved_by_lambda(random_adj_results)

    legit = achieved_by_lambda.get(0)

    name = _latex_dataset_name(dataset_id)
    legit_cell = _latex_legit_cell(legit)

    targets = [lam for lam, _ in lambda_spec]
    resolved_lambdas = targets

    groups: List[ThreeCells] = []
    for lam_resolved in resolved_lambdas:
        prop = achieved_by_lambda.get(lam_resolved)
        prop_cell = _latex_prop_cell(prop, legit)
        rand_cell = _latex_prop_cell(rand_by_lambda.get(lam_resolved), legit)
        radj_cell = _latex_prop_cell(radj_by_lambda.get(lam_resolved), legit)
        groups.append((prop_cell, rand_cell, radj_cell))

    if len(groups) != 2:
        raise ValueError(f"expected two λ blocks, got {len(groups)}")
    return name, legit_cell, (groups[0], groups[1]), resolved_lambdas


def format_latex_mopt_dup_stacked_eps(
    print_label: str,
    resolution_lines: List[str],
    epsilon_blocks: List[Tuple[int, List[Tuple[str, str, DualThreeCells]]]],
) -> str:
    if not epsilon_blocks:
        return ""

    n_ds = len(epsilon_blocks[0][1])
    col_spec = r"@{}c l r rrr rrr@{}"
    lines: List[str] = [
        f"% m_opt (duplicate workloads) — {print_label}, "
        f"$\\lambda \\in \\{{0.01\\,n, 0.1\\,n\\}}$; "
        f"$\\varepsilon \\in \\{{16,32,64,128\\}}$ stacked.",
        r"% \usepackage{booktabs}",
        r"% \setlength{\tabcolsep}{3pt}  % optional",
        r"% Random / Random-Adj.: inject_poisons_to_minimize_segment_length_random / _random_adjacent.",
        *resolution_lines,
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{$\lambda = 0.01\,n$} & \multicolumn{3}{c}{$\lambda = 0.1\,n$} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}",
        r"$\varepsilon$ & Dataset & Legit. & Proposed & Random & Random-Adj."
        r" & Proposed & Random & Random-Adj. \\",
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
            p1, p2, p3 = g0
            q1, q2, q3 = g1
            eps_cell = str(eps) if row_idx == 0 else ""
            lines.append(
                f"{eps_cell} & {name} & {legit_cell} & {p1} & {p2} & {p3} & "
                f"{q1} & {q2} & {q3}"
                + _LATEX_ROW_END
            )
        if b_idx < n_eps - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main_stacked_table(
    variant: str,
    log_append: Path | None,
) -> None:
    seed = 0
    mu = 100
    epsilons = list(TABLE_EPSILONS)
    swing_base_dir = (
        "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
    )
    random_base_dir = "results/poisoning/inject_poisons_to_minimize_segment_length_random"
    random_adj_base_dir = (
        "results/poisoning/inject_poisons_to_minimize_segment_length_random_adjacent"
    )

    if variant != "full":
        raise ValueError("only variant 'full' is supported")

    datasets = upper_bound_dataset_specs_full_duplicate_only()
    lambda_table_spec = LAMBDA_TABLE_SPEC_FULL_DUPLICATES
    print_label = "full"
    random_leaf_seed: int | None = seed

    def swing_path_for_dataset(d: Dict[str, str]) -> str:
        return f"{swing_base_dir}/{d['path']}"

    def inject_random_path_for_dataset(d: Dict[str, str], base: str) -> str:
        return f"{base}/{d['path']}"

    dataset_payloads: List[Dict] = []
    for dataset in datasets:
        swing_path = swing_path_for_dataset(dataset)
        random_path = inject_random_path_for_dataset(dataset, random_base_dir)
        random_adj_path = inject_random_path_for_dataset(dataset, random_adj_base_dir)

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
                "swing_results_by_epsilon": swing_results_by_epsilon,
                "random_results_by_epsilon": random_results_by_epsilon,
                "random_adj_results_by_epsilon": random_adj_results_by_epsilon,
            }
        )

    out_dir = Path("fig/table_mopt_dup")
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

    epsilon_blocks: List[Tuple[int, List[Tuple[str, str, DualThreeCells]]]] = []
    for epsilon in epsilons:
        block_rows: List[Tuple[str, str, DualThreeCells]] = []
        for pl in dataset_payloads:
            name = pl["name"]
            lambda_spec = _lambda_spec_list_for_dataset(lambda_table_spec, name)
            name_tex, legit_cell, g_pair, _ = build_mopt_dup_row(
                name,
                pl["swing_results_by_epsilon"],
                pl["random_results_by_epsilon"],
                pl["random_adj_results_by_epsilon"],
                epsilon,
                lambda_spec,
            )
            block_rows.append((name_tex, legit_cell, g_pair))
        epsilon_blocks.append((epsilon, block_rows))

    body = format_latex_mopt_dup_stacked_eps(
        print_label,
        resolution_lines,
        epsilon_blocks,
    )
    output = out_dir / f"mopt_dup_table_{variant}.tex"
    output.write_text(body + "\n", encoding="utf-8")
    msg = f"Wrote LaTeX table: {output}"
    print(msg, file=sys.stderr)
    if log_append is not None:
        with open(log_append, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LaTeX tables: m_opt for duplicate-key datasets (no upper bound)."
    )
    args = parser.parse_args()

    main_stacked_table(
        "full",
        Path("fig/table_mopt_dup/print_table_mopt_dup.log"),
    )
