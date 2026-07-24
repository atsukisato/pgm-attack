"""
LaTeX table: legit vs poisoned index size and query time for PGM-index, FITing-Tree, RadixSpline.

Reads benchmark JSON from inject_poisons_to_minimize_segment_length_swing_lambda_with_theta
(same path layout as print_table_mopt_upper_bound_small.py). Legit uses lambda0; poisoned uses
one λ per REAL_SYSTEM_POISON_FRAC. μ is REAL_SYSTEM_MU; ε is chosen per output file (see
constants below; ε is recorded in file comments only, not in the tabular header).
Column headers: Index Size [MB] and Query Time [$\mu\mathrm{s}$]; optional Construction Time [s]
for non-multi layouts; cells omit units.
Index Size uses decimal MB: index_size_in_kb / 1000; |MB| < 0.1 is shown with 3 decimal places.
Full multi-index table uses sub-column labels Legit., Proposed, Random, Random-Adj. (swing /
inject_poisons random / _random_adjacent at the same λ), without Construction Time; the PGM-only
ε-suffixed file keeps Legit. / Poisoned.

Output:
  fig/table_mopt_upper_bound_real_system/mopt_upper_bound_table_real_system_eps{16,128}_full.tex:
    PGM-index, FITing-Tree, RadixSpline with an Index column; four poisoned variants per metric
    (index size and query time only); one file per ε in REAL_SYSTEM_FULL_TABLE_EPSILONS.
  fig/table_mopt_upper_bound_real_system/mopt_upper_bound_table_real_system_eps128.tex:
    PGM-index only; no Index column (tabular / \\cmidrule adjusted); no \\midrule between datasets.
  (1M variant not emitted).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from load_results_common import extract_mu_value
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

# Same λ = frac·n specs as print_table_mopt_upper_bound_small.py
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

LAMBDA_TABLE_SPEC_1M: Dict[str, Dict[str, int]] = {
    "0.01": {ds: 10_000 for ds in DATASET_IDS_1M},
    "0.1": {ds: 100_000 for ds in DATASET_IDS_1M},
}

# Multi-index full tables: mopt_upper_bound_table_real_system_eps{ε}_full.tex
REAL_SYSTEM_FULL_TABLE_EPSILONS: Tuple[int, ...] = (16, 128)
REAL_SYSTEM_TABLE_EPSILONS: Tuple[int, ...] = (128,)
REAL_SYSTEM_MU = 100.0
REAL_SYSTEM_POISON_FRAC = "0.1"
REAL_SYSTEM_SEED = 0

# Paths relative to repository root (parent of plot/).
_REPO_ROOT = Path(__file__).resolve().parent.parent

SWING_BASE_DIR = (
    "results/poisoning/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta"
)
RANDOM_BASE_DIR = "results/poisoning/inject_poisons_to_minimize_segment_length_random"
RANDOM_ADJ_BASE_DIR = (
    "results/poisoning/inject_poisons_to_minimize_segment_length_random_adjacent"
)

INDEX_SPECS: Tuple[Tuple[str, str], ...] = (
    ("PGM-index", "benchmark_pgm.json"),
    ("FITing-Tree", "benchmark_fiting_tree.json"),
    ("RadixSpline", "benchmark_radix_spline.json"),
)


def _find_mu_dir(epsilon_path: Path, mu: Optional[float] = None) -> Optional[Path]:
    if not epsilon_path.exists():
        return None
    mu_dirs = [d for d in epsilon_path.iterdir() if d.is_dir() and d.name.startswith("mu")]
    if not mu_dirs:
        return None
    if mu is not None:
        for d in mu_dirs:
            if extract_mu_value(d.name) == mu:
                return d
        return None
    mu_dirs_sorted = sorted(mu_dirs, key=lambda d: extract_mu_value(d.name))
    return mu_dirs_sorted[0]


def _resolve_benchmark_container_dir(mu_dir: Path, leaf_seed: Optional[int]) -> Optional[Path]:
    if leaf_seed is not None:
        seed_dir = mu_dir / f"seed{leaf_seed}"
        if (seed_dir / "benchmark_pgm.json").exists():
            return seed_dir
    if (mu_dir / "benchmark_pgm.json").exists():
        return mu_dir
    return None


def _benchmark_container(
    dataset_root: Path,
    lambda_val: int,
    epsilon: int,
    mu: float,
    leaf_seed: Optional[int] = None,
) -> Optional[Path]:
    lambda_dir = dataset_root / f"lambda{lambda_val}"
    epsilon_path = lambda_dir / f"epsilon{epsilon}"
    mu_dir = _find_mu_dir(epsilon_path, mu)
    if mu_dir is None:
        return None
    return _resolve_benchmark_container_dir(mu_dir, leaf_seed)


def _dataset_root(
    swing_base: Path, dataset_relpath: str, variant_is_full: bool, seed: int
) -> Path:
    p = swing_base / dataset_relpath
    if not variant_is_full:
        p = p / f"seed{seed}"
    return p


def _load_benchmark_metrics(
    json_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (index_size_in_kb, median_avg_query_time_ns, build_time_sec) or None if missing."""
    if not json_path.exists():
        return None, None, None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None, None
    kb = data.get("index_size_in_kb")
    med_ns = None
    qts = data.get("query_time_summary")
    if isinstance(qts, dict):
        med_ns = qts.get("median_avg_query_time_ns")
    bt = data.get("build_time_sec")
    try:
        kb_f = float(kb) if kb is not None else None
    except (TypeError, ValueError):
        kb_f = None
    try:
        ns_f = float(med_ns) if med_ns is not None else None
    except (TypeError, ValueError):
        ns_f = None
    try:
        bt_f = float(bt) if bt is not None else None
    except (TypeError, ValueError):
        bt_f = None
    return kb_f, ns_f, bt_f


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


def _latex_insert_digit_thousands_commas(decimal_str: str) -> str:
    """Insert {,} every 3 digits from the right in the integer part (LaTeX math)."""
    if not decimal_str or decimal_str == ".":
        return decimal_str
    sign = ""
    if decimal_str[0] == "-":
        sign = "-"
        decimal_str = decimal_str[1:]
    if "." in decimal_str:
        int_part, frac = decimal_str.split(".", 1)
        suffix = "." + frac
    else:
        int_part, suffix = decimal_str, ""
    if not int_part.isdigit():
        return sign + decimal_str + suffix
    n = len(int_part)
    if n <= 3:
        return sign + int_part + suffix
    r = n % 3
    if r == 0:
        r = 3
    chunks = [int_part[:r]]
    for i in range(r, n, 3):
        chunks.append(int_part[i : i + 3])
    return sign + "{,}".join(chunks) + suffix


def _latex_math_with_commas(x: float, decimals: int) -> str:
    """Format float as $...$ with thousands separators in the integer part."""
    s = f"{x:.{decimals}f}"
    return f"${_latex_insert_digit_thousands_commas(s)}$"


def _latex_index_size_mb_number(kb: float) -> str:
    """Decimal MB for table cells: index_size_in_kb / 1000; header carries [MB].

    Small values (|MB| < 0.1) use three decimal places, not a coarse 0.01.
    """
    mb = float(kb) / 1000.0
    ax = abs(mb)
    if ax >= 1.0:
        return _latex_math_with_commas(mb, 1)
    if ax >= 0.1:
        return _latex_math_with_commas(mb, 2)
    return _latex_math_with_commas(mb, 3)


def _format_index_mb_cell(kb: Optional[float]) -> str:
    if kb is None:
        return r"\dots"
    return _latex_index_size_mb_number(float(kb))


def _format_index_mb_poisoned(legit_kb: Optional[float], poison_kb: Optional[float]) -> str:
    if poison_kb is None:
        return r"\dots"
    s = _latex_index_size_mb_number(float(poison_kb))
    if legit_kb is not None and float(legit_kb) > 0:
        r_p = float(poison_kb) / float(legit_kb)
        s += rf" {{\footnotesize ({_format_ratio_times(r_p)}$\times$)}}"
    return s


def _format_mus_cell(ns: Optional[float]) -> str:
    if ns is None:
        return r"\dots"
    mus = float(ns) / 1000.0
    return _latex_math_with_commas(mus, 2)


def _format_mus_poisoned(legit_ns: Optional[float], poison_ns: Optional[float]) -> str:
    if poison_ns is None:
        return r"\dots"
    mus = float(poison_ns) / 1000.0
    s = _latex_math_with_commas(mus, 2)
    if legit_ns is not None and float(legit_ns) > 0:
        r_p = float(poison_ns) / float(legit_ns)
        s += rf" {{\footnotesize ({_format_ratio_times(r_p)}$\times$)}}"
    return s


def _latex_sec_number(sec: float) -> str:
    """Unitless seconds for table cells; column header carries [s]."""
    x = float(sec)
    ax = abs(x)
    if ax >= 100:
        return _latex_math_with_commas(x, 1)
    if ax >= 1:
        return _latex_math_with_commas(x, 2)
    return _latex_math_with_commas(x, 3)


def _format_sec_cell(sec: Optional[float]) -> str:
    if sec is None:
        return r"\dots"
    if not math.isfinite(float(sec)):
        return r"\dots"
    return _latex_sec_number(float(sec))


def _format_sec_poisoned(legit_sec: Optional[float], poison_sec: Optional[float]) -> str:
    if poison_sec is None:
        return r"\dots"
    if not math.isfinite(float(poison_sec)):
        return r"\dots"
    s = _latex_sec_number(float(poison_sec))
    if legit_sec is not None and float(legit_sec) > 0:
        r_p = float(poison_sec) / float(legit_sec)
        s += rf" {{\footnotesize ({_format_ratio_times(r_p)}$\times$)}}"
    return s


def _latex_dataset_name(dataset_id: str) -> str:
    return dataset_display_name(dataset_id).replace("_", r"\_")


def format_latex_real_system_table(
    blocks: List[List[str]],
    resolution_lines: List[str],
    print_label: str,
    epsilon: int,
    *,
    include_index_column: bool = True,
    use_multi_poison_columns: bool = False,
    include_construction_time: bool = True,
) -> str:
    if include_index_column:
        if use_multi_poison_columns:
            four = r" & Legit. & Proposed & Random & Random-Adj."
            if include_construction_time:
                tabular_spec = r"\begin{tabular}{@{}l l rrrr rrrr rrrr@{}}"
                header_top = (
                    r"Dataset & Index"
                    r" & \multicolumn{4}{c}{Index Size [MB]}"
                    r" & \multicolumn{4}{c}{Query Time [$\mu\mathrm{s}$]}"
                    r" & \multicolumn{4}{c}{Construction Time [s]} \\"
                )
                cmidrules = (
                    r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}"
                )
                header_sub = r"& " + four + four + four + r" \\"
                col_comment = (
                    r"% Columns: Dataset, Index, Index Size [MB] (×4), Query Time [$\mu\mathrm{s}$] (×4),"
                    r" Construction Time [s] (×4); Proposed=swing, Random/Random-Adj.=inject_poisons."
                )
            else:
                tabular_spec = r"\begin{tabular}{@{}l l rrrr rrrr@{}}"
                header_top = (
                    r"Dataset & Index"
                    r" & \multicolumn{4}{c}{Index Size [MB]}"
                    r" & \multicolumn{4}{c}{Query Time [$\mu\mathrm{s}$]} \\"
                )
                cmidrules = r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}"
                header_sub = r"& " + four + four + r" \\"
                col_comment = (
                    r"% Columns: Dataset, Index, Index Size [MB] (×4), Query Time [$\mu\mathrm{s}$] (×4);"
                    r" Proposed=swing, Random/Random-Adj.=inject_poisons."
                )
        else:
            tabular_spec = r"\begin{tabular}{@{}l l rr rr rr@{}}"
            header_top = (
                r"Dataset & Index"
                r" & \multicolumn{2}{c}{Index Size [MB]}"
                r" & \multicolumn{2}{c}{Query Time [$\mu\mathrm{s}$]}"
                r" & \multicolumn{2}{c}{Construction Time [s]} \\"
            )
            cmidrules = (
                r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}"
            )
            header_sub = (
                r"& "
                r"& Legit."
                r" & Poisoned"
                r" & Legit."
                r" & Poisoned"
                r" & Legit."
                r" & Poisoned \\"
            )
            col_comment = (
                r"% Columns: Dataset, Index, Index Size [MB] (×2), Query Time [$\mu\mathrm{s}$] (×2),"
                r" Construction Time [s] (×2)."
            )
    else:
        tabular_spec = r"\begin{tabular}{@{}l rr rr@{}}"
        header_top = (
            r"Dataset & \multicolumn{2}{c}{Index Size [MB]}"
            r" & \multicolumn{2}{c}{Query Time [$\mu\mathrm{s}$]} \\"
        )
        cmidrules = r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}"
        header_sub = (
            r"& Legit."
            r" & Poisoned"
            r" & Legit."
            r" & Poisoned \\"
        )
        col_comment = (
            r"% PGM-index only: Dataset, Index Size [MB] (×2), Query Time [$\mu\mathrm{s}$] (×2)."
        )

    if include_index_column:
        if use_multi_poison_columns and not include_construction_time:
            title_bits = "index size / query time"
        else:
            title_bits = "index size / query time / construction time"
    else:
        title_bits = "index size / query time"
    lines: List[str] = [
        f"% Real-system {title_bits} — {print_label}, $\\varepsilon={epsilon}$.",
        r"% \usepackage{booktabs}",
        col_comment,
        r"% Cells are unitless numbers; units in column headers.",
        *resolution_lines,
        tabular_spec,
        r"\toprule",
        header_top,
        cmidrules,
        header_sub,
        r"\midrule",
    ]
    lines.extend(blocks)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(
    variant: str,
    output: Path,
    log_append: Optional[Path],
    epsilon: int,
    *,
    duplicate_only: bool = False,
    pgm_only: bool = False,
    use_multi_poison_columns: bool = False,
    include_construction_time: bool = True,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if log_append is not None:
        log_append.parent.mkdir(parents=True, exist_ok=True)

    swing_base = _REPO_ROOT / SWING_BASE_DIR
    random_base = _REPO_ROOT / RANDOM_BASE_DIR
    random_adj_base = _REPO_ROOT / RANDOM_ADJ_BASE_DIR
    multi_poison = use_multi_poison_columns and not pgm_only
    if duplicate_only:
        if variant == "full":
            datasets = upper_bound_dataset_specs_full_duplicate_only()
            lambda_spec = LAMBDA_TABLE_SPEC_FULL_DUPLICATES
        else:
            datasets = upper_bound_dataset_specs_duplicate_only()
            lambda_spec = LAMBDA_TABLE_SPEC_1M_DUPLICATES
    elif variant == "full":
        datasets = upper_bound_dataset_specs_full()
        lambda_spec = LAMBDA_TABLE_SPEC_FULL
    else:
        datasets = upper_bound_dataset_specs()
        lambda_spec = LAMBDA_TABLE_SPEC_1M

    if variant == "full":
        print_label = "full"
        variant_is_full = True
    else:
        print_label = "1M"
        variant_is_full = False

    poison_lambda = lambda_spec[REAL_SYSTEM_POISON_FRAC]
    eps = epsilon
    mu = REAL_SYSTEM_MU
    seed = REAL_SYSTEM_SEED

    body_lines: List[List[str]] = []
    resolution_lines: List[str] = []

    for dataset in datasets:
        ds_id = dataset["name"]
        root = _dataset_root(swing_base, dataset["path"], variant_is_full, seed)
        lam_p = poison_lambda[ds_id]

        legit_c = _benchmark_container(root, 0, eps, mu, None)
        poison_c = _benchmark_container(root, lam_p, eps, mu, None)
        if multi_poison:
            rand_leaf = seed if variant_is_full else None
            root_rand = _dataset_root(random_base, dataset["path"], variant_is_full, seed)
            root_radj = _dataset_root(random_adj_base, dataset["path"], variant_is_full, seed)
            rand_c = _benchmark_container(root_rand, lam_p, eps, mu, rand_leaf)
            radj_c = _benchmark_container(root_radj, lam_p, eps, mu, rand_leaf)
        else:
            rand_c = None
            radj_c = None

        resolution_lines.append(
            f"% {ds_id}: legit λ=0, poison λ={lam_p} (f={REAL_SYSTEM_POISON_FRAC})"
        )

        index_specs = INDEX_SPECS[:1] if pgm_only else INDEX_SPECS
        group_lines: List[str] = []
        for i, (index_label, json_name) in enumerate(index_specs):
            legit_path = (legit_c / json_name) if legit_c else Path()
            poison_path = (poison_c / json_name) if poison_c else Path()
            l_kb, l_ns, l_bt = _load_benchmark_metrics(legit_path)
            p_kb, p_ns, p_bt = _load_benchmark_metrics(poison_path)

            ds_cell = _latex_dataset_name(ds_id) if i == 0 else ""
            if multi_poison:
                rand_path = (rand_c / json_name) if rand_c else Path()
                radj_path = (radj_c / json_name) if radj_c else Path()
                rand_kb, rand_ns, rand_bt = _load_benchmark_metrics(rand_path)
                radj_kb, radj_ns, radj_bt = _load_benchmark_metrics(radj_path)
                cells = (
                    f"{_format_index_mb_cell(l_kb)} & {_format_index_mb_poisoned(l_kb, p_kb)} & "
                    f"{_format_index_mb_poisoned(l_kb, rand_kb)} & {_format_index_mb_poisoned(l_kb, radj_kb)} & "
                    f"{_format_mus_cell(l_ns)} & {_format_mus_poisoned(l_ns, p_ns)} & "
                    f"{_format_mus_poisoned(l_ns, rand_ns)} & {_format_mus_poisoned(l_ns, radj_ns)}"
                )
                if include_construction_time:
                    cells += (
                        " & "
                        f"{_format_sec_cell(l_bt)} & {_format_sec_poisoned(l_bt, p_bt)} & "
                        f"{_format_sec_poisoned(l_bt, rand_bt)} & {_format_sec_poisoned(l_bt, radj_bt)}"
                    )
            else:
                cells = (
                    f"{_format_index_mb_cell(l_kb)} & {_format_index_mb_poisoned(l_kb, p_kb)} & "
                    f"{_format_mus_cell(l_ns)} & {_format_mus_poisoned(l_ns, p_ns)}"
                )
                if not pgm_only:
                    cells += (
                        f" & {_format_sec_cell(l_bt)} & {_format_sec_poisoned(l_bt, p_bt)}"
                    )
            if pgm_only:
                row = f"{ds_cell} & {cells}" r" \\"
            else:
                row = f"{ds_cell} & {index_label} & {cells}" r" \\"
            group_lines.append("    " + row)
        body_lines.append(group_lines)
        if not pgm_only:
            body_lines.append(["    " + r"\midrule"])

    # Drop trailing \midrule before \bottomrule
    flat: List[str] = []
    for i, g in enumerate(body_lines):
        if i == len(body_lines) - 1 and len(g) == 1 and g[0].strip() == r"\midrule":
            continue
        flat.extend(g)

    body = format_latex_real_system_table(
        flat,
        resolution_lines,
        print_label,
        epsilon,
        include_index_column=not pgm_only,
        use_multi_poison_columns=multi_poison,
        include_construction_time=include_construction_time,
    )
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
    ap = argparse.ArgumentParser(description="LaTeX real-system PGM / FITing-Tree / RadixSpline tables.")
    ap.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Duplicate-key datasets only; *_dup under fig/table_mopt_upper_bound_real_system_dup/.",
    )
    args = ap.parse_args()
    dup = args.duplicate_only
    out_dir = _REPO_ROOT / (
        "fig/table_mopt_upper_bound_real_system_dup"
        if dup
        else "fig/table_mopt_upper_bound_real_system"
    )
    for eps_full in REAL_SYSTEM_FULL_TABLE_EPSILONS:
        for variant in ("full",):
            main(
                variant,
                _out_path_with_dup(
                    out_dir
                    / f"mopt_upper_bound_table_real_system_eps{eps_full}_full.tex",
                    dup,
                ),
                _out_path_with_dup(
                    out_dir / f"print_table_real_system_eps{eps_full}_full.log", dup
                ),
                eps_full,
                duplicate_only=dup,
                use_multi_poison_columns=True,
                include_construction_time=False,
            )
    for eps in REAL_SYSTEM_TABLE_EPSILONS:
        for variant in ("full",):
            main(
                variant,
                _out_path_with_dup(
                    out_dir / f"mopt_upper_bound_table_real_system_eps{eps}.tex",
                    dup,
                ),
                _out_path_with_dup(
                    out_dir / f"print_table_real_system_eps{eps}.log", dup
                ),
                eps,
                duplicate_only=dup,
                pgm_only=True,
            )
