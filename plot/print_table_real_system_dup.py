"""
LaTeX table: real-system index size / query time for duplicate-key workloads only.

Same layout as print_table_mopt_upper_bound_real_system.py with --duplicate-only, but writes under
fig/table_real_system_dup/ without *_dup filename suffix (the directory is dup-only).

Outputs:
  mopt_real_system_table_eps{16,128}_full.tex — PGM, FITing-Tree, RadixSpline; Legit., Proposed,
    Random, Random-Adj.; no construction time.
  mopt_real_system_table_eps128.tex — PGM-index only.

Poison budget: REAL_SYSTEM_POISON_FRAC (=0.1) → λ from LAMBDA_TABLE_SPEC_FULL_DUPLICATES per dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

from print_table_mopt_upper_bound_real_system import (
    REAL_SYSTEM_FULL_TABLE_EPSILONS,
    REAL_SYSTEM_TABLE_EPSILONS,
    _REPO_ROOT,
    main,
)

_OUT_DIR = _REPO_ROOT / "fig/table_real_system_dup"


def _run() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for eps_full in REAL_SYSTEM_FULL_TABLE_EPSILONS:
        main(
            "full",
            _OUT_DIR / f"mopt_real_system_table_eps{eps_full}_full.tex",
            _OUT_DIR / f"print_table_real_system_eps{eps_full}_full.log",
            eps_full,
            duplicate_only=True,
            use_multi_poison_columns=True,
            include_construction_time=False,
        )
    for eps in REAL_SYSTEM_TABLE_EPSILONS:
        main(
            "full",
            _OUT_DIR / f"mopt_real_system_table_eps{eps}.tex",
            _OUT_DIR / f"print_table_real_system_eps{eps}.log",
            eps,
            duplicate_only=True,
            pgm_only=True,
        )


if __name__ == "__main__":
    _run()
    print(f"Wrote tables under {_OUT_DIR}", file=sys.stderr)
