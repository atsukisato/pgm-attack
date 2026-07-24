"""Load m_opt upper bound (fix_w_per_block) results as a long-form DataFrame."""
import json
from pathlib import Path

import pandas as pd

from load_results_common import extract_lambda_value


def load_upper_bound_results(
    results_dir: str,
    epsilon: int = 64,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Load upper bound data from fix_w_per_block results.

    results_dir: path to dataset dir, e.g. results/upper_bound/fix_w_per_block/books_1M
                 or .../books_1M/seed0

    Returns columns: dataset, epsilon, seed, lambda, m_opt_upper_bound,
    computation_time_sec (from upper_bound.json when present)
    """
    results_dir_path = Path(results_dir)
    if not results_dir_path.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "epsilon",
                "seed",
                "lambda",
                "m_opt_upper_bound",
                "computation_time_sec",
            ]
        )

    seed_dir = results_dir_path / f"seed{seed}"
    if seed_dir.exists():
        scan_path = seed_dir
    else:
        scan_path = results_dir_path

    if scan_path.name.startswith("seed"):
        dataset_name = scan_path.parent.name
    else:
        dataset_name = scan_path.name

    rows = []
    for lambda_dir in sorted(scan_path.iterdir()):
        if not lambda_dir.is_dir() or not lambda_dir.name.startswith("lambda"):
            continue
        lambda_value = extract_lambda_value(lambda_dir.name)
        ub_file = lambda_dir / f"epsilon{epsilon}" / "upper_bound.json"
        if not ub_file.exists():
            continue
        try:
            with open(ub_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            ub = data.get("m_opt_upper_bound")
            if ub is not None:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "epsilon": epsilon,
                        "seed": seed,
                        "lambda": lambda_value,
                        "m_opt_upper_bound": ub,
                        "computation_time_sec": data.get("computation_time_sec"),
                    }
                )
        except (json.JSONDecodeError, Exception):
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "epsilon",
                "seed",
                "lambda",
                "m_opt_upper_bound",
                "computation_time_sec",
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by=["lambda"], kind="mergesort").reset_index(drop=True)
