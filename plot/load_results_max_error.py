"""Load max_error poisoning results as long-form DataFrames."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from load_results_common import build_method_seed_lambda_matrix, extract_lambda_value


def load_max_error_results(
    results_dir: str,
    method: str,
    dataset: str,
    epsilon: int = 64,
    n: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load max_error rows from maximize_maxerror results.

    Returns columns: dataset, method, epsilon, n, seed, lambda, max_error, generation_time_sec
    """
    base_path = Path(results_dir) / f"maximize_maxerror_{method}" / dataset
    if not base_path.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "epsilon",
                "n",
                "seed",
                "lambda",
                "max_error",
                "generation_time_sec",
            ]
        )

    rows = []
    for lambda_dir in sorted(base_path.iterdir()):
        if not lambda_dir.is_dir():
            continue

        lambda_value = extract_lambda_value(lambda_dir.name)
        epsilon_dir = lambda_dir / f"epsilon{epsilon}"

        if not epsilon_dir.exists():
            continue

        if n is not None:
            n_dirs = [epsilon_dir / f"n{n}"]
        else:
            n_dirs = [d for d in epsilon_dir.iterdir() if d.is_dir() and d.name.startswith("n")]

        for n_dir in n_dirs:
            if not n_dir.exists():
                continue

            n_value = int(n_dir.name[1:]) if n_dir.name.startswith("n") else None

            if seed is not None:
                seed_dirs = [n_dir / f"seed{seed}"]
            else:
                seed_dirs = [d for d in n_dir.iterdir() if d.is_dir() and d.name.startswith("seed")]

            for seed_dir in seed_dirs:
                seed_value = int(seed_dir.name[4:]) if seed_dir.name.startswith("seed") else None

                poisons_info_file = seed_dir / "poisons_info.json"

                if not poisons_info_file.exists():
                    continue

                try:
                    with open(poisons_info_file, "r", encoding="utf-8") as f:
                        poisons_info = json.load(f)

                    max_error = poisons_info.get("max_error")
                    if max_error is None:
                        continue

                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "epsilon": epsilon,
                            "n": n_value,
                            "seed": seed_value,
                            "lambda": lambda_value,
                            "max_error": max_error,
                            "generation_time_sec": poisons_info.get("generation_time_sec"),
                        }
                    )

                except (json.JSONDecodeError, Exception):
                    continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "epsilon",
                "n",
                "seed",
                "lambda",
                "max_error",
                "generation_time_sec",
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(
        by=["lambda", "n", "seed"], kind="mergesort"
    ).reset_index(drop=True)


def load_all_max_errors(
    results_dir: str,
    methods: List[str],
    dataset: str,
    epsilon: int,
    n: int,
    seeds: List[int],
    lambdas: List[int],
) -> Dict[str, Dict[int, Dict[int, Optional[float]]]]:
    """
    Load max_error for multiple methods; returns nested dict for fast plot lookup.
    """
    dfs = []
    for method in methods:
        dfs.append(
            load_max_error_results(results_dir, method, dataset, epsilon, n=n, seed=None)
        )
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        return build_method_seed_lambda_matrix(df, methods, seeds, lambdas, "max_error")
    df = df[df["seed"].isin(seeds)]
    return build_method_seed_lambda_matrix(df, methods, seeds, lambdas, "max_error")


def load_all_generation_times(
    results_dir: str,
    methods: List[str],
    dataset: str,
    epsilon: int,
    n: int,
    seeds: List[int],
    lambdas: List[int],
) -> Dict[str, Dict[int, Dict[int, Optional[float]]]]:
    """Same layout as load_all_max_errors but values are generation_time_sec from poisons_info."""
    dfs = []
    for method in methods:
        dfs.append(
            load_max_error_results(results_dir, method, dataset, epsilon, n=n, seed=None)
        )
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        return build_method_seed_lambda_matrix(
            df, methods, seeds, lambdas, "generation_time_sec"
        )
    df = df[df["seed"].isin(seeds)]
    return build_method_seed_lambda_matrix(
        df, methods, seeds, lambdas, "generation_time_sec"
    )
