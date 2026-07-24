"""Shared path/name parsing helpers for plot result loaders."""
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def extract_lambda_value(lambda_dir_name: str) -> int:
    """Extract lambda from directory name (e.g. 'lambda0' -> 0, 'lambda10000' -> 10000)."""
    match = re.match(r"lambda(\d+)", lambda_dir_name)
    if match:
        return int(match.group(1))
    return 0


def extract_lambda_per_segment_value(lambda_dir_name: str) -> Optional[int]:
    """Extract lambda_per_segment from name (e.g. 'lambda_per_segment8' -> 8)."""
    match = re.match(r"lambda_per_segment(\d+)", lambda_dir_name)
    if match:
        return int(match.group(1))
    return None


def extract_mu_value(mu_dir_name: str) -> float:
    """Extract mu from directory name (e.g. 'mu1' -> 1.0, 'mu2.5' -> 2.5)."""
    match = re.match(r"mu([\d.]+)", mu_dir_name)
    if match:
        return float(match.group(1))
    return 0.0


def seed_directory_sort_key(path: Path) -> int:
    """Sort key for seed{N} directories."""
    m = re.search(r"seed(\d+)", path.name)
    return int(m.group(1)) if m else 0


def build_method_seed_lambda_matrix(
    df: pd.DataFrame,
    methods: List[str],
    seeds: List[int],
    lambdas: List[int],
    value_col: str,
) -> Dict[str, Dict[int, Dict[int, Optional[float]]]]:
    """
    Nested dict {method: {seed: {lambda: value}}} for plotting.
    Requires columns: method, seed, lambda, value_col.
    """
    cache: Dict[str, Dict[int, Dict[int, Optional[float]]]] = {}
    for method in methods:
        cache[method] = {}
        for s in seeds:
            cache[method][s] = {}
            if df.empty or value_col not in df.columns:
                sub = pd.DataFrame()
            else:
                sub = df[(df["method"] == method) & (df["seed"] == s)]
            for _, row in sub.iterrows():
                lam = int(row["lambda"])
                if lam in lambdas:
                    cache[method][s][lam] = row[value_col]
            for lam in lambdas:
                if lam not in cache[method][s]:
                    cache[method][s][lam] = None
    return cache
