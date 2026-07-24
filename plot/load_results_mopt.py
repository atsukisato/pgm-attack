"""
Load m_opt / swing-lambda benchmark results (module: load_results_mopt).

Data under inject_poisons_to_minimize_segment_length_swing_lambda_with_theta.

Directory structure:
  base_dir/{dataset}/lambda{N}/epsilon{E}/mu{M}/benchmark_pgm.json
  1M: base_dir/{dataset}/seed{S}/lambda{N}/epsilon{E}/mu{M}/benchmark_pgm.json
  Full-scale inject_poisons_*_random (run_experiment): same with optional
  .../mu{M}/seed{S}/benchmark_pgm.json (use load_swing_lambda_benchmark_results(..., leaf_seed=S)).

Two views:
  - μ-scan: fixed (dataset, λ, ε), all mu subdirs → m_opt mean/std (see load_mu_to_mopt_*).
  - λ-scan: fixed (dataset, ε, optional μ), one row per λ with PGM + poisons metrics
    (see load_swing_lambda_benchmark_results).
"""
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from load_results_common import extract_lambda_value, extract_mu_value, seed_directory_sort_key


def _read_first_level_mopt_from_benchmark_pgm(benchmark_file: Path) -> Optional[int]:
    """segments_per_level[0] from benchmark_pgm.json, or None."""
    try:
        with open(benchmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments_per_level", [])
        return segments[0] if segments else None
    except (json.JSONDecodeError, Exception):
        return None


# --- μ vs m_opt (scan all mu under lambda{λ}/epsilon{ε}) ---


def _load_mu_to_mopt_from_path(path: Path) -> List[Tuple[float, int]]:
    """Load (mu, m_opt) from a path containing mu subdirs."""
    if not path.exists():
        return []
    results = []
    for mu_dir in sorted(path.iterdir(), key=lambda d: extract_mu_value(d.name)):
        if not mu_dir.is_dir() or not mu_dir.name.startswith("mu"):
            continue
        benchmark_file = mu_dir / "benchmark_pgm.json"
        if not benchmark_file.exists():
            continue
        m_opt = _read_first_level_mopt_from_benchmark_pgm(benchmark_file)
        if m_opt is not None:
            mu_val = extract_mu_value(mu_dir.name)
            results.append((mu_val, m_opt))
    return sorted(results, key=lambda x: x[0])


def load_mu_to_mopt_results(
    base_dir: str,
    dataset: str,
    lambda_val: int,
    epsilon: int,
    aggregate_1M: bool = True,
) -> pd.DataFrame:
    """
    Long-form rows: dataset, lambda, epsilon, mu, m_opt_mean, m_opt_std.
    m_opt_std is NaN when a single seed or non-aggregated run.
    """
    dataset_path = Path(base_dir) / dataset
    if not dataset_path.exists():
        return pd.DataFrame(
            columns=["dataset", "lambda", "epsilon", "mu", "m_opt_mean", "m_opt_std"]
        )

    seed_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("seed")],
        key=seed_directory_sort_key,
    )

    rows = []
    if aggregate_1M and seed_dirs:
        mu_to_mopts: Dict[float, List[int]] = {}
        for seed_dir in seed_dirs:
            path = seed_dir / f"lambda{lambda_val}" / f"epsilon{epsilon}"
            for mu_val, m_opt in _load_mu_to_mopt_from_path(path):
                mu_to_mopts.setdefault(mu_val, []).append(m_opt)
        for mu_val in sorted(mu_to_mopts.keys()):
            vals = mu_to_mopts[mu_val]
            mean_m_opt = statistics.mean(vals)
            std_m_opt = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "lambda": lambda_val,
                    "epsilon": epsilon,
                    "mu": mu_val,
                    "m_opt_mean": mean_m_opt,
                    "m_opt_std": std_m_opt,
                }
            )
    else:
        path = dataset_path / f"lambda{lambda_val}" / f"epsilon{epsilon}"
        for mu_val, m_opt in _load_mu_to_mopt_from_path(path):
            rows.append(
                {
                    "dataset": dataset,
                    "lambda": lambda_val,
                    "epsilon": epsilon,
                    "mu": mu_val,
                    "m_opt_mean": float(m_opt),
                    "m_opt_std": float("nan"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["dataset", "lambda", "epsilon", "mu", "m_opt_mean", "m_opt_std"]
        )
    return pd.DataFrame(rows)


def load_mu_to_mopt_data(
    base_dir: str,
    dataset: str,
    lambda_val: int,
    epsilon: int,
    aggregate_1M: bool = True,
) -> List[Tuple[float, float, Optional[float]]]:
    """Backward-compatible (mu, m_opt_mean, m_opt_std); std is None when unavailable."""
    df = load_mu_to_mopt_results(base_dir, dataset, lambda_val, epsilon, aggregate_1M)
    out: List[Tuple[float, float, Optional[float]]] = []
    for _, row in df.iterrows():
        std = row["m_opt_std"]
        if pd.isna(std):
            std = None
        else:
            std = float(std)
        out.append((float(row["mu"]), float(row["m_opt_mean"]), std))
    return out


def load_lambda0_baseline_df(
    base_dir: str,
    dataset: str,
    epsilon: int,
    aggregate_1M: bool = True,
) -> pd.DataFrame:
    """Single-row DataFrame: dataset, epsilon, m_opt_mean, m_opt_std (std may be NaN)."""
    mean, std = load_lambda0_baseline(base_dir, dataset, epsilon, aggregate_1M)
    if mean is None:
        return pd.DataFrame(
            columns=["dataset", "epsilon", "m_opt_mean", "m_opt_std"]
        )
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "epsilon": epsilon,
                "m_opt_mean": mean,
                "m_opt_std": std if std is not None else float("nan"),
            }
        ]
    )


def load_lambda0_baseline(
    base_dir: str,
    dataset: str,
    epsilon: int,
    aggregate_1M: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """
    m_opt baseline for lambda=0 (no poisoning). Uses mu1 when present.
    Returns (mean, std) — std is None for a single run.
    """
    dataset_path = Path(base_dir) / dataset
    if not dataset_path.exists():
        return None, None

    seed_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("seed")],
        key=seed_directory_sort_key,
    )

    if aggregate_1M and seed_dirs:
        m_opts = []
        for seed_dir in seed_dirs:
            path = seed_dir / "lambda0" / f"epsilon{epsilon}" / "mu1"
            bf = path / "benchmark_pgm.json"
            if not bf.exists():
                path = seed_dir / "lambda0" / f"epsilon{epsilon}"
                for mu_dir in path.iterdir() if path.exists() else []:
                    if mu_dir.is_dir():
                        bf = mu_dir / "benchmark_pgm.json"
                        if bf.exists():
                            break
            if bf.exists():
                v = _read_first_level_mopt_from_benchmark_pgm(bf)
                if v is not None:
                    m_opts.append(v)
        if m_opts:
            return statistics.mean(m_opts), statistics.stdev(m_opts) if len(m_opts) > 1 else None
        return None, None

    path = dataset_path / "lambda0" / f"epsilon{epsilon}" / "mu1"
    bf = path / "benchmark_pgm.json"
    if not bf.exists():
        path = dataset_path / "lambda0" / f"epsilon{epsilon}"
        if path.exists():
            for mu_dir in path.iterdir():
                if mu_dir.is_dir():
                    bf = mu_dir / "benchmark_pgm.json"
                    if bf.exists():
                        break
    if not bf.exists():
        return None, None
    v = _read_first_level_mopt_from_benchmark_pgm(bf)
    return (float(v), None) if v is not None else (None, None)


def discover_datasets(base_dir: str) -> List[str]:
    """Discover dataset names from base directory."""
    path = Path(base_dir)
    if not path.exists():
        return []
    return sorted([d.name for d in path.iterdir() if d.is_dir()])


def discover_epsilons(base_dir: str, dataset: str, lambda_val: int) -> List[int]:
    """Discover epsilon values for given dataset and lambda."""
    dataset_path = Path(base_dir) / dataset
    if not dataset_path.exists():
        return []
    seed_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("seed")],
        key=seed_directory_sort_key,
    )
    search_path = seed_dirs[0] / f"lambda{lambda_val}" if seed_dirs else dataset_path / f"lambda{lambda_val}"
    if not search_path.exists():
        return []
    epsilons = []
    for d in search_path.iterdir():
        if d.is_dir():
            m = re.match(r"epsilon(\d+)", d.name)
            if m:
                epsilons.append(int(m.group(1)))
    return sorted(epsilons)


# --- λ vs PGM metrics (single μ per λ, optional poisons_info) ---


def _infer_dataset_name(results_dir_path: Path) -> str:
    if results_dir_path.name.startswith("seed"):
        return results_dir_path.parent.name
    return results_dir_path.name


def _find_mu_dir(epsilon_path: Path, mu: Optional[float] = None) -> Optional[Path]:
    """Find mu directory: explicit mu, else smallest available."""
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


def _resolve_benchmark_container_dir(
    mu_dir: Path, leaf_seed: Optional[int]
) -> Optional[Path]:
    """
    Directory that holds benchmark_pgm.json for this (lambda, epsilon, mu).

    Full-scale inject_poisons_*_random runs use .../mu{M}/seed{S}/benchmark_pgm.json;
    swing / 1M layouts use .../mu{M}/benchmark_pgm.json directly.
    """
    if leaf_seed is not None:
        seed_dir = mu_dir / f"seed{leaf_seed}"
        if (seed_dir / "benchmark_pgm.json").exists():
            return seed_dir
    if (mu_dir / "benchmark_pgm.json").exists():
        return mu_dir
    return None


def _load_single_seed(
    results_dir_path: Path,
    lambda_dir: Path,
    epsilon: int,
    mu: Optional[float],
    leaf_seed: Optional[int] = None,
) -> Optional[Dict]:
    """Load one (lambda, epsilon, mu) row from a single run."""
    lambda_value = extract_lambda_value(lambda_dir.name)
    epsilon_path = lambda_dir / f"epsilon{epsilon}"
    mu_dir = _find_mu_dir(epsilon_path, mu)
    if mu_dir is None:
        return None

    container = _resolve_benchmark_container_dir(mu_dir, leaf_seed)
    if container is None:
        return None

    benchmark_file = container / "benchmark_pgm.json"
    poisons_info_file = container / "poisons_info.json"

    try:
        with open(benchmark_file, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

        segments_per_level = benchmark_data.get("segments_per_level", [])
        first_level_segments_num = segments_per_level[0] if segments_per_level else None

        result = {
            "lambda": lambda_value,
            "epsilon": benchmark_data.get("epsilon"),
            "build_time_sec": benchmark_data.get("build_time_sec"),
            "index_size_in_kb": benchmark_data.get("index_size_in_kb"),
            "levels": benchmark_data.get("levels"),
            "first_level_segments_num": first_level_segments_num,
            "median_avg_query_time_ns": benchmark_data.get("query_time_summary", {}).get(
                "median_avg_query_time_ns"
            ),
        }

        if poisons_info_file.exists():
            try:
                with open(poisons_info_file, "r", encoding="utf-8") as f:
                    poisons_info = json.load(f)

                result["num_poisons_generated"] = poisons_info.get("num_poisons_generated")
                result["m_opt_after_poisoning"] = poisons_info.get("m_opt_after_poisoning")
                result["generation_time_sec"] = poisons_info.get("generation_time_sec")
            except (json.JSONDecodeError, Exception):
                pass

        return result
    except (json.JSONDecodeError, Exception):
        return None


def load_swing_lambda_benchmark_results(
    results_dir: str,
    epsilon: int = 64,
    mu: Optional[float] = None,
    aggregate_1M: bool = True,
    leaf_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Benchmark / poisons metrics per lambda (optionally aggregated across seeds for 1M).

    Columns include dataset, lambda, epsilon, and metric columns; aggregated runs add *_std and n_seeds as 'n'.

    leaf_seed: when aggregate_1M is False, if set, look for benchmark under mu*/seed{leaf_seed}/
    (full-scale random inject layout) before mu*/.
    """
    results_dir_path = Path(results_dir)
    dataset_name = _infer_dataset_name(results_dir_path)
    if not results_dir_path.exists():
        return pd.DataFrame()

    seed_dirs = sorted(
        [d for d in results_dir_path.iterdir() if d.is_dir() and d.name.startswith("seed")],
        key=seed_directory_sort_key,
    )

    if aggregate_1M and seed_dirs:
        METRIC_KEYS = [
            "build_time_sec",
            "index_size_in_kb",
            "levels",
            "first_level_segments_num",
            "median_avg_query_time_ns",
            "num_poisons_generated",
            "m_opt_after_poisoning",
            "generation_time_sec",
        ]
        lambda_by_key: Dict[int, Dict[str, List[float]]] = {}

        for seed_dir in seed_dirs:
            for lambda_dir in seed_dir.iterdir():
                if not lambda_dir.is_dir() or not lambda_dir.name.startswith("lambda"):
                    continue
                result = _load_single_seed(seed_dir, lambda_dir, epsilon, mu)
                if result is None:
                    continue

                lam = result["lambda"]
                if lam not in lambda_by_key:
                    lambda_by_key[lam] = {k: [] for k in METRIC_KEYS}
                for k in METRIC_KEYS:
                    v = result.get(k)
                    if v is not None:
                        lambda_by_key[lam][k].append(v)

        rows = []
        for lam in sorted(lambda_by_key.keys()):
            row = {"dataset": dataset_name, "lambda": lam, "epsilon": epsilon}
            n_seeds = 0
            for k in METRIC_KEYS:
                vals = lambda_by_key[lam][k]
                if vals:
                    n_seeds = len(vals)
                    row[k] = statistics.mean(vals)
                    row[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row["n"] = n_seeds
            rows.append(row)
        return pd.DataFrame(rows)

    rows = []
    for lambda_dir in sorted(results_dir_path.iterdir()):
        if not lambda_dir.is_dir() or not lambda_dir.name.startswith("lambda"):
            continue
        result = _load_single_seed(
            results_dir_path, lambda_dir, epsilon, mu, leaf_seed=leaf_seed
        )
        if result is not None:
            result["dataset"] = dataset_name
            result["n"] = 1
            rows.append(result)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(by=["lambda"], kind="mergesort").reset_index(drop=True)
