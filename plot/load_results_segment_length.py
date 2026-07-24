"""Load segment-length poisoning results as long-form DataFrames."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from load_results_common import build_method_seed_lambda_matrix, extract_lambda_per_segment_value


def _segment_length_rows(
    results_dir: str,
    method: str,
    dataset: str,
    epsilon: int,
    seed: Optional[int],
    intercept_candidate_num: Optional[int],
    allow_intercept_fallback: bool,
) -> List[dict]:
    """Build row dicts (used with lru_cache via immutable key)."""
    base_path = Path(results_dir) / method / dataset / f"epsilon{epsilon}"
    if not base_path.exists():
        return []

    rows = []
    lambda_zero_values: Dict[int, float] = {}

    for lambda_dir in sorted(base_path.iterdir()):
        if not lambda_dir.is_dir():
            continue

        lambda_value = extract_lambda_per_segment_value(lambda_dir.name)
        if lambda_value is None:
            continue

        if intercept_candidate_num is not None:
            intercept_dir = lambda_dir / f"intercept_candidate_num{intercept_candidate_num}"
            if intercept_dir.exists():
                search_dir = intercept_dir
            elif allow_intercept_fallback:
                search_dir = lambda_dir
            else:
                continue
        else:
            search_dir = lambda_dir

        if seed is not None:
            seed_dirs = [search_dir / f"seed{seed}"]
        else:
            seed_dirs = [d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("seed")]

        for seed_dir in sorted(
            seed_dirs, key=lambda p: int(p.name[4:]) if p.name.startswith("seed") else 0
        ):
            seed_value = int(seed_dir.name[4:]) if seed_dir.name.startswith("seed") else None
            poisons_info_file = seed_dir / "poisons_info.json"

            if not poisons_info_file.exists():
                continue

            try:
                with open(poisons_info_file, "r", encoding="utf-8") as f:
                    poisons_info = json.load(f)

                covered_after = poisons_info.get("covered_keys_after_attack")
                if covered_after is None:
                    fl = poisons_info.get("final_segment_length")
                    npg = poisons_info.get("num_poisons_generated", 0)
                    if fl is not None:
                        covered_after = max(0, fl - npg)
                if covered_after is None:
                    continue

                num_poisons_generated = poisons_info.get("num_poisons_generated", 0)
                generation_time_sec = poisons_info.get("generation_time_sec")

                covered_before = poisons_info.get("covered_keys_before_attack") or poisons_info.get(
                    "initial_segment_length"
                )
                if (
                    covered_before is not None
                    and seed_value is not None
                    and seed_value not in lambda_zero_values
                ):
                    lambda_zero_values[seed_value] = covered_before

                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "epsilon": epsilon,
                        "seed": seed_value,
                        "lambda": lambda_value,
                        "covered_keys_after_attack": covered_after,
                        "covered_keys_before_attack": covered_before,
                        "num_poisons_generated": num_poisons_generated,
                        "generation_time_sec": generation_time_sec,
                        "intercept_candidate_num": intercept_candidate_num,
                    }
                )
            except (json.JSONDecodeError, Exception):
                continue

    for seed_val, init_seg_len in lambda_zero_values.items():
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "epsilon": epsilon,
                "seed": seed_val,
                "lambda": 0,
                "covered_keys_after_attack": init_seg_len,
                "covered_keys_before_attack": init_seg_len,
                "num_poisons_generated": 0,
                "generation_time_sec": None,
                "intercept_candidate_num": intercept_candidate_num,
            }
        )

    rows.sort(key=lambda x: (x["lambda"], x.get("seed", 0)))
    return rows


@lru_cache(maxsize=256)
def _segment_length_rows_cached(
    results_dir: str,
    method: str,
    dataset: str,
    epsilon: int,
    seed: Optional[int],
    intercept_candidate_num: Optional[int],
    allow_intercept_fallback: bool,
) -> tuple:
    """Cache tuple of row dicts (hashable args)."""
    rows = _segment_length_rows(
        results_dir, method, dataset, epsilon, seed, intercept_candidate_num, allow_intercept_fallback
    )
    return tuple(tuple(sorted(r.items())) for r in rows)


def load_segment_length_results(
    results_dir: str,
    method: str,
    dataset: str,
    epsilon: int,
    seed: Optional[int] = None,
    intercept_candidate_num: Optional[int] = None,
    allow_intercept_fallback: bool = True,
) -> pd.DataFrame:
    """
    Load segment length rows from poisoning_segment_length results.

    Returns columns including: dataset, method, epsilon, seed, lambda,
    covered_keys_after_attack, covered_keys_before_attack, num_poisons_generated,
    generation_time_sec, intercept_candidate_num.
    """
    cached = _segment_length_rows_cached(
        results_dir,
        method,
        dataset,
        epsilon,
        seed,
        intercept_candidate_num,
        allow_intercept_fallback,
    )
    rows = [dict(t) for t in cached]
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "epsilon",
                "seed",
                "lambda",
                "covered_keys_after_attack",
                "covered_keys_before_attack",
                "num_poisons_generated",
                "generation_time_sec",
                "intercept_candidate_num",
            ]
        )
    return pd.DataFrame(rows)


def load_all_segment_lengths(
    results_dir: str,
    methods: List[str],
    dataset: str,
    epsilon: int,
    seeds: List[int],
    lambdas: List[int],
    intercept_candidate_num: Optional[int] = None,
    allow_intercept_fallback: bool = True,
) -> Dict[str, Dict[int, Dict[int, Optional[float]]]]:
    """Nested dict for covered_keys_after_attack (same shape as max_error cache)."""
    dfs = []
    for method in methods:
        dfs.append(
            load_segment_length_results(
                results_dir,
                method,
                dataset,
                epsilon,
                seed=None,
                intercept_candidate_num=intercept_candidate_num,
                allow_intercept_fallback=allow_intercept_fallback,
            )
        )
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        return build_method_seed_lambda_matrix(
            df, methods, seeds, lambdas, "covered_keys_after_attack"
        )
    df = df[df["seed"].isin(seeds)]
    return build_method_seed_lambda_matrix(
        df, methods, seeds, lambdas, "covered_keys_after_attack"
    )
