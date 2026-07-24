"""
Shared plot configuration: dataset display names, ordered dataset lists for multi-panel figures.

Import from sibling plot scripts (run with plot/ on sys.path, e.g. python plot/foo.py from repo root).
"""
from typing import Dict, List, Tuple

# dataset_id (directory / result folder name) -> short title for axis and subplot titles
DATASET_DISPLAY_NAMES: Dict[str, str] = {
    # SOSD 800M
    "books_800M": "Amzn",
    "fb_200M": "Face",
    "osm_cellids_800M": "Osmc",
    "wiki_ts_200M": "Wiki",
    # mgbench / alex / synthetic 200M-scale
    "bench2": "Weblogs",
    "bench3": "IoT",
    "lognormal_sigma1_range2pow63_200M": "Lognormal",
    "longitudes": "Longitudes",
    "longlat": "Longlat",
    "normal_mu0_sigma1_range2pow63_200M": "Normal",
    "uniform_range2pow63_200M": "Uniform",
    "ycsb": "YCSB",
    "zipf_s1_200M": "Zipf",
    # 1M (same ids as result dirs under *_swing_lambda_with_theta / upper_bound)
    "books_1M": "Amzn 1M",
    "fb_1M": "Face 1M",
    "osm_cellids_1M": "Osmc 1M",
    "wiki_ts_1M": "Wiki 1M",
    "bench2_1M": "Weblogs 1M",
    "bench3_1M": "IoT 1M",
    "uniform_range2pow63_1M": "Uniform 1M",
    "lognormal_sigma1_range2pow63_1M": "Lognormal 1M",
    "longitudes_1M": "Longitudes 1M",
    "longlat_1M": "Longlat 1M",
    "normal_mu0_sigma1_range2pow63_1M": "Normal 1M",
    "ycsb_1M": "YCSB 1M",
    "zipf_s1_1M": "Zipf 1M",
}


def dataset_display_name(dataset_id: str) -> str:
    """Return short display title for a dataset folder name (fallback: raw id)."""
    return DATASET_DISPLAY_NAMES.get(dataset_id, dataset_id)


# Order: SOSD Amzn → Osmc → Face first, then YCSB / geos / synthetics (key_rank / grids follow this)
DATASET_IDS_800M_SWING: List[str] = [
    # "bench2",
    # "bench3",
    "books_800M",
    "osm_cellids_800M",
    "fb_200M",
    # "wiki_ts_200M",
    "ycsb",
    "longitudes",
    "longlat",
    "uniform_range2pow63_200M",
    "normal_mu0_sigma1_range2pow63_200M",
    "lognormal_sigma1_range2pow63_200M",
    # "zipf_s1_200M",
]

# 1M datasets for swing_lambda, upper_bound, mu_to_mopt (same order as 800M swing)
DATASET_IDS_1M: List[str] = [
    # "bench2_1M",
    # "bench3_1M",
    "books_1M",
    "osm_cellids_1M",
    "fb_1M",
    # "wiki_ts_1M",
    "ycsb_1M",
    "longitudes_1M",
    "longlat_1M",
    "uniform_range2pow63_1M",
    "normal_mu0_sigma1_range2pow63_1M",
    "lognormal_sigma1_range2pow63_1M",
    # "zipf_s1_1M",
]

# Duplicate-key workloads (commented out of DATASET_IDS_* above). Order fixed for figures/tables.
DATASET_IDS_800M_DUPLICATES: List[str] = [
    "bench2",
    "bench3",
    "wiki_ts_200M",
    "zipf_s1_200M",
]

DATASET_IDS_1M_DUPLICATES: List[str] = [
    "bench2_1M",
    "bench3_1M",
    "wiki_ts_1M",
    "zipf_s1_1M",
]

# Full-scale key counts n (for λ = 0.01·n and 0.1·n).
DUPLICATE_FULL_N: Dict[str, int] = {
    "wiki_ts_200M": 200_000_000,
    "bench2": 75_748_120,
    "bench3": 108_957_040,
    "zipf_s1_200M": 200_000_000,
}

# 1M-scale experiments: assume n = 1_000_000 unless overridden.
DUPLICATE_1M_N: Dict[str, int] = {ds: 1_000_000 for ds in DATASET_IDS_1M_DUPLICATES}


def _lambda_table_from_n(n_by_ds: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """Nested { '0.01': {ds: int}, '0.1': {ds: int} } with integer λ = frac·n."""
    out: Dict[str, Dict[str, int]] = {"0.01": {}, "0.1": {}}
    for ds, n in n_by_ds.items():
        out["0.01"][ds] = int(round(0.01 * n))
        out["0.1"][ds] = int(round(0.1 * n))
    return out


LAMBDA_TABLE_SPEC_FULL_DUPLICATES: Dict[str, Dict[str, int]] = _lambda_table_from_n(
    DUPLICATE_FULL_N
)

LAMBDA_TABLE_SPEC_1M_DUPLICATES: Dict[str, Dict[str, int]] = _lambda_table_from_n(
    DUPLICATE_1M_N
)

# Single-frac tables (e.g. various_epsilon at λ = 0.1·n only).
LAMBDA_TABLE_SPEC_VARIOUS_EPS_FULL_DUPLICATES: Dict[str, int] = {
    ds: LAMBDA_TABLE_SPEC_FULL_DUPLICATES["0.1"][ds] for ds in DATASET_IDS_800M_DUPLICATES
}

LAMBDA_TABLE_SPEC_VARIOUS_EPS_1M_DUPLICATES: Dict[str, int] = {
    ds: LAMBDA_TABLE_SPEC_1M_DUPLICATES["0.1"][ds] for ds in DATASET_IDS_1M_DUPLICATES
}


def upper_bound_dataset_specs_duplicate_only() -> List[Dict[str, str]]:
    """1M duplicate ids; path == name."""
    return [{"name": s, "path": s} for s in DATASET_IDS_1M_DUPLICATES]


def upper_bound_dataset_specs_full_duplicate_only() -> List[Dict[str, str]]:
    """Full-scale duplicate ids; path == name."""
    return [{"name": s, "path": s} for s in DATASET_IDS_800M_DUPLICATES]


def swing_dataset_specs_800m_duplicate(base_dir: str) -> List[Dict[str, str]]:
    """800M-scale duplicate workloads for swing plots."""
    b = base_dir.rstrip("/")
    return [{"name": s, "path": f"{b}/{s}"} for s in DATASET_IDS_800M_DUPLICATES]


def swing_dataset_specs_1m_duplicate(base_dir: str) -> List[Dict[str, str]]:
    """1M duplicate workloads for swing plots."""
    b = base_dir.rstrip("/")
    return [{"name": s, "path": f"{b}/{s}"} for s in DATASET_IDS_1M_DUPLICATES]


def key_rank_specs_duplicate_only() -> List[Tuple[str, str, str]]:
    """
    KEY_RANK_SPECS entries for DATASET_IDS_800M_DUPLICATES only, same order as that list.
    """
    by_id: Dict[str, Tuple[str, str, str]] = {}
    for title, path, inset in KEY_RANK_SPECS:
        base = path.rsplit("/", 1)[-1]
        if base.endswith("_uint64"):
            ds_id = base[: -len("_uint64")]
        else:
            ds_id = base
        by_id[ds_id] = (title, path, inset)
    out: List[Tuple[str, str, str]] = []
    for ds_id in DATASET_IDS_800M_DUPLICATES:
        spec = by_id.get(ds_id)
        if spec is not None:
            out.append(spec)
    return out


def swing_dataset_specs_800m(base_dir: str) -> List[Dict[str, str]]:
    """[{"name": dataset_id, "path": base_dir/dataset_id}, ...] for 800M swing plots."""
    b = base_dir.rstrip("/")
    return [{"name": s, "path": f"{b}/{s}"} for s in DATASET_IDS_800M_SWING]


def swing_dataset_specs_1m(base_dir: str) -> List[Dict[str, str]]:
    """[{"name": dataset_id, "path": base_dir/dataset_id}, ...] for 1M swing plots."""
    b = base_dir.rstrip("/")
    return [{"name": s, "path": f"{b}/{s}"} for s in DATASET_IDS_1M]


def upper_bound_dataset_specs() -> List[Dict[str, str]]:
    """Upper_bound layout uses path == name under ub_base_dir / swing_base_dir."""
    return [{"name": s, "path": s} for s in DATASET_IDS_1M]


def upper_bound_dataset_specs_full() -> List[Dict[str, str]]:
    """Full-scale (800M / 200M) dataset ids for upper_bound vs swing; path == name like 1M specs."""
    return [{"name": s, "path": s} for s in DATASET_IDS_800M_SWING]


def key_rank_specs_swing_800m() -> List[Tuple[str, str, str]]:
    """
    Subset of KEY_RANK_SPECS: only datasets in DATASET_IDS_800M_SWING, same order.
    Paths are data/<dataset_id>_uint64 as in KEY_RANK_SPECS.
    """
    by_id: Dict[str, Tuple[str, str, str]] = {}
    for title, path, inset in KEY_RANK_SPECS:
        base = path.rsplit("/", 1)[-1]
        if base.endswith("_uint64"):
            ds_id = base[: -len("_uint64")]
        else:
            ds_id = base
        by_id[ds_id] = (title, path, inset)
    out: List[Tuple[str, str, str]] = []
    for ds_id in DATASET_IDS_800M_SWING:
        spec = by_id.get(ds_id)
        if spec is not None:
            out.append(spec)
    return out


# Key-rank CDF grid: (title, data path under project root, inset_loc)
KEY_RANK_SPECS: List[Tuple[str, str, str]] = [
    ("Weblogs", "data/bench2_uint64", "lower right"),
    ("IoT", "data/bench3_uint64", "upper left"),
    ("Longitudes", "data/longitudes_uint64", "lower right"),
    ("Longlat", "data/longlat_uint64", "lower right"),
    ("YCSB", "data/ycsb_uint64", "lower right"),
    ("Amzn", "data/books_800M_uint64", "lower right"),
    ("Face", "data/fb_200M_uint64", "lower right"),
    ("Osmc", "data/osm_cellids_800M_uint64", "lower right"),
    ("Wiki", "data/wiki_ts_200M_uint64", "upper left"),
    ("Zipf", "data/zipf_s1_200M_uint64", "lower right"),
    ("Uniform", "data/uniform_range2pow63_200M_uint64", "lower right"),
    ("Lognormal", "data/lognormal_sigma1_range2pow63_200M_uint64", "lower right"),
    ("Normal", "data/normal_mu0_sigma1_range2pow63_200M_uint64", "lower right"),
]
