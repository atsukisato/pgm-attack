#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from plot_config import key_rank_specs_duplicate_only, key_rank_specs_swing_800m
from plot_lambda_to_max_error import (
    FONTSIZE_TEXT,
    FONTSIZE_TICKS,
    FONTSIZE_TITLE,
    FONTSIZE_XLABEL,
    FONTSIZE_YLABEL,
    PLOT_LINEWIDTH,
    TEXT_COLOR,
)
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ----------------------------
# Binary format helpers
# ----------------------------

def read_count_u64(path: str) -> int:
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise RuntimeError(f"Failed to read count header: {path}")
        (n,) = struct.unpack("<Q", head)
    return int(n)

def read_u64_at_indices(path: str, indices: np.ndarray) -> np.ndarray:
    """
    Read uint64 values at 0-indexed positions 'indices' from file:
      [8B count][n * 8B data]
    Efficiently batches consecutive runs to reduce seeks.
    """
    if indices.ndim != 1:
        indices = indices.reshape(-1)
    if len(indices) == 0:
        return np.empty((0,), dtype=np.uint64)

    idx = np.asarray(indices, dtype=np.int64)
    if np.any(idx < 0):
        raise ValueError("indices contain negative values")

    order = np.argsort(idx, kind="mergesort")
    idx_sorted = idx[order]

    out_sorted = np.empty((len(idx_sorted),), dtype=np.uint64)

    # Detect runs of consecutive indices
    # runs: [start_pos_in_idx_sorted, end_pos_exclusive)
    run_starts = [0]
    diffs = np.diff(idx_sorted)
    cut = np.where(diffs != 1)[0] + 1
    run_starts.extend(cut.tolist())
    run_starts.append(len(idx_sorted))

    with open(path, "rb") as f:
        base = 8  # skip count header
        for a, b in zip(run_starts[:-1], run_starts[1:]):
            run = idx_sorted[a:b]
            start_i = int(run[0])
            count = int(len(run))
            offset = base + start_i * 8
            f.seek(offset, os.SEEK_SET)
            buf = f.read(count * 8)
            if len(buf) != count * 8:
                raise RuntimeError(f"Short read: {path} at index {start_i} count {count}")
            out_sorted[a:b] = np.frombuffer(buf, dtype="<u8", count=count)

    # Unsort back to original order
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out

def sample_u64(path: str, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    try:
        n = read_count_u64(path)
    except FileNotFoundError:
        return np.empty((0,), dtype=np.uint64)

    if n == 0:
        return np.empty((0,), dtype=np.uint64)

    s = min(sample_size, n)

    # Choose indices without replacement when possible (better coverage).
    # For very large n, numpy's choice is fine; memory is O(s).
    idx = rng.choice(n, size=s, replace=False)
    vals = read_u64_at_indices(path, idx)
    return vals


# ----------------------------
# Plotting
# ----------------------------

@dataclass
class DatasetSpec:
    title: str
    path: str
    inset_loc: str = "lower right"  # "upper left" or "lower right"

def approx_cdf_from_sample(vals_u64: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) for plotting CDF:
      x = normalized key in [0,1] using sample min/max
      y = position in [0,1] (empirical CDF)
    """
    if len(vals_u64) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    v = vals_u64.astype(np.float64, copy=False)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax == vmin:
        x = np.zeros_like(v, dtype=np.float64)
    else:
        x = (v - vmin) / (vmax - vmin)

    x.sort()  # sort normalized keys
    y = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    return x, y

def read_consecutive_u64(path: str, start_idx: int, count: int) -> np.ndarray:
    """Read consecutive keys from file at indices [start_idx, start_idx+count)."""
    if count <= 0:
        return np.empty((0,), dtype=np.uint64)
    indices = np.arange(start_idx, start_idx + count, dtype=np.int64)
    return read_u64_at_indices(path, indices)


ALL_KEY_COUNT = 2_000
INSET_KEY_COUNT = 100


def add_inset(ax, path: str, n: int, vmin: float, vmax: float,
              q: float = 0.8, inset_loc: str = "lower right"):
    """
    Create inset: select start position by q, read consecutive keys, plot in inset.
    """
    if n == 0:
        return

    # Start position from q
    start_idx = int(q * n)
    start_idx = max(0, min(start_idx, n - 1))
    count = min(INSET_KEY_COUNT, n - start_idx)

    keys = read_consecutive_u64(path, start_idx, count)
    if len(keys) == 0:
        return

    # Normalize keys to x (same scale as main plot)
    k = keys.astype(np.float64)
    if vmax == vmin:
        x_inset = np.zeros_like(k)
    else:
        x_inset = (k - vmin) / (vmax - vmin)
    y_inset = (start_idx + np.arange(len(keys))) / n

    # Zoom region bounds
    x1, x2 = float(np.min(x_inset)), float(np.max(x_inset))
    y1 = start_idx / n
    y2 = (start_idx + len(keys) - 1) / n
    # Avoid zero-width range
    if x2 == x1:
        x1, x2 = max(0, x1 - 0.01), min(1, x2 + 0.01)
    if y2 == y1:
        y1, y2 = max(0, y1 - 0.01), min(1, y2 + 0.01)

    # Anchor point (first key of extracted region)
    x0 = (float(keys[0]) - vmin) / (vmax - vmin) if vmax != vmin else 0.0
    y0 = start_idx / n

    # Inset axes
    iax = inset_axes(ax, width="38%", height="38%", loc=inset_loc, borderpad=1.0)
    iax.plot(x_inset, y_inset, linewidth=min(PLOT_LINEWIDTH, 2.5))
    iax.set_xlim(x1, x2)
    iax.set_ylim(y1, y2)
    iax.set_xticks([])
    iax.set_yticks([])

    # Orange dot (anchor)
    ax.plot([x0], [y0], marker="o", markersize=10, color="tab:orange", zorder=5)

    # Orange dashed rectangle (zoom region)
    rect = Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, ec="tab:orange", linestyle="--", linewidth=PLOT_LINEWIDTH, zorder=5,
    )
    ax.add_patch(rect)

    # Connect dot to rectangle centroid (zoom region center)
    # Use ConnectionPatch: dot (in ax) -> center (in iax) for visible line
    xc = x1 + (x2 - x1) * 0.5
    yc = y1 + (y2 - y1) * 0.5
    con = ConnectionPatch(
        xyA=(x0, y0), coordsA=ax.transData,
        xyB=(xc, yc), coordsB=iax.transData,
        axesA=ax, axesB=iax,
        color="tab:orange", linestyle="--", linewidth=PLOT_LINEWIDTH,
        connectionstyle="arc3,rad=0", zorder=4,
    )
    ax.add_artist(con)

def plot_grid(specs: List[DatasetSpec],
              seed: int,
              out_path: str,
              ncols: int = 3,
              figsize: Tuple[float, float] = (16.0, 2.6),
              inset_q: float = 0.8):
    rng = np.random.default_rng(seed)

    n = len(specs)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
    )
    axes = axes.ravel()

    for i, spec in enumerate(specs):
        ax = axes[i]
        vals = sample_u64(spec.path, sample_size=ALL_KEY_COUNT, rng=rng)
        x, y = approx_cdf_from_sample(vals)

        ax.set_title(spec.title, fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # Do not use set_box_aspect(1): it shrinks each Axes inside its GridSpec cell and
        # leaves wide empty gutters horizontally even when wspace=0. figsize is chosen so
        # each cell is ~square (see main()), so x/y scales match without box_aspect.

        if len(vals) == 0:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                fontsize=FONTSIZE_TEXT,
                color=TEXT_COLOR,
            )
        else:
            ax.plot(x, y, linewidth=PLOT_LINEWIDTH)
            # Inset with truly consecutive keys
            n_total = read_count_u64(spec.path)
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            add_inset(ax, path=spec.path, n=n_total, vmin=vmin, vmax=vmax,
                      q=inset_q, inset_loc=spec.inset_loc)

        # Clean look
        ax.tick_params(axis="both", which="both", labelsize=FONTSIZE_TICKS, length=4)

        if i % ncols != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Position", fontsize=FONTSIZE_YLABEL)
        if i < (nrows - 1) * ncols:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Key", fontsize=FONTSIZE_XLABEL)

    # Turn off unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Avoid fig.tight_layout: inset_axes (add_inset) are not compatible with tight_layout and
    # emit UserWarning. Reserve the same figure margins with subplots_adjust instead.
    fig.subplots_adjust(left=0.030, bottom=0.07, right=0.995, top=0.96)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--cols", type=int, default=3, help="Number of columns in grid.")
    ap.add_argument("--inset_q", type=float, default=0.8,
                    help="Inset anchor CDF position q in [0,1] (default: 0.8).")
    ap.add_argument(
        "--duplicate-only",
        action="store_true",
        help="Only duplicate-key datasets (bench2/bench3/wiki/zipf); fig/key_rank/key_rank_dup.pdf",
    )
    args = ap.parse_args()

    if args.duplicate_only:
        spec_fn = key_rank_specs_duplicate_only
        output_path = "fig/key_rank/key_rank_dup.pdf"
    else:
        spec_fn = key_rank_specs_swing_800m
        output_path = "fig/key_rank/key_rank.pdf"

    all_specs = [
        DatasetSpec(title, path, inset_loc)
        for title, path, inset_loc in spec_fn()
    ]

    # All datasets -> auto rows (figsize height = width * nrows/ncols for square subplots)
    plot_grid(
        specs=all_specs,
        seed=args.seed,
        out_path=output_path,
        ncols=args.cols,
        figsize=(12.0, 8.0 * (int(np.ceil(len(all_specs) / args.cols)) / args.cols)),
        inset_q=args.inset_q,
    )

    print(f"Saved plot: {output_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
