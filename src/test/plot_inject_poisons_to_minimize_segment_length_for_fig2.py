#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


FIGSIZE: Tuple[float, float] = (1.95, 1.95)
SEGMENT_LINEWIDTH = 3.0


@dataclass
class _Point:
    x: float
    y: float


def _cross(O: _Point, A: _Point, B: _Point) -> float:
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)


class _CanonicalSegment:
    def __init__(self, rect4: List[_Point], first_x: float):
        self.r = rect4
        self.first = first_x

    def one_point(self) -> bool:
        r = self.r
        return (
            r[0].x == r[2].x
            and r[0].y == r[2].y
            and r[1].x == r[3].x
            and r[1].y == r[3].y
        )

    def get_slope_range(self) -> Tuple[float, float]:
        if self.one_point():
            return (0.0, 0.0)
        r = self.r
        min_slope = (r[2].y - r[0].y) / (r[2].x - r[0].x)
        max_slope = (r[3].y - r[1].y) / (r[3].x - r[1].x)
        return (min_slope, max_slope)

    def get_intersection(self) -> Tuple[float, float]:
        r = self.r
        p0, p1, p2, p3 = r[0], r[1], r[2], r[3]
        if self.one_point():
            return (p0.x, p0.y)

        s1x, s1y = (p2.x - p0.x), (p2.y - p0.y)
        s2x, s2y = (p3.x - p1.x), (p3.y - p1.y)

        a = s1x * s2y - s1y * s2x
        if a == 0:
            return (p0.x, p0.y)

        p0p1x, p0p1y = (p1.x - p0.x), (p1.y - p0.y)
        b = (p0p1x * s2y - p0p1y * s2x) / a
        ix = p0.x + b * s1x
        iy = p0.y + b * s1y
        return (ix, iy)

    def get_floating_point_segment(self, origin: float) -> Tuple[float, float]:
        if self.one_point():
            ymid = 0.5 * (self.r[0].y + self.r[1].y)
            w = 0.0
            b = ymid
            return (w, b)

        ix, iy = self.get_intersection()
        mn, mx = self.get_slope_range()
        w = 0.5 * (mn + mx)
        intercept = iy - (ix - origin) * w
        b = intercept - w * origin
        return (w, b)


class OptimalPiecewiseLinearModelFloat:
    """Check if a line exists that fits all points with error <= epsilon by add_point."""

    def __init__(self, epsilon: float):
        if epsilon < 0:
            raise ValueError("epsilon cannot be negative")
        self.eps = float(epsilon)
        self.lower: List[_Point] = []
        self.upper: List[_Point] = []
        self.first_x: float = 0.0
        self.last_x: float = 0.0
        self.lower_start: int = 0
        self.upper_start: int = 0
        self.points_in_hull: int = 0
        self.rect: List[_Point] = [_Point(0.0, 0.0) for _ in range(4)]

    def reset(self) -> None:
        self.lower.clear()
        self.upper.clear()
        self.points_in_hull = 0
        self.lower_start = 0
        self.upper_start = 0

    def add_point(self, x: float, y: float) -> bool:
        if self.points_in_hull > 0 and x <= self.last_x:
            raise ValueError("Points must be strictly increasing by x.")
        self.last_x = x

        p1 = _Point(x, y + self.eps)
        p2 = _Point(x, y - self.eps)

        if self.points_in_hull == 0:
            self.first_x = x
            self.rect[0] = p1
            self.rect[1] = p2
            self.upper[:] = [p1]
            self.lower[:] = [p2]
            self.upper_start = self.lower_start = 0
            self.points_in_hull = 1
            return True

        if self.points_in_hull == 1:
            self.rect[2] = p2
            self.rect[3] = p1
            self.upper.append(p1)
            self.lower.append(p2)
            self.points_in_hull = 2
            return True

        r = self.rect
        slope1 = ((r[2].x - r[0].x), (r[2].y - r[0].y))
        slope2 = ((r[3].x - r[1].x), (r[3].y - r[1].y))

        def slope_less(a_dx, a_dy, b_dx, b_dy) -> bool:
            return a_dy * b_dx < a_dx * b_dy

        def slope_greater(a_dx, a_dy, b_dx, b_dy) -> bool:
            return a_dy * b_dx > a_dx * b_dy

        outside_line1 = slope_less(p1.x - r[2].x, p1.y - r[2].y, slope1[0], slope1[1])
        outside_line2 = slope_greater(p2.x - r[3].x, p2.y - r[3].y, slope2[0], slope2[1])

        if outside_line1 or outside_line2:
            self.points_in_hull = 0
            return False

        if slope_less(p1.x - r[1].x, p1.y - r[1].y, slope2[0], slope2[1]):
            min_dx = self.lower[self.lower_start].x - p1.x
            min_dy = self.lower[self.lower_start].y - p1.y
            min_i = self.lower_start
            for i in range(self.lower_start + 1, len(self.lower)):
                val_dx = self.lower[i].x - p1.x
                val_dy = self.lower[i].y - p1.y
                if slope_greater(val_dx, val_dy, min_dx, min_dy):
                    break
                min_dx, min_dy, min_i = val_dx, val_dy, i

            r[1] = self.lower[min_i]
            r[3] = p1
            self.lower_start = min_i

            end = len(self.upper)
            while end >= self.upper_start + 2 and _cross(self.upper[end - 2], self.upper[end - 1], p1) <= 0:
                end -= 1
            self.upper = self.upper[:end]
            self.upper.append(p1)

        if slope_greater(p2.x - r[0].x, p2.y - r[0].y, slope1[0], slope1[1]):
            max_dx = self.upper[self.upper_start].x - p2.x
            max_dy = self.upper[self.upper_start].y - p2.y
            max_i = self.upper_start
            for i in range(self.upper_start + 1, len(self.upper)):
                val_dx = self.upper[i].x - p2.x
                val_dy = self.upper[i].y - p2.y
                if slope_less(val_dx, val_dy, max_dx, max_dy):
                    break
                max_dx, max_dy, max_i = val_dx, val_dy, i

            r[0] = self.upper[max_i]
            r[2] = p2
            self.upper_start = max_i

            end = len(self.lower)
            while end >= self.lower_start + 2 and _cross(self.lower[end - 2], self.lower[end - 1], p2) >= 0:
                end -= 1
            self.lower = self.lower[:end]
            self.lower.append(p2)

        self.points_in_hull += 1
        return True

    def get_segment(self) -> _CanonicalSegment:
        if self.points_in_hull == 1:
            r = [self.rect[0], self.rect[1], self.rect[0], self.rect[1]]
            return _CanonicalSegment(r, self.first_x)
        return _CanonicalSegment(self.rect.copy(), self.first_x)


# ----------------------------
# Input parsing / segment calculation
# ----------------------------

def parse_keys_and_poisons(input_str: str) -> Tuple[List[int], List[int], List[int]]:
    """Returns: (all_keys_sorted, legitimate_keys_sorted, poisons_sorted)."""
    parts = input_str.split("|")
    all_keys: List[int] = []
    poisons: List[int] = []

    for part in parts:
        matches = re.findall(r"\[(\d+)\]|(\d+)", part)
        for match in matches:
            if match[0]:
                v = int(match[0])
                all_keys.append(v)
                poisons.append(v)
            else:
                all_keys.append(int(match[1]))

    all_keys.sort()
    poisons.sort()
    poison_set = set(poisons)
    legitimate_keys = [k for k in all_keys if k not in poison_set]
    return all_keys, legitimate_keys, poisons


def get_segment_points(keys: List[int], start_idx: int, end_idx: int, y_offset: int) -> Tuple[List[int], List[float]]:
    """Return the point sequence in the segment, following the duplicate key handling of PGM-index."""
    segment_keys: List[int] = []
    segment_ranks: List[float] = []
    n = len(keys)

    segment_keys.append(keys[start_idx])
    segment_ranks.append((start_idx + 1) + y_offset)

    for i in range(start_idx + 1, min(end_idx, n - 1)):
        if keys[i] == keys[i - 1]:
            if i + 1 < n and keys[i] + 1 < keys[i + 1]:
                segment_keys.append(keys[i] + 1)
                segment_ranks.append((i + 1) + y_offset)
        else:
            segment_keys.append(keys[i])
            segment_ranks.append((i + 1) + y_offset)

    if end_idx < n and end_idx > start_idx:
        if keys[end_idx] != keys[end_idx - 1]:
            segment_keys.append(keys[end_idx])
            segment_ranks.append((end_idx + 1) + y_offset)

    return segment_keys, segment_ranks


def can_fit_with_epsilon(keys: List[int], ranks: List[float], epsilon: float) -> bool:
    if len(keys) <= 2:
        return True
    model = OptimalPiecewiseLinearModelFloat(float(epsilon))
    for x, y in zip(keys, ranks):
        if not model.add_point(float(x), float(y)):
            return False
    return True


def compute_optimal_regression(keys: List[int], ranks: List[float], epsilon: float) -> Tuple[float, float]:
    if len(keys) == 0:
        return 0.0, 0.0
    if len(keys) == 1:
        return 0.0, ranks[0]

    model = OptimalPiecewiseLinearModelFloat(float(epsilon))
    for x, y in zip(keys, ranks):
        if not model.add_point(float(x), float(y)):
            raise RuntimeError("Failed to add point to model.")
    segment = model.get_segment()
    slope, intercept = segment.get_floating_point_segment(origin=float(keys[0]))
    return slope, intercept


def extend_segment_end(keys: List[int], start_idx: int, epsilon: float, y_offset: int) -> int:
    n = len(keys)
    if start_idx >= n:
        return n - 1
    if start_idx == n - 1:
        return n - 1

    for j in range(start_idx + 1, n):
        test_keys, test_ranks = get_segment_points(keys, start_idx, j, y_offset)
        if can_fit_with_epsilon(test_keys, test_ranks, epsilon):
            continue
        return j - 1

    return n - 1


def compute_segments(keys: List[int], epsilon: float, y_offset: int = 1_000_000) -> List[Tuple[int, int, float, float]]:
    """Returns list of (start_idx, end_idx, slope, intercept)."""
    if len(keys) == 0:
        return []
    segments: List[Tuple[int, int, float, float]] = []
    n = len(keys)
    current_i = 0

    while current_i < n:
        segment_end = extend_segment_end(keys, current_i, epsilon, y_offset)
        segment_keys, segment_ranks = get_segment_points(keys, current_i, segment_end, y_offset)
        if len(segment_keys) > 1:
            slope, intercept = compute_optimal_regression(segment_keys, segment_ranks, epsilon)
            segments.append((current_i, segment_end, slope, intercept))
        else:
            segments.append((current_i, segment_end, 0.0, segment_ranks[0]))
        current_i = segment_end + 1

    return segments


def compute_max_error_on_sorted_keys(
    keys: List[int],
    segments: List[Tuple[int, int, float, float]],
    *,
    y_offset: float,
) -> float:
    if len(keys) == 0:
        return 0.0
    ranks = list(range(1, len(keys) + 1))
    max_error = 0.0
    for start_idx, end_idx, slope, intercept in segments:
        for i in range(start_idx, min(end_idx + 1, len(keys))):
            x_point = keys[i]
            y_actual = ranks[i]
            y_predicted = slope * x_point + intercept - y_offset
            max_error = max(max_error, abs(y_predicted - y_actual))
    return float(max_error)


def count_covered_legitimate_keys(
    all_keys: List[int],
    legitimate_keys: List[int],
    segments: List[Tuple[int, int, float, float]],
    *,
    only_segment_index: Optional[int] = None,
) -> int:
    """
    Count of **unique** legitimate keys covered by the segment.
    (Count only once even if the same key appears multiple times.)
    Assume that poisons are not included in legitimate_keys.

    If only_segment_index is specified, only the specified segment (0-indexed) is considered.
    """
    if len(all_keys) == 0:
        return 0
    legit_set = set(legitimate_keys)
    covered = [False] * len(all_keys)
    if only_segment_index is None:
        for start_idx, end_idx, _slope, _intercept in segments:
            lo = max(0, start_idx)
            hi = min(len(all_keys) - 1, end_idx)
            for i in range(lo, hi + 1):
                covered[i] = True
    else:
        if only_segment_index < 0 or only_segment_index >= len(segments):
            return 0
        start_idx, end_idx, _slope, _intercept = segments[only_segment_index]
        lo = max(0, start_idx)
        hi = min(len(all_keys) - 1, end_idx)
        for i in range(lo, hi + 1):
            covered[i] = True

    covered_legit: Set[int] = set()
    for i, k in enumerate(all_keys):
        if k in legit_set and covered[i]:
            covered_legit.add(k)
    return int(len(covered_legit))


# ----------------------------
# Plotting
# ----------------------------

def plot_segments(
    ax,
    keys: List[int],
    segments: List[Tuple[int, int, float, float]],
    *,
    color: str,
    label: str,
    marker: str = "o",
    markersize: int = 30,
    y_offset: float = 0.0,
    x_range: Optional[Tuple[int, int]] = None,
    show_residuals: bool = True,
    linestyle: str = "-",
    line_label: Optional[str] = None,
) -> None:
    ranks = list(range(1, len(keys) + 1))
    ax.scatter(keys, ranks, marker=marker, s=markersize, edgecolors=color, facecolors=color, label=label, zorder=4)

    if x_range is None:
        x_min, x_max = min(keys), max(keys)
    else:
        x_min, x_max = x_range
    _ = (x_min, x_max)

    first_segment = True
    for start_idx, end_idx, slope, intercept in segments:
        start_key = keys[start_idx]
        end_key = keys[end_idx]
        seg_span = end_key - start_key
        margin = max(2.5, seg_span * 0.2)
        x_segment = np.linspace(start_key - margin, end_key + margin, 100)
        y_segment = slope * x_segment + intercept - y_offset

        seg_label = line_label if (line_label is not None and first_segment) else ""
        ax.plot(x_segment, y_segment, color=color, linestyle=linestyle, linewidth=SEGMENT_LINEWIDTH, label=seg_label, zorder=3)
        first_segment = False

        if show_residuals:
            for i in range(start_idx, min(end_idx + 1, len(keys))):
                x_point = keys[i]
                y_point = ranks[i]
                y_predicted = slope * x_point + intercept - y_offset
                ax.plot([x_point, x_point], [y_point, y_predicted], linestyle="-", color="tab:blue", linewidth=1.5, zorder=2)


def apply_plot_style(
    ax,
    *,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    num_segments: Optional[int] = None,
    show_legend: bool = True,
    fine_grid_1: bool = False,
) -> None:
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Keys", fontsize=16)
    ax.set_ylabel("Rank", fontsize=16)

    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6, zorder=1)
    ax.minorticks_on()
    if fine_grid_1:
        # Show a fine grid with 1-step intervals (labels remain the same as before).
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.45, zorder=1)
    else:
        ax.grid(which="minor", linestyle="--", linewidth=0.6, alpha=0.6, axis="x", zorder=1)
        ax.yaxis.set_minor_locator(plt.NullLocator())

    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    if not fine_grid_1:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(2))

    # Tick (tick labels) are always hidden. The grid remains with the locator settings.
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
        length=0,
        width=0,
    )

    if show_legend:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="black",
            fontsize=12,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.00),
            borderpad=0.25,
            labelspacing=0.25,
            handletextpad=0.5,
            handlelength=1.4,
            borderaxespad=0.2,
            handleheight=0.7,
        )

    if num_segments is not None:
        ax.text(
            x_range[1] - 0.5,
            y_range[0] + 0.3,
            f"#Segments: {num_segments}",
            fontsize=16,
            fontweight="bold",
            ha="right",
            va="bottom",
            zorder=5,
        )

def add_corner_text(ax, text: str) -> None:
    ax.text(
        0.99,
        0.01,
        text,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="heavy",
        color="black",
        alpha=1.0,
        ha="right",
        va="bottom",
        zorder=6,
    )


def _plot_segments_with_residuals(
    ax,
    *,
    keys: List[int],
    ranks: List[int],
    segments: List[Tuple[int, int, float, float]],
    y_offset: int,
    hide_segment_line_indices: Optional[Set[int]] = None,
) -> None:
    hide_line = hide_segment_line_indices or set()
    for seg_idx, (start_idx, end_idx, slope, intercept) in enumerate(segments):
        start_key = keys[start_idx]
        end_key = keys[end_idx]
        seg_span = end_key - start_key
        margin = max(2.5, seg_span * 0.2)
        x_segment = np.linspace(start_key - margin, end_key + margin, 100)
        y_segment = slope * x_segment + intercept - y_offset
        if seg_idx not in hide_line:
            ax.plot(x_segment, y_segment, color="tab:red", linestyle="-", linewidth=SEGMENT_LINEWIDTH, label="", zorder=3)
        for i in range(start_idx, min(end_idx + 1, len(keys))):
            x_point = keys[i]
            y_point = ranks[i]
            y_predicted = slope * x_point + intercept - y_offset
            ax.plot([x_point, x_point], [y_point, y_predicted], linestyle="-", color="tab:blue", linewidth=1.5, zorder=2)


def plot_keys_rank_with_segments(
    *,
    keys: List[int],
    segments: List[Tuple[int, int, float, float]],
    legitimate_keys: List[int],
    poisons: List[int],
    out_pdf: str,
    corner_metric: Optional[str] = None,
    highlight_legitimate_keys: Optional[List[int]] = None,
    dim_legitimate_keys: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fine_grid_1: bool = False,
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None,
    color_split_legit: bool = True,
    hide_segment_line_indices: Optional[Set[int]] = None,
    covered_legit_only_segment_index: Optional[int] = None,
) -> None:
    if figsize is None:
        figsize = FIGSIZE

    y_offset = 1_000_000

    if x_range is None:
        x_min = min(keys) if len(keys) else 0
        x_max = max(keys) if len(keys) else 1
        x_range = (x_min - 3, x_max + 5)
    if y_range is None:
        y_range = (0, len(keys) + 1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ranks = list(range(1, len(keys) + 1))
    legitimate_set = set(legitimate_keys)
    poison_set = set(poisons)

    highlight_set = set(highlight_legitimate_keys or [])
    dim_set = set(dim_legitimate_keys or [])
    other_legit_set = legitimate_set - highlight_set - dim_set

    def scatter_indices(indices: List[int], *, edge: str, face: str) -> None:
        if not indices:
            return
        ax.scatter(
            [keys[i] for i in indices],
            [ranks[i] for i in indices],
            marker="o",
            s=30,
            edgecolors=edge,
            facecolors=face,
            zorder=4,
        )

    if color_split_legit:
        highlight_indices = [i for i, k in enumerate(keys) if k in highlight_set]
        dim_indices = [i for i, k in enumerate(keys) if k in dim_set]
        other_legit_indices = [i for i, k in enumerate(keys) if k in other_legit_set]

        scatter_indices(highlight_indices, edge="k", face="k")
        scatter_indices(dim_indices, edge="0.55", face="0.55")
        scatter_indices(other_legit_indices, edge="k", face="k")
    else:
        legit_indices = [i for i, k in enumerate(keys) if k in legitimate_set]
        scatter_indices(legit_indices, edge="k", face="k")

    poison_indices = [i for i, k in enumerate(keys) if k in poison_set]
    ax.scatter(
        [keys[i] for i in poison_indices],
        [ranks[i] for i in poison_indices],
        marker="o",
        s=30,
        edgecolors="r",
        facecolors="r",
        zorder=4,
    )

    _plot_segments_with_residuals(
        ax,
        keys=keys,
        ranks=ranks,
        segments=segments,
        y_offset=y_offset,
        hide_segment_line_indices=hide_segment_line_indices,
    )

    # Do not use num_segments display in apply_plot_style, display the metrics in the lower right corner.
    apply_plot_style(ax, x_range=x_range, y_range=y_range, num_segments=None, show_legend=False, fine_grid_1=fine_grid_1)
    if corner_metric == "max_error":
        max_err = compute_max_error_on_sorted_keys(keys, segments, y_offset=float(y_offset))
        add_corner_text(ax, f"Err: {max_err:.1f}")
    elif corner_metric == "covered_legit":
        cov = count_covered_legitimate_keys(
            keys,
            legitimate_keys,
            segments,
            only_segment_index=covered_legit_only_segment_index,
        )
        if covered_legit_only_segment_index is not None:
            print(
                f"[Fig2] Cov (legit keys in segment {covered_legit_only_segment_index} only): {cov} "
                f"(pdf={out_pdf})"
            )
        add_corner_text(ax, f"Cov: {cov}")
    elif corner_metric == "num_segments":
        add_corner_text(ax, f"#Seg: {len(segments)}")

    # tight_layout may cause warnings due to incompatibility with axis labels, so adjust the margins manually.
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.96)

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def plot_before_after_pair(
    *,
    poisoned_input_str: str,
    epsilon: float,
    out_pdf_after: str,
    out_pdf_before: str,
    corner_metric: Optional[str],
    highlight_legitimate_keys: Optional[List[int]] = None,
    dim_legitimate_keys: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fine_grid_1: bool = False,
    hide_segment_line_indices_after: Optional[Set[int]] = None,
    covered_legit_only_segment_index_after: Optional[int] = None,
) -> None:
    all_keys, legitimate_keys, poisons = parse_keys_and_poisons(poisoned_input_str)

    poisoned_segments = compute_segments(all_keys, epsilon, 1_000_000)
    legitimate_segments = compute_segments(legitimate_keys, epsilon, 1_000_000)

    x_min = min(all_keys) if len(all_keys) else 0
    x_max = max(all_keys) if len(all_keys) else 1
    x_range = (x_min - 3, x_max + 5)
    y_range_after = (0, len(all_keys) + 1)

    plot_keys_rank_with_segments(
        keys=all_keys,
        segments=poisoned_segments,
        legitimate_keys=legitimate_keys,
        poisons=poisons,
        out_pdf=out_pdf_after,
        corner_metric=corner_metric,
        highlight_legitimate_keys=highlight_legitimate_keys,
        dim_legitimate_keys=dim_legitimate_keys,
        figsize=figsize,
        fine_grid_1=fine_grid_1,
        x_range=x_range,
        y_range=y_range_after,
        color_split_legit=True,
        hide_segment_line_indices=hide_segment_line_indices_after,
        covered_legit_only_segment_index=covered_legit_only_segment_index_after,
    )

    plot_keys_rank_with_segments(
        keys=legitimate_keys,
        segments=legitimate_segments,
        legitimate_keys=legitimate_keys,
        poisons=[],
        out_pdf=out_pdf_before,
        corner_metric=corner_metric,
        highlight_legitimate_keys=highlight_legitimate_keys,
        dim_legitimate_keys=dim_legitimate_keys,
        figsize=figsize,
        fine_grid_1=fine_grid_1,
        x_range=x_range,
        y_range=y_range_after,
        color_split_legit=False,
    )


# ----------------------------
# Default input (edit as needed)
# ----------------------------

DEFAULT_INPUT_MAX_ERROR_MAXIMIZE = "2 3 [4] 6 16 18"
DEFAULT_INPUT_MINIMIZE_SEGMENT_LENGTH = "2 3 [4] 6 16 18 34 36"
DEFAULT_INPUT_SEGMENTS_NUM = "2 3 [4] 6 16 18 34 [35] 36 38 43 49 64 68 74 81 87"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="results/test_inject_poisons_to_minimize_segment_length_for_fig2")
    parser.add_argument("--input_max_error_maximize", type=str, default=DEFAULT_INPUT_MAX_ERROR_MAXIMIZE)
    parser.add_argument("--input_minimize_segment_length", type=str, default=DEFAULT_INPUT_MINIMIZE_SEGMENT_LENGTH)
    parser.add_argument("--input_segments_num", type=str, default=DEFAULT_INPUT_SEGMENTS_NUM)
    parser.add_argument(
        "--fig_width",
        type=float,
        default=None,
        help="Figure width (inches). If not specified, use FIGSIZE[0].",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=None,
        help="Figure height (inches). If not specified, use FIGSIZE[1].",
    )
    args = parser.parse_args()

    fig_w = float(args.fig_width) if args.fig_width is not None else float(FIGSIZE[0])
    fig_h = float(args.fig_height) if args.fig_height is not None else float(FIGSIZE[1])
    figsize = (fig_w, fig_h)

    plot_before_after_pair(
        poisoned_input_str=args.input_max_error_maximize,
        epsilon=args.epsilon,
        out_pdf_after=os.path.join(args.out_dir, "plot_max_error_maximize.pdf"),
        out_pdf_before=os.path.join(args.out_dir, "plot_max_error_maximize_before.pdf"),
        corner_metric="max_error",
        fine_grid_1=True,
        figsize=figsize,
    )

    plot_before_after_pair(
        poisoned_input_str=args.input_minimize_segment_length,
        epsilon=args.epsilon,
        out_pdf_after=os.path.join(args.out_dir, "plot_minimize_segment_length.pdf"),
        out_pdf_before=os.path.join(args.out_dir, "plot_minimize_segment_length_before.pdf"),
        corner_metric="covered_legit",
        highlight_legitimate_keys=[2, 3, 6, 16, 18],
        dim_legitimate_keys=[34, 36],
        fine_grid_1=True,
        figsize=figsize,
        # Only for After: Hide the red dotted line of the second segment (0-indexed 1)
        hide_segment_line_indices_after={1},
        # Cov is the number of legitimate keys in the "first segment" (only for After)
        covered_legit_only_segment_index_after=0,
    )

    plot_before_after_pair(
        poisoned_input_str=args.input_segments_num,
        epsilon=args.epsilon,
        out_pdf_after=os.path.join(args.out_dir, "plot_segments_num.pdf"),
        out_pdf_before=os.path.join(args.out_dir, "plot_segments_num_before.pdf"),
        corner_metric="num_segments",
        figsize=figsize,
    )

    print(f"Saved plots under: {args.out_dir}")


if __name__ == "__main__":
    main()

