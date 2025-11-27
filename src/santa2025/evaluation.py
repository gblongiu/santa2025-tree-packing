"""
Evaluation utilities for the Santa 2025 Christmas Tree Packing project.

This module provides a local implementation of the Kaggle competition metric.

For a submission with rows of the form

    id,x,y,deg
    001_0,s0.0,s0.0,s20.411299
    002_0,s0.0,s0.0,s20.411299
    002_1,s-0.541068,s0.259317,s51.66348
    ...

we do the following:

1. Parse the "id" column to extract n (tree count) and idx (tree index).
2. Decode the s-prefixed numeric values into floats.
3. For each n:
   - Build all tree polygons from (x, y, deg).
   - Compute the axis aligned bounding box of the union.
   - Take the side length s_n as max(width, height).
   - Compute the contribution s_n^2 / n.
4. Sum the contributions over n = 1..N_MAX_TREES to get the total score.

The functions here are intended to match the Kaggle metric to within
floating point noise, so that local development and leaderboard scores
are consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union

import math

import numpy as np
import pandas as pd
from shapely.ops import unary_union

from .config import (
    N_MAX_TREES,
    COORD_MIN,
    COORD_MAX,
)
from .geometry import make_tree_polygon


# Type alias for clarity
ScoreTable = pd.DataFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_val(value) -> float:
    """
    Decode competition style values like "s0.123456" into floats.

    If the value is already numeric, it is cast to float.
    If it is a string that starts with "s", the prefix is stripped.
    Otherwise it is passed to float().
    """
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str) and value.startswith("s"):
        return float(value[1:])
    return float(value)


def parse_submission_df(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Parse a raw submission DataFrame into a canonical numeric form.

    Adds columns:
    - n: tree count for the puzzle (from id prefix)
    - idx: index of the tree within the puzzle (from id suffix)

    Decodes x, y, deg to floats and sorts by (n, idx).

    If `strict` is True, performs basic validation checks:
    - n is in [1, N_MAX_TREES]
    - each n has exactly n rows
    - coordinates lie within [COORD_MIN, COORD_MAX]
    """
    required_cols = {"id", "x", "y", "deg"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Submission is missing required columns: {sorted(missing)}")

    parsed = df.copy()

    # Extract n and idx from "id" of the form "037_12"
    try:
        parts = parsed["id"].astype(str).str.split("_", n=1, expand=True)
        parsed["n"] = parts[0].astype(int)
        parsed["idx"] = parts[1].astype(int)
    except Exception as exc:
        raise ValueError(
            "Could not parse 'id' column into n and idx. "
            "Expected format like '037_12'."
        ) from exc

    # Decode x, y, deg
    for col in ["x", "y", "deg"]:
        parsed[col] = parsed[col].apply(decode_val)

    # Sort by n then idx for determinism
    parsed = parsed.sort_values(["n", "idx"]).reset_index(drop=True)

    if not strict:
        return parsed

    # Basic validation
    if parsed["n"].min() < 1 or parsed["n"].max() > N_MAX_TREES:
        raise ValueError(
            f"Found n outside the allowed range [1, {N_MAX_TREES}]. "
            f"Min n={parsed['n'].min()}, max n={parsed['n'].max()}."
        )

    # Each n should have exactly n rows
    counts = parsed.groupby("n")["idx"].count()
    bad_ns = [int(n) for n, c in counts.items() if c != n]
    if bad_ns:
        raise ValueError(
            "Submission has incorrect number of rows for some n. "
            f"Examples: {bad_ns[:5]}"
        )

    # Coordinate bounds
    min_x = parsed["x"].min()
    max_x = parsed["x"].max()
    min_y = parsed["y"].min()
    max_y = parsed["y"].max()
    if not (COORD_MIN <= min_x <= COORD_MAX and COORD_MIN <= max_x <= COORD_MAX):
        raise ValueError(
            f"x coordinates out of bounds [{COORD_MIN}, {COORD_MAX}]: "
            f"min_x={min_x}, max_x={max_x}"
        )
    if not (COORD_MIN <= min_y <= COORD_MAX and COORD_MIN <= max_y <= COORD_MAX):
        raise ValueError(
            f"y coordinates out of bounds [{COORD_MIN}, {COORD_MAX}]: "
            f"min_y={min_y}, max_y={max_y}"
        )

    return parsed


def bounding_square_side_for_group(group: pd.DataFrame) -> float:
    """
    Compute the side length of the smallest axis aligned square
    that contains all trees in a group for a single n.

    The group must contain columns x, y, deg.

    Note: this function does not check for overlaps. The competition
    enforces non overlap separately when evaluating submissions.
    """
    polys = [
        make_tree_polygon(row["x"], row["y"], row["deg"])
        for _, row in group.iterrows()
    ]

    union = unary_union(polys)
    minx, miny, maxx, maxy = union.bounds
    width = maxx - minx
    height = maxy - miny
    return max(width, height)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def metric_from_df(raw_df: pd.DataFrame, strict: bool = True) -> ScoreTable:
    """
    Compute the metric table from a raw submission DataFrame.

    Parameters
    ----------
    raw_df:
        DataFrame with at least columns id, x, y, deg.
    strict:
        If True, perform validation checks on n counts and coordinate bounds.

    Returns
    -------
    ScoreTable (pd.DataFrame)
        Columns:
        - n: tree count
        - side: bounding square side length s_n
        - score_n: contribution s_n^2 / n
        - count: number of rows for this n (should equal n)
    """
    df = parse_submission_df(raw_df, strict=strict)

    records = []
    for n, group in df.groupby("n"):
        side = bounding_square_side_for_group(group)
        contribution = side * side / float(n)
        records.append(
            {
                "n": int(n),
                "side": float(side),
                "score_n": float(contribution),
                "count": int(len(group)),
            }
        )

    table = pd.DataFrame(records).sort_values("n").reset_index(drop=True)
    return table


def evaluate_submission_csv(
    path: Union[str, Path],
    strict: bool = True,
) -> Tuple[ScoreTable, float]:
    """
    Load a submission CSV and compute its score table and total score.

    Parameters
    ----------
    path:
        Path to a CSV file with columns id, x, y, deg.
    strict:
        If True, enable validation checks in `metric_from_df`.

    Returns
    -------
    score_table:
        Per n metric details.
    total_score:
        Sum of score_n over all n in the table.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    score_table = metric_from_df(raw_df, strict=strict)
    total_score = float(score_table["score_n"].sum())
    return score_table, total_score


__all__ = [
    "ScoreTable",
    "decode_val",
    "parse_submission_df",
    "bounding_square_side_for_group",
    "metric_from_df",
    "evaluate_submission_csv",
]
