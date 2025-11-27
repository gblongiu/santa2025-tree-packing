"""
High-level submission builder for the Santa 2025 Christmas Tree Packing project.

This module glues together:

- A baseline packer (hexagonal lattice) from `santa2025.packers.hex_lattice`.
- An optional local search refinement from `santa2025.packers.local_search`.
- The evaluation configuration from `santa2025.config`.

It exposes functions to:

- Build an in-memory submission `DataFrame` with all trees for n = 1..N_MAX_TREES.
- Write a Kaggle-ready CSV (with `s`-prefixed numeric strings) to disk.
- Use a small CLI for convenience:

      python -m santa2025.submission
      # or
      python src/santa2025/submission.py

Once the `packers` implementations are in place, this is what you run to
generate your competition submissions.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import (
    N_MAX_TREES,
    DATA_SUBMISSIONS_DIR,
    set_global_seeds,
    local_search_iters_for_n,
)
from .geometry import TreePose

# We deliberately import the packers lazily inside functions so that this module
# can be imported before all packers are fully implemented, if needed.


# ---------------------------------------------------------------------------
# Core layout builder for a single n
# ---------------------------------------------------------------------------

def build_layout_for_n(
    n: int,
    use_local_search: bool = True,
    local_search_iters: Optional[int] = None,
) -> List[TreePose]:
    """
    Build a layout for a single puzzle with `n` trees.

    Pipeline:

    1. Use the hexagonal lattice baseline to produce a non-overlapping layout.
    2. Optionally run a global radial compaction toward the origin.
    3. Optionally run hill-climbing local search to shrink the bounding square.

    Parameters
    ----------
    n:
        Number of trees in this puzzle (1..N_MAX_TREES).
    use_local_search:
        If False, only the hex baseline is used.
    local_search_iters:
        If provided, override the default iteration count for this n, which
        otherwise comes from `local_search_iters_for_n` in `config.py`.

    Returns
    -------
    List[TreePose]
        A list of TreePose instances for each tree (index 0..n-1).
    """
    if n < 1 or n > N_MAX_TREES:
        raise ValueError(f"n must be in [1, {N_MAX_TREES}], got {n}")

    # Baseline hexagonal layout
    from .packers.hex_lattice import initial_hex_layout_for_n

    poses: List[TreePose] = initial_hex_layout_for_n(n)

    if not use_local_search:
        return poses

    # Import both global compaction and local search refinements.
    from .packers.local_search import refine_layout, radial_compact_layout

    # First: global radial compaction toward the origin
    _scale = radial_compact_layout(poses)

    # Then: local hill-climbing to nibble away at corners
    iters = (
        local_search_iters
        if local_search_iters is not None
        else local_search_iters_for_n(n)
    )
    _final_side = refine_layout(poses, iters=iters)
    return poses


# ---------------------------------------------------------------------------
# Build full submission DataFrame
# ---------------------------------------------------------------------------

def build_submission_df(
    use_local_search: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a full submission DataFrame covering n = 1..N_MAX_TREES.

    The resulting DataFrame has:

    - index: Kaggle `id` strings like '001_0', '001_1', ..., '200_199'
    - columns: 'x', 'y', 'deg'
    - values: float (not yet `s`-prefixed; that happens in `format_submission_df`)

    Parameters
    ----------
    use_local_search:
        Toggle local search refinement after the hex baseline.
    seed:
        Optional seed for reproducibility. Passed to `set_global_seeds`.

    Returns
    -------
    df : pd.DataFrame
        Raw numeric layout suitable for further formatting.
    """
    set_global_seeds(seed)

    ids: List[str] = []
    xs: List[float] = []
    ys: List[float] = []
    degs: List[float] = []

    for n in range(1, N_MAX_TREES + 1):
        poses_n = build_layout_for_n(n, use_local_search=use_local_search)
        if len(poses_n) != n:
            raise RuntimeError(
                f"Packer for n={n} returned {len(poses_n)} poses, expected {n}."
            )

        for idx, pose in enumerate(poses_n):
            ids.append(f"{n:03d}_{idx}")
            xs.append(float(pose.x))
            ys.append(float(pose.y))
            degs.append(float(pose.angle))

    df = pd.DataFrame({"id": ids, "x": xs, "y": ys, "deg": degs}).set_index("id")
    return df


def _format_value_for_submission(x: float) -> str:
    """
    Format a numeric value as a competition-style string:

        0.123456 -> 's0.123456'

    We fix to 6 decimal places, which is usually sufficient and aligns with the
    sample code style, while avoiding scientific notation.
    """
    return "s" + f"{x:.6f}"


def format_submission_df(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a numeric layout DataFrame into a Kaggle-ready submission DataFrame.

    - Ensures columns are x, y, deg.
    - Applies `s` prefix formatting to all values.
    - Leaves the index as 'id'.

    Parameters
    ----------
    df_numeric:
        DataFrame with index 'id' and numeric columns x, y, deg.

    Returns
    -------
    pd.DataFrame
        Formatted submission.
    """
    for col in ["x", "y", "deg"]:
        if col not in df_numeric.columns:
            raise ValueError(f"Expected column '{col}' in df_numeric.")

    sub = df_numeric.copy()
    for col in ["x", "y", "deg"]:
        sub[col] = sub[col].astype(float).apply(_format_value_for_submission)

    return sub


def build_and_format_submission(
    use_local_search: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: build numeric layouts then format for Kaggle.
    """
    df_numeric = build_submission_df(use_local_search=use_local_search, seed=seed)
    return format_submission_df(df_numeric)


# ---------------------------------------------------------------------------
# Write CSV helper
# ---------------------------------------------------------------------------

def write_submission_csv(
    output_path: Optional[Path] = None,
    use_local_search: bool = True,
    seed: Optional[int] = None,
) -> Path:
    """
    Build a full submission and write it to CSV.

    Parameters
    ----------
    output_path:
        If provided, the CSV will be written here. If None, a timestamped
        name will be created under DATA_SUBMISSIONS_DIR.
    use_local_search:
        Toggle local search refinement.
    seed:
        Optional global seed for reproducibility.

    Returns
    -------
    Path
        The path to the written CSV.
    """
    if output_path is None:
        DATA_SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = DATA_SUBMISSIONS_DIR / f"submission_hex_local_{timestamp}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df = build_and_format_submission(
        use_local_search=use_local_search,
        seed=seed,
    )
    submission_df.to_csv(output_path, index_label="id")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Kaggle submission CSV for the Santa 2025 packing challenge.",
    )
    parser.add_argument(
        "--no-local-search",
        action="store_true",
        help="Disable local search; use pure hex lattice baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output path for the CSV. If omitted, a timestamped name "
            "will be created under data/submissions/."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    use_local = not args.no_local_search
    seed = args.seed
    output = Path(args.output) if args.output is not None else None

    out_path = write_submission_csv(
        output_path=output,
        use_local_search=use_local,
        seed=seed,
    )
    print(f"Submission written to: {out_path}")


if __name__ == "__main__":
    main()
