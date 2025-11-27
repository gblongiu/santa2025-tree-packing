"""
Global configuration for the Santa 2025 – Christmas Tree Packing project.

This module centralizes:

- Project-root and data paths
- Random seeds for reproducibility
- Global numeric constants used across packers / evaluation
- Tunable parameters for local search

All of these are kept in one place so that experiments are easy to
reproduce and configuration changes don’t require hunting through
multiple files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives in: <repo>/src/santa2025/config.py
# Project root is therefore two levels up from here.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_SUBMISSIONS_DIR: Path = DATA_DIR / "submissions"
DATA_INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"  # optional, for experiments


# ---------------------------------------------------------------------------
# Randomness / reproducibility
# ---------------------------------------------------------------------------

# Default global seed for experiments. You can override per run.
DEFAULT_SEED: int = 1234


def set_global_seeds(seed: Optional[int] = None) -> int:
    """
    Set Python's and NumPy's global random seeds and return the seed used.

    Use this at the start of scripts / notebooks to make runs reproducible.

        from santa2025.config import set_global_seeds
        set_global_seeds(2025)
    """
    if seed is None:
        seed = DEFAULT_SEED

    random.seed(seed)
    np.random.seed(seed)
    return seed


# ---------------------------------------------------------------------------
# Problem-scale constants
# ---------------------------------------------------------------------------

# Number of distinct puzzles (n = 1..N_MAX_TREES)
# If Kaggle ever changes this, updating here should be enough.
N_MAX_TREES: int = 200

# Competition requires coordinates to lie in [-100, 100].
COORD_MIN: float = -100.0
COORD_MAX: float = 100.0
COORD_RANGE: float = COORD_MAX - COORD_MIN


# ---------------------------------------------------------------------------
# Local-search configuration
# ---------------------------------------------------------------------------

# These control the *step size* of the local search (see packers.local_search).
# They are tuned to give small but meaningful moves; you can override per call.

# Typical coordinate scale is [-100, 100], so a move of 0.02 is very small
# relative to the full range but still enough to refine a hex lattice.
LOCAL_SEARCH_MOVE_SCALE: float = 0.02

# Angle step in degrees. Small rotations tend to be safer; larger values explore
# more but also cause more rejections. 3–5 degrees is a good compromise.
LOCAL_SEARCH_ANGLE_SCALE: float = 3.0

# Iteration budgets by problem size.
# These are *recommendations* – use them from packers/submission code rather
# than hard-coding magic numbers.

LOCAL_SEARCH_ITERS_SMALL: int = 2_000   # e.g. n <= 40
LOCAL_SEARCH_ITERS_MEDIUM: int = 3_500  # e.g. 40 < n <= 100
LOCAL_SEARCH_ITERS_LARGE: int = 5_000   # e.g. n > 100

# Backwards-compatible names (if any code already imports these):
LOCAL_SEARCH_DEFAULT_ITERS_SMALL: int = LOCAL_SEARCH_ITERS_SMALL
LOCAL_SEARCH_DEFAULT_ITERS_MEDIUM: int = LOCAL_SEARCH_ITERS_MEDIUM
LOCAL_SEARCH_DEFAULT_ITERS_LARGE: int = LOCAL_SEARCH_ITERS_LARGE


def local_search_iters_for_n(n: int) -> int:
    """
    Heuristic helper to choose a local-search iteration budget based
    on the number of trees in the puzzle.

    This keeps runtime roughly linear in n while giving larger instances
    more optimization steps.

    Parameters
    ----------
    n:
        Number of trees for the current puzzle.

    Returns
    -------
    int
        Number of local-search iterations to use.
    """
    if n <= 40:
        return LOCAL_SEARCH_ITERS_SMALL
    if n <= 100:
        return LOCAL_SEARCH_ITERS_MEDIUM
    return LOCAL_SEARCH_ITERS_LARGE


__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "DATA_RAW_DIR",
    "DATA_SUBMISSIONS_DIR",
    "DATA_INTERMEDIATE_DIR",
    # Seeds / randomness
    "DEFAULT_SEED",
    "set_global_seeds",
    # Problem constants
    "N_MAX_TREES",
    "COORD_MIN",
    "COORD_MAX",
    "COORD_RANGE",
    # Local search parameters
    "LOCAL_SEARCH_MOVE_SCALE",
    "LOCAL_SEARCH_ANGLE_SCALE",
    "LOCAL_SEARCH_ITERS_SMALL",
    "LOCAL_SEARCH_ITERS_MEDIUM",
    "LOCAL_SEARCH_ITERS_LARGE",
    "LOCAL_SEARCH_DEFAULT_ITERS_SMALL",
    "LOCAL_SEARCH_DEFAULT_ITERS_MEDIUM",
    "LOCAL_SEARCH_DEFAULT_ITERS_LARGE",
    "local_search_iters_for_n",
]
