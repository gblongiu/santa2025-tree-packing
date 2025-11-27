"""
Global configuration for the Santa 2025 – Christmas Tree Packing project.

This module centralizes:
- Project-root and data paths
- Random seeds for reproducibility
- A few global numeric constants used across packers / evaluation
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

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

# You can use these in code instead of hard-coding paths, e.g.:
#   from santa2025.config import DATA_RAW_DIR
#   sample_path = DATA_RAW_DIR / "sample_submission.csv"


# ---------------------------------------------------------------------------
# Randomness / reproducibility
# ---------------------------------------------------------------------------

# Default global seed for experiments. You can override per run.
DEFAULT_SEED: int = 1234


def set_global_seeds(seed: Optional[int] = None) -> int:
    """
    Set Python's and NumPy's global random seeds.
    Returns the seed actually used.
    """
    if seed is None:
        seed = DEFAULT_SEED

    random.seed(seed)
    np.random.seed(seed)
    return seed


# ---------------------------------------------------------------------------
# Global numeric constants
# ---------------------------------------------------------------------------

# Number of distinct puzzles (n = 1..N_MAX_TREES)
N_MAX_TREES: int = 200

# Competition requires coordinates to lie in [-100, 100].
COORD_MIN: float = -100.0
COORD_MAX: float = 100.0

# Default local-search parameters (you can override per call if needed).
# These are just reasonable starting points.
LOCAL_SEARCH_DEFAULT_ITERS_SMALL: int = 2_000
LOCAL_SEARCH_DEFAULT_ITERS_MEDIUM: int = 3_000
LOCAL_SEARCH_DEFAULT_ITERS_LARGE: int = 4_000

LOCAL_SEARCH_MOVE_SCALE: float = 0.02
LOCAL_SEARCH_ANGLE_SCALE: float = 2.0
