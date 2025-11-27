"""
I/O utilities for the Santa 2025 Christmas Tree Packing project.

This module centralizes common file and path operations so that:
- Scripts and notebooks do *not* hard-code paths.
- Reading/writing submissions is consistent across the project.
- Loading the Kaggle `sample_submission.csv` is one line.

Typical usage from code or notebooks
------------------------------------

    from santa2025.utils.io import (
        ensure_data_dirs,
        load_sample_submission,
        save_submission_df,
        get_raw_path,
        get_submissions_dir,
    )

    ensure_data_dirs()
    df_sample = load_sample_submission()

    # Suppose `submission_df` is a Kaggle-ready DataFrame with columns x, y, deg
    csv_path = save_submission_df(submission_df)
    print("Wrote submission to:", csv_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import datetime as dt
import pandas as pd

from ..config import DATA_DIR, DATA_RAW_DIR, DATA_SUBMISSIONS_DIR
from ..evaluation import evaluate_submission_csv


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_data_dirs() -> None:
    """
    Ensure that the main data directories exist:

    - data/
    - data/raw/
    - data/submissions/

    It is safe to call this repeatedly.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_path(*parts: str) -> Path:
    """
    Build a path under the `data/raw/` directory.

    Example
    -------
    >>> get_raw_path("sample_submission.csv")
    PosixPath('.../data/raw/sample_submission.csv')
    """
    return DATA_RAW_DIR.joinpath(*parts)


def get_submissions_dir() -> Path:
    """
    Return the submissions directory path (`data/submissions/`), creating it
    if needed.
    """
    DATA_SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_SUBMISSIONS_DIR


def get_timestamped_submission_path(
    prefix: str = "submission",
    suffix: str = ".csv",
) -> Path:
    """
    Build a timestamped submission path under `data/submissions/`.

    Example output filename:
        submission_20251126_153045.csv
    """
    get_submissions_dir()
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}{suffix}"
    return DATA_SUBMISSIONS_DIR / filename


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_sample_submission(filename: str = "sample_submission.csv") -> pd.DataFrame:
    """
    Load Kaggle's sample submission from `data/raw/`.

    Parameters
    ----------
    filename:
        Name of the sample submission file, default 'sample_submission.csv'.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame as read by `pandas.read_csv`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = get_raw_path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Sample submission not found at {path}. "
            f"Place Kaggle's '{filename}' under data/raw/."
        )
    return pd.read_csv(path)


def load_submission_csv(path: PathLike) -> pd.DataFrame:
    """
    Load an arbitrary submission CSV from disk as a DataFrame.

    Parameters
    ----------
    path:
        Path to a CSV file.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_submission_df(
    submission_df: pd.DataFrame,
    path: Optional[PathLike] = None,
    index_label: str = "id",
    prefix: str = "submission",
) -> Path:
    """
    Save a Kaggle-ready submission DataFrame to CSV.

    The DataFrame is expected to either:
    - Have an index containing the 'id' values and no 'id' column, or
    - Have an 'id' column and a default RangeIndex (in which case we leave
      the column in place and do *not* write the index).

    Parameters
    ----------
    submission_df:
        DataFrame with at least columns x, y, deg, formatted as strings
        with an 's' prefix (competition format).
    path:
        Optional explicit output path. If None, a timestamped filename is
        created under `data/submissions/` via `get_timestamped_submission_path`.
    index_label:
        Label to use for the index column when writing to CSV, if the index
        encodes the 'id'. Default is 'id'.
    prefix:
        Filename prefix when generating a timestamped path.

    Returns
    -------
    Path
        The path to the written CSV.
    """
    if path is None:
        out_path = get_timestamped_submission_path(prefix=prefix)
    else:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide whether to write the index as 'id' or rely on an 'id' column.
    if "id" in submission_df.columns and not isinstance(submission_df.index, pd.MultiIndex):
        # We assume 'id' is a regular column; keep it and do not write index.
        submission_df.to_csv(out_path, index=False)
    else:
        # Use the index as the 'id' column.
        submission_df.to_csv(out_path, index_label=index_label)

    return out_path


# ---------------------------------------------------------------------------
# Convenience: evaluate a saved submission
# ---------------------------------------------------------------------------

def evaluate_saved_submission(path: PathLike, strict: bool = True) -> float:
    """
    Evaluate a saved submission CSV using the project's local metric.

    Parameters
    ----------
    path:
        Path to the CSV to evaluate.
    strict:
        Passed through to `evaluate_submission_csv`.

    Returns
    -------
    float
        The total score as computed by the local metric.
    """
    score_table, total_score = evaluate_submission_csv(path, strict=strict)
    # You can print or inspect `score_table` in a notebook as needed.
    return total_score


__all__ = [
    "ensure_data_dirs",
    "get_raw_path",
    "get_submissions_dir",
    "get_timestamped_submission_path",
    "load_sample_submission",
    "load_submission_csv",
    "save_submission_df",
    "evaluate_saved_submission",
]
