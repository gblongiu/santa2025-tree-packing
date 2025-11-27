"""
Tests for the local metric implementation in santa2025.evaluation.

Goals
-----
- Ensure basic helpers (decode_val, parse_submission_df) behave as expected.
- Verify that metric_from_df and evaluate_submission_csv are internally
  consistent (same total score for the same input).
- Optionally validate behaviour against Kaggle's sample_submission.csv if it
  is available under data/raw/.

These tests are designed so that they **skip gracefully** if the sample
submission file is not present, rather than failing the entire test suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from santa2025.evaluation import (
    decode_val,
    parse_submission_df,
    metric_from_df,
    evaluate_submission_csv,
)
from santa2025.config import (
    N_MAX_TREES,
    COORD_MIN,
    COORD_MAX,
    DATA_RAW_DIR,
)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("s0.0", 0.0),
        ("s1.234567", 1.234567),
        ("-3.5", -3.5),
        (2, 2.0),
        (2.5, 2.5),
    ],
)
def test_decode_val_basic(raw, expected):
    assert decode_val(raw) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_parse_submission_df_adds_n_and_idx():
    df = pd.DataFrame(
        {
            "id": ["001_0", "001_1", "002_0", "002_1"],
            "x": ["s0.0", "s1.0", "s0.5", "s-0.5"],
            "y": ["s0.0", "s0.0", "s1.0", "s-1.0"],
            "deg": ["s0.0", "s10.0", "s20.0", "s30.0"],
        }
    )
    parsed = parse_submission_df(df, strict=False)

    assert set(["id", "x", "y", "deg", "n", "idx"]).issubset(parsed.columns)
    assert parsed["n"].tolist() == [1, 1, 2, 2]
    assert parsed["idx"].tolist() == [0, 1, 0, 1]

    # Values should be decoded to floats
    assert parsed["x"].dtype == float
    assert parsed["y"].dtype == float
    assert parsed["deg"].dtype == float


def test_parse_submission_df_strict_validates_counts_and_bounds():
    # Build a tiny valid submission for n=1..3
    ids = []
    xs = []
    ys = []
    degs = []
    for n in range(1, 4):
        for idx in range(n):
            ids.append(f"{n:03d}_{idx}")
            xs.append("s0.0")
            ys.append("s0.0")
            degs.append("s0.0")

    df = pd.DataFrame({"id": ids, "x": xs, "y": ys, "deg": degs})
    parsed = parse_submission_df(df, strict=True)

    # Check ranges and group sizes
    assert parsed["n"].min() == 1
    assert parsed["n"].max() == 3

    counts = parsed.groupby("n")["idx"].count().to_dict()
    assert counts == {1: 1, 2: 2, 3: 3}

    assert COORD_MIN <= parsed["x"].min() <= COORD_MAX
    assert COORD_MIN <= parsed["x"].max() <= COORD_MAX
    assert COORD_MIN <= parsed["y"].min() <= COORD_MAX
    assert COORD_MIN <= parsed["y"].max() <= COORD_MAX


def test_parse_submission_df_strict_rejects_bad_counts():
    # For n=2 we only provide 1 row -> should raise
    df = pd.DataFrame(
        {
            "id": ["001_0", "002_0"],
            "x": ["s0.0", "s0.0"],
            "y": ["s0.0", "s0.0"],
            "deg": ["s0.0", "s0.0"],
        }
    )
    with pytest.raises(ValueError):
        parse_submission_df(df, strict=True)


def test_parse_submission_df_strict_rejects_out_of_bounds_coords():
    df = pd.DataFrame(
        {
            "id": ["001_0"],
            "x": ["s1000.0"],  # definitely outside [-100, 100]
            "y": ["s0.0"],
            "deg": ["s0.0"],
        }
    )
    with pytest.raises(ValueError):
        parse_submission_df(df, strict=True)


# ---------------------------------------------------------------------------
# Integration tests with sample_submission.csv (if available)
# ---------------------------------------------------------------------------

def _get_sample_submission_path() -> Path:
    """
    Helper to locate the sample_submission.csv if present.

    By convention we expect it to live under data/raw/ as configured by
    santa2025.config.DATA_RAW_DIR. If not found, tests that depend on it
    will be skipped.
    """
    path = DATA_RAW_DIR / "sample_submission.csv"
    return path


@pytest.mark.integration
def test_metric_consistency_between_api_helpers():
    """
    If sample_submission.csv is available, ensure that:

    - metric_from_df(pd.read_csv(...)) and evaluate_submission_csv(path)
      produce compatible score tables and identical total scores.
    """
    path = _get_sample_submission_path()
    if not path.exists():
        pytest.skip(f"sample_submission.csv not found at {path}, skipping integration test.")

    raw_df = pd.read_csv(path)
    table_from_df = metric_from_df(raw_df, strict=True)
    table_from_csv, total_from_csv = evaluate_submission_csv(path, strict=True)

    # Total scores should match
    total_from_df = float(table_from_df["score_n"].sum())
    assert total_from_df == pytest.approx(total_from_csv, rel=1e-9, abs=1e-12)

    # Per-n side lengths should match
    merged = table_from_df.merge(table_from_csv, on="n", suffixes=("_df", "_csv"))
    assert len(merged) == len(table_from_df)
    assert np.allclose(merged["side_df"], merged["side_csv"], rtol=1e-9, atol=1e-12)


@pytest.mark.integration
def test_metric_handles_full_n_range_in_sample_if_present():
    """
    If the sample submission contains all n=1..N_MAX_TREES, verify that
    the metric table covers the same range with no gaps.
    """
    path = _get_sample_submission_path()
    if not path.exists():
        pytest.skip(f"sample_submission.csv not found at {path}, skipping integration test.")

    raw_df = pd.read_csv(path)
    table = metric_from_df(raw_df, strict=True)

    unique_ns = sorted(table["n"].unique().tolist())
    # Some samples might not include all 1..N_MAX_TREES, so we only assert
    # that the metric does not skip any n present in the submission.
    assert unique_ns == sorted(unique_ns)
    assert min(unique_ns) >= 1
    assert max(unique_ns) <= N_MAX_TREES
