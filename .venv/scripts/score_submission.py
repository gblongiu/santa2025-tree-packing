#!/usr/bin/env python
"""
CLI tool to evaluate a submission CSV using the local metric.

This script is a thin wrapper around the library metric:

- santa2025.evaluation.evaluate_submission_csv

Typical usage from the project root
-----------------------------------

    python scripts/score_submission.py data/submissions/my_sub.csv
    python scripts/score_submission.py data/submissions/my_sub.csv --non-strict
    python scripts/score_submission.py data/submissions/my_sub.csv --show-table

It automatically adds `src/` to PYTHONPATH so that it can import the
`santa2025` package without requiring installation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, List


def _ensure_src_on_path() -> Path:
    """
    Ensure that <project_root>/src is on sys.path and return project_root.

    Assumes this file lives in <project_root>/scripts/score_submission.py.
    """
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Santa 2025 tree packing submission CSV with the local metric.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the submission CSV file.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Disable strict validation (skip some submission sanity checks).",
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="Print the per-n score table in addition to the total score.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    project_root = _ensure_src_on_path()

    # Import after path configuration
    from santa2025.evaluation import evaluate_submission_csv

    args = _parse_args(argv)
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise SystemExit(f"[score_submission] CSV not found: {csv_path}")

    strict = not args.non_strict

    print(f"[score_submission] Project root: {project_root}")
    print(f"[score_submission] Evaluating: {csv_path}")
    print(f"[score_submission] Strict mode: {strict}")

    score_table, total_score = evaluate_submission_csv(csv_path, strict=strict)

    if args.show_table:
        # Show a compact view of the table
        with pd.option_context("display.max_rows", None, "display.float_format", "{:.9f}".format):
            print("[score_submission] Per-n score table:")
            print(score_table)

    print(f"[score_submission] Total score: {total_score:.9f}")


if __name__ == "__main__":
    # Lazy import pandas only if we run as a script (needed for pretty printing).
    import pandas as pd  # type: ignore[import-not-found]
    main()
