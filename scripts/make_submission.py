#!/usr/bin/env python
"""
CLI helper to build and save a competition submission CSV.

This script is a thin wrapper around the library entry points:

- santa2025.submission.write_submission_csv
- santa2025.utils.io.evaluate_saved_submission  (optional)

Typical usage from the project root
-----------------------------------

    python scripts/make_submission.py
    # or
    python scripts/make_submission.py --seed 123
    python scripts/make_submission.py --no-local-search
    python scripts/make_submission.py --output data/submissions/my_sub.csv
    python scripts/make_submission.py --evaluate

The script automatically adds `src/` to PYTHONPATH so that it can import the
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

    Assumes this file lives in <project_root>/scripts/make_submission.py.
    """
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Kaggle submission CSV for the Santa 2025 tree packing challenge.",
    )
    parser.add_argument(
        "--no-local-search",
        action="store_true",
        help="Disable local search refinement; use pure hex lattice baseline.",
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
            "Optional output path for the CSV. "
            "If omitted, a timestamped name will be created under data/submissions/."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After writing the submission, evaluate it with the local metric and print the score.",
    )
    parser.add_argument(
        "--non-strict-eval",
        action="store_true",
        help="Use non-strict validation when evaluating (skip some submission checks).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    project_root = _ensure_src_on_path()

    # Imports done after path configuration
    from santa2025.submission import write_submission_csv
    from santa2025.utils.io import evaluate_saved_submission

    args = _parse_args(argv)

    use_local_search = not args.no_local_search
    seed = args.seed

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    print(f"[make_submission] Project root: {project_root}")
    print(f"[make_submission] Local search: {'ON' if use_local_search else 'OFF'}")
    if seed is not None:
        print(f"[make_submission] Seed: {seed}")

    csv_path = write_submission_csv(
        output_path=output_path,
        use_local_search=use_local_search,
        seed=seed,
    )
    print(f"[make_submission] Submission written to: {csv_path}")

    if args.evaluate:
        strict = not args.non_strict_eval
        print(f"[make_submission] Evaluating submission (strict={strict})...")
        score = evaluate_saved_submission(csv_path, strict=strict)
        print(f"[make_submission] Local metric score: {score:.9f}")


if __name__ == "__main__":
    main()
