"""
Simple timing / profiling helpers for the Santa 2025 Christmas Tree Packing project.

These utilities provide lightweight ways to measure execution time for:

- Individual code blocks (context manager).
- Functions (decorator).
- Quick ad-hoc micro-benchmarks.

They are intentionally minimal and have no external dependencies beyond the
standard library, so they can be used in scripts, notebooks, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Optional, Iterable, Dict

import statistics
import time
import functools


# ---------------------------------------------------------------------------
# Context-manager timer
# ---------------------------------------------------------------------------

@dataclass
class Timer:
    """
    Context manager for measuring wall-clock time of a code block.

    Usage
    -----
        from santa2025.utils.timing import Timer

        with Timer("build submission"):
            df = build_submission_df()

    Attributes
    ----------
    name:
        Optional label printed when exiting the context.
    start:
        Start time (perf_counter units).
    end:
        End time (perf_counter units).
    elapsed:
        Duration in seconds (float). Available after the context exits.
    """

    name: Optional[str] = None
    start: float = 0.0
    end: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        label = f"[Timer] {self.name}: " if self.name else "[Timer] "
        print(f"{label}{self.elapsed:.4f} s")


def time_block(name: Optional[str] = None) -> Timer:
    """
    Convenience function to create a `Timer` context manager with a name.

    Example
    -------
        from santa2025.utils.timing import time_block

        with time_block("local search n=100"):
            refine_layout(poses, iters=5000)
    """
    return Timer(name=name)


# ---------------------------------------------------------------------------
# Decorator for timing functions
# ---------------------------------------------------------------------------

def timeit(name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to time a function call and print its duration.

    Usage
    -----
        from santa2025.utils.timing import timeit

        @timeit("build_submission_df")
        def build():
            return build_submission_df()

        df = build()  # prints timing information and returns df
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start
            print(f"[timeit] {label}: {elapsed:.4f} s")
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Simple benchmarking helper
# ---------------------------------------------------------------------------

def benchmark(
    func: Callable[..., Any],
    *args,
    repeats: int = 5,
    warmup: int = 1,
    **kwargs,
) -> Dict[str, float]:
    """
    Run a simple micro-benchmark of `func(*args, **kwargs)`.

    - Runs the function `warmup` times without recording.
    - Then runs it `repeats` times, recording elapsed durations.
    - Returns a dictionary with min/mean/max and number of repeats.

    This is *not* a sophisticated benchmark tool, but is useful for quick,
    apples-to-apples comparisons in notebooks or scripts.

    Example
    -------
        from santa2025.utils.timing import benchmark

        stats = benchmark(build_submission_df, repeats=3)
        print(stats)

    Returns
    -------
    dict with keys:
        - 'min'
        - 'mean'
        - 'max'
        - 'repeats'
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "max": max(times),
        "repeats": float(repeats),
    }


__all__ = [
    "Timer",
    "time_block",
    "timeit",
    "benchmark",
]
